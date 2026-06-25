# SAE scorer — egress / delivery design notes

How captured results (SAE scores, and by extension raw hidden states) get **out
of the engine and back to a caller**. This is the part of the design that is
*not* about computing scores — the scoring (device-resident SAE encode on the
GPU, no D2H) already works. The open question is the delivery channel, and this
file records the options, the trade-offs, and the caveats we evaluated so they
aren't lost.

**Decision: implement solution 2 (write-to-store + handle).** It is the
reliable, general path and the only sane channel for large artifacts. The other
options below are documented for context and possible follow-up.

---

## The delivery race (why `out.capture_results` is unreliable today)

Capture finalize for a request is triggered on the step **after** it finishes
(`gpu_model_runner.py` processes `scheduler_output.finished_req_ids`), but:

- The scheduler emits an `EngineCoreOutput` (which carries
  `capture_results.get(req_id)`, `scheduler.py:1524`) **only for requests
  scheduled that step**.
- A finished request is **freed** in the same step it finishes
  (`scheduler.py:_free_request`), and its final output already went out without
  capture results.
- So the finalize result (ready one step later) has no output to ride on. It is
  dropped — for **every** request, not just the last.
- For the **last / only** request it is worse: the finalize step never runs at
  all (the engine goes idle), so finalize doesn't even fire. (Verified: the
  request stays in `mgr._requests`, chunks sit in the consumer's `_pending`, no
  result is produced.)

This is exactly what the (reverted) `feat/capture-wait` branch was built to fix.

The synchronous `CaptureManager.finalize_request(req_id)` (used by the runner
unit tests) does work in-process — that is how the e2e retrieves results today.

---

## The three options we compared

### Option 1 — CPU consumer
Run the SAE on the host tensor after the framework's D2H. Rejected: pays the full
dense D2H *and* runs the matmul on CPU. Only viable for tiny/offline SAEs.

### Option 2 — write to a store + return a handle  ✅ implemented
The consumer writes its result to disk (a store) as a side effect, at a
**deterministic per-request path**, and returns only a small **handle**. The
on-disk file is the source of truth; the client reads it by request id and does
not depend on `capture_results` delivery at all.

- **Performance:** egress is off the model critical path (finalize thread). For
  small sparse scores the write is negligible; for large hidden states the
  store is the *only* sane channel anyway.
- **Generalises:** the same mechanism serves small SAE scores and large raw
  hidden states (the existing `FilesystemConsumer` already streams tensors to
  disk under `{root}/{tag_slug}/{request_id_slug}/{layer}_{hook}.bin`).
- **Dependency:** still needs the worker-side **finalize to fire** (that is what
  runs `on_capture` / seals files). In a busy server another request keeps the
  engine stepping, so finalize fires normally; on a drained/last request it may
  not — mitigated by the early-finalize follow-up below, or by the fact that for
  hidden states the raw bytes are written incrementally during dispatch and only
  the seal/index waits on finalize.

### Option 3 — capture-wait (inline in `out.capture_results`)
Hold the finished request's `RequestOutput` until capture finalizes, deliver the
result inline. Gives the clean `out = llm.generate(...); out[0].capture_results`
API. Not implemented — see surgery below.

---

## "Two round trips?" / the handle pattern

For option 2, getting tokens and getting scores are separate steps, but:

- At the **vLLM** boundary: `generate()` is one call; reading the score file is a
  separate read — a **local file read**, not a network call, when the store is
  local disk.
- At the **UI** boundary: a front service (e.g. `ml_service`) can hide this —
  after `generate()` it reads the store and returns `{tokens, scores}` in **one**
  response. The UI sees one call.

The clean shape is **trigger + collect**: `generate()` returns tokens + a handle
(request id / path); collect-from-store is a cheap separate read by whoever needs
the data. The path is deterministic (`score_path(out_dir, request_id, layer,
hook)`), so a client can find the file **without** the handle — retrieval never
depends on `capture_results` delivery.

---

## Hidden states (the API must support these too)

Hidden states are large (`[tokens, 4096]` bf16 ≈ 8 KB/token; MBs across
layers/positions). They **cannot** be inlined in a `generate()` response, so:

- Option 3 (inline) **cannot** serve hidden states at all.
- Option 2 is the natural and only sane channel — use the existing
  `FilesystemConsumer` (no new code). Per-request consumer selection
  (`SamplingParams.capture[name]`, `reads_client_spec`) lets a request ask for
  `sae_scores` (this consumer) and/or `hidden_states` (filesystem) without
  separate endpoints.

So the hidden-states requirement **settles the design toward option 2** as the
backbone; option 3 would only ever be inline sugar for the small-scores case.

---

## Streaming

The **handle decouples capture from the token stream**, which is why option 2
fits streaming cleanly:

1. The handle (request id / path) is known at admission → send it in the first
   SSE event or a header, before any token. No race.
2. Tokens stream normally, never blocked by capture.
3. Data lands in the store asynchronously: prompt-position captures ~after
   prefill; generated-token captures sealed at finalize.
4. The client fetches via the handle out of band (small scores can also be
   inlined in the final SSE chunk if ready; large hidden states are downloaded).

Option 3 in streaming would have to delay the final SSE chunk until finalize and
**cannot** carry hidden states — so option 2 is the only coherent streaming
design.

### vLLM is already chunked — per-token score streaming
Decode is one-token-per-step-per-request (continuous batching) and capture runs
**per step** (`on_hook` + `dispatch_step_captures` every step). A **per-step**
result rides the *same* step's token output while the request is still scheduled
(`scheduler.py:1524`), so it **sidesteps the delivery race entirely** — that race
is specifically about the *terminal/whole-request* result computed after free.

Per-token score streaming is therefore the **least-surgical inline mode**:
- Score per step (a direct `CaptureSink` that encodes in `submit_chunk`) instead
  of accumulating to finalize via `_BatchedAdapter`.
- Have the runner populate `model_runner_output.capture_results[req_id]` each
  step; the scheduler + streaming path already deliver it.

Caveats: prompt tokens emit no per-token delta (prefill), so prompt scores attach
to the *first generated token's* chunk or go via the store; per-token **hidden
states** are heavy (~8 KB/token) → prefer the store.

---

## Option 3 surgery (if/when whole-request inline delivery is wanted)

Four touch points; a precedent exists for three (the KV-connector async-finish
path: `delay_free_blocks` + `finished_recving_kv_req_ids` + the engine continuing
to step):

1. **Defer freeing** a finished request with pending capture (new tracking set).
   *Template:* `scheduler.py:1838-1847`.
2. **Keep the engine stepping** — `has_unfinished_requests()` /
   `get_num_unfinished_requests()` must count pending-capture requests
   (`scheduler.py:1881`, gated at `core.py:619/647`).
3. **Worker→scheduler signal** of "capture finalized for req X" on a later
   `ModelRunnerOutput`; scheduler emits a trailing `EngineCoreOutput` with the
   results, then frees. *Template:* `KVConnectorOutput.finished_recving` →
   `_update_from_kv_xfer_finished` (`scheduler.py:2145`).
4. **Hold the client output** — for non-streaming `generate()` the final
   `RequestOutput` must be buffered until capture arrives and merged
   (`output_processor.py:662`). **This is the new/risky part** — KV transfer
   delays *freeing*, not *client output* — and is where `feat/capture-wait` got
   tricky.

Scope: ~3 files + a new output field. Medium surgery; risk concentrated in #4.

---

## Early finalize — IMPLEMENTED (makes store-mode finalize fire automatically)

**Finalize-on-data-complete** instead of finalize-on-request-finish. For
prompt-bounded captures (`last_prompt` / `all_prompt` / explicit prompt
positions) the data is complete after prefill, while the request is still alive
and scheduled — so the manager finalizes the request then, rather than waiting
for the post-finish step (which never runs for a drained/last request).

Implementation:
- `manager.py` — `_RequestCaptureState.prompt_bounded` (computed at
  `register_request` via `_is_prompt_bounded`); `build_step_plan` flags a request
  in `_prompt_complete` once `step_end >= num_prompt_tokens`;
  `take_prompt_complete_requests()` drains the flags (filtered to still-registered
  requests).
- `gpu_model_runner.py` — `_finalize_capture_step()` finalizes those requests
  right after dispatch. The post-finish path stays as a fallback; re-finalizing
  an already-popped request is a no-op (`finalize_request_async` returns `False`).

No scheduler / output-processor surgery.

**What this guarantees (verified e2e):** in **store mode** the score file is
written **during generation** with no manual finalize and no dependence on the
trailing step — this is the reliable solution-2 contract.

**What it does NOT make reliable:** the *inline* `out.capture_results` handle.
Even after early-finalize, the result is produced on the finalize thread and only
rides a later decode step's output if it lands in time (and under async
scheduling the timing is not guaranteed) — so inline delivery stays best-effort.
Treat the **store file** as the source of truth; read the handle from
`capture_results` when present, else read the file by path.

Limitations:
- Covers only captures that complete before the request finishes (prompt
  positions). `all` / `all_generated` still finalize on finish (option-3 territory
  for reliable inline delivery).
- The store path is keyed by the **internal** request id (`key[0]`), not the
  client-facing `RequestOutput.request_id`. A client that needs to locate the
  file without the handle should either read the handle from `capture_results`
  (when delivered) or scan `{out_dir}/*/L{layer}_{hook}.json`. Keying by a
  client-supplied id would need that id plumbed into the consumer.

---

## Quick map

| Want | Consumer | Channel | Status |
|---|---|---|---|
| SAE scores (small), reliable | `sae_scorer` store mode (`out_dir`) | disk (+ handle when delivered) | done ✅ — written during generation via early finalize |
| SAE scores inline from `generate()` (prompt) | `sae_scorer` inline mode | `capture_results` | best-effort — early finalize helps but inline timing not guaranteed |
| Per-token scores streamed with tokens | direct-sink per-step variant | per-step `capture_results` | not built — low surgery (runner + consumer) |
| Raw hidden states (any size) | `filesystem` | disk + handle | done ✅ (existing consumer) |
| Whole-request scores inline, incl. generated tokens | any | `capture_results` | not built — capture-wait (medium, risk in output hold) |
