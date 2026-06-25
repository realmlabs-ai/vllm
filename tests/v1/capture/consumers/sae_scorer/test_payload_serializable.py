# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SAE-scorer payload must cross the engine→driver boundary.

``CaptureResult`` rides an ``EngineCoreOutput`` msgspec.Struct, so its
``payload`` must be msgspec-serializable. The SAE scorer returns plain
lists/dicts/ints/floats, which msgpack handles; this test guards against a
regression (e.g. someone returning a raw tensor) that would break the offline
``LLM.generate`` return path.
"""

from __future__ import annotations

import msgspec

from vllm.v1.capture.types import CaptureResult, VllmInternalRequestId


def _sample_payload() -> dict:
    return {
        "layer": 20,
        "hook": "post_mlp",
        "sae_size": 32768,
        "num_rows": 2,
        "indices": [[3, 17, 999], [42]],
        "values": [[0.5, 1.25, 0.1], [2.0]],
    }


def test_capture_result_with_sae_payload_roundtrips():
    result = CaptureResult(
        key=(VllmInternalRequestId("r1"), 20, "post_mlp"),
        status="ok",
        payload=_sample_payload(),
    )

    encoded = msgspec.msgpack.encode(result)
    decoded = msgspec.msgpack.decode(encoded, type=CaptureResult)

    assert decoded.status == "ok"
    assert decoded.payload == _sample_payload()
    # key tuple survives (request id, layer, hook).
    assert tuple(decoded.key) == ("r1", 20, "post_mlp")


def test_payload_inside_engine_core_output_shape():
    """Mirror the real nesting: dict[str, CaptureResult] keyed by consumer."""
    results = {
        "sae_scorer": CaptureResult(
            key=(VllmInternalRequestId("r1"), 20, "post_mlp"),
            status="ok",
            payload=_sample_payload(),
        )
    }
    encoded = msgspec.msgpack.encode(results)
    decoded = msgspec.msgpack.decode(encoded, type=dict[str, CaptureResult])
    assert decoded["sae_scorer"].payload["sae_size"] == 32768
