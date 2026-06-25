# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SAE scoring capture consumer (device-resident)."""

from vllm.v1.capture.consumers.sae_scorer.consumer import (
    SaeScorerConsumer,
    score_path,
)

__all__ = ["SaeScorerConsumer", "score_path"]
