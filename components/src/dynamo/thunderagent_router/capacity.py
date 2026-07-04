# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-worker capacity snapshot derived from worker model deployment cards.

``capacity_tokens`` (``block_size * total_kv_blocks``) is published once per
worker via its MDC. We piggyback on ``FpmEventSubscriber`` only as the
existing Python channel that already tracks per-worker MDCs; the
forward-pass-metric payloads themselves are not consumed.

HiCache awareness
-----------------
``total_kv_blocks`` in the MDC reflects only the GPU (L1) KV pool. When the
SGLang worker runs with hierarchical caching (``--enable-hierarchical-cache``),
cold KV blocks are transparently offloaded to a host-memory (L2) pool and,
optionally, distributed storage (L3), so the *effective* KV capacity a worker
can serve without evicting a running program is larger than the GPU pool alone.

The ThunderAgent scheduler pauses/demotes programs once a worker's estimated
token usage crosses ``pause_threshold * capacity``. If ``capacity`` only counts
the GPU pool, the scheduler pauses programs that HiCache could otherwise keep
resident by offloading — the two mechanisms then fight (reduce-concurrency vs.
expand-capacity). To let them cooperate, ``WorkerCapacityProvider`` accepts a
``hicache_ratio`` (host-pool-size / device-pool-size, mirroring SGLang's
``--hicache-ratio``) and reports the HiCache-aware effective capacity:

    effective_capacity = block_size * total_kv_blocks * (1 + hicache_ratio)

``hicache_ratio=0.0`` (the default) preserves the original GPU-only behavior, so
existing deployments are unaffected.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from dynamo.llm import FpmEventSubscriber
from dynamo.runtime import Endpoint

logger = logging.getLogger(__name__)


class WorkerCapacityProvider:
    """Maps ``worker_id -> effective_kv_pool_tokens`` from each worker's MDC.

    With ``hicache_ratio > 0`` the reported capacity includes the HiCache host
    (L2) pool so the scheduler does not pause programs that HiCache can keep
    resident by offloading.
    """

    def __init__(self, endpoint: Endpoint, hicache_ratio: float = 0.0) -> None:
        self._endpoint = endpoint
        self._subscriber: Optional[FpmEventSubscriber] = None
        # host_pool_tokens / device_pool_tokens; 0.0 == GPU-only (original behavior).
        self._hicache_ratio = max(0.0, float(hicache_ratio))
        # Cache parsed cards keyed on the raw JSON string so a subsequent
        # snapshot() call avoids re-parsing on the request hot path.
        self._parsed: dict[str, Optional[int]] = {}

    def start(self) -> None:
        if self._subscriber is not None:
            return
        self._subscriber = FpmEventSubscriber(self._endpoint)
        self._subscriber.start_tracking()
        logger.info(
            "WorkerCapacityProvider: subscribed to MDC stream (hicache_ratio=%.2f%s)",
            self._hicache_ratio,
            ", HiCache-aware capacity enabled" if self._hicache_ratio > 0 else "",
        )

    def stop(self) -> None:
        if self._subscriber is None:
            return
        try:
            self._subscriber.shutdown()
        except Exception as exc:
            logger.warning("WorkerCapacityProvider shutdown error: %s", exc)
        self._subscriber = None

    def snapshot(self) -> dict[int, int]:
        if self._subscriber is None:
            return {}
        try:
            cards = self._subscriber.get_model_cards()
        except Exception as exc:
            logger.debug("WorkerCapacityProvider snapshot error: %s", exc)
            return {}

        out: dict[int, int] = {}
        for worker_id_str, card_json in cards.items():
            try:
                worker_id = int(worker_id_str)
            except (ValueError, TypeError):
                continue
            pool_tokens = self._parse_pool_tokens(card_json)
            if pool_tokens is not None:
                out[worker_id] = pool_tokens
        return out

    def _parse_pool_tokens(self, card_json: str) -> Optional[int]:
        # Cache key folds in hicache_ratio so a ratio change (new provider) never
        # returns a stale GPU-only value from a previous instance.
        cache_key = card_json
        if cache_key in self._parsed:
            return self._parsed[cache_key]
        result: Optional[int] = None
        try:
            card = json.loads(card_json)
        except json.JSONDecodeError:
            card = None
        if isinstance(card, dict):
            block_size = card.get("kv_cache_block_size")
            total_blocks = (card.get("runtime_config") or {}).get("total_kv_blocks")
            if (
                isinstance(block_size, (int, float))
                and block_size > 0
                and isinstance(total_blocks, (int, float))
                and total_blocks > 0
            ):
                gpu_tokens = int(block_size) * int(total_blocks)
                # HiCache-aware effective capacity: GPU (L1) pool plus the host
                # (L2) pool, which SGLang sizes as device_pool * hicache_ratio.
                result = int(gpu_tokens * (1.0 + self._hicache_ratio))
        self._parsed[cache_key] = result
        return result
