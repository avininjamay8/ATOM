# SPDX-License-Identifier: MIT
"""Exercise producer block ownership with real scheduling and offload frontiers.

GPU forwards and transfer completion times are simulated. Metadata construction,
MultiConnector routing, PP quorum, idle draining and block release are real.
"""

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
from aiter_stub import stubbed_aiter
from conftest import MockConfig

from atom.kv_transfer.disaggregation.multi.multi_connector import (
    MultiConnector,
    MultiConnectorScheduler,
)
from atom.kv_transfer.disaggregation.types import ConnectorMetadata, KVConnectorOutput
from atom.kv_transfer.offload.dense.connector import DenseOffloadScheduler
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

with stubbed_aiter():
    from atom.model_engine.engine_core import EngineCore
    from atom.model_engine.pp_engine_core import PPEngineCoreProc


class _Transfer:
    def __init__(self, *, producer=False):
        self.is_producer = producer
        self.pending = deque()
        self.saved = []
        self.output = KVConnectorOutput()

    def start_load_kv(self, metadata):
        for req in getattr(metadata, "requests", ()):
            if req.save_spec is not None:
                self.pending.append(req)

    def get_finished(self):
        result, self.output = self.output, KVConnectorOutput()
        return result

    def complete_save(self):
        req = self.pending.popleft()
        self.saved.append((req.save_spec.skip_leading_tokens, len(req.token_ids)))
        self.output.finished_saving.add(req.save_operation)
        return req.save_operation


def _offload():
    # Supply backend-independent scheduler state without starting LMCache IPC.
    off = DenseOffloadScheduler.__new__(DenseOffloadScheduler)
    off._init_offload_statistics()
    off._do_save, off._do_load = True, False
    off.block_size, off.chunk_size = 16, 64
    off._lookup_client = None
    off._save_rr_last = None
    off._save_nonce = off._load_nonce = 0
    for name in (
        "_save_tracker",
        "_save_inflight",
        "_load_specs",
        "_reqs_need_recv",
        "_load_save_floors",
        "_hit_save_floors",
        "_load_lifecycles",
        "_active_load_operations",
    ):
        setattr(off, name, {})
    off._lookup_in_step = []
    off._handoff_loads = set()
    return off


class _Lifecycle:
    def __init__(self, pp_size, prompt=192):
        self.off = _offload()
        self.sched = Scheduler(
            MockConfig(
                pipeline_parallel_size=pp_size,
                kv_cache_block_size=16,
                num_kvcache_blocks=128,
                max_model_len=1024,
                max_num_batched_tokens=64,
            )
        )
        producer = SimpleNamespace(
            is_producer=True,
            get_num_new_matched_tokens=lambda seq: (0, False),
            update_state_after_alloc=lambda seq: None,
            build_connector_meta=ConnectorMetadata,
            request_finished=lambda seq: None,
        )
        composite = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
        composite._connectors = [producer, self.off]
        composite.is_producer = composite.is_offload = True
        composite._load_winner = {}
        self.sched.kv_connector = composite
        self.seq = Sequence(
            list(range(prompt)),
            sampling_params=SamplingParams(max_tokens=1),
            block_size=16,
        )
        self.sched.add(self.seq)
        self.workers = []
        self.savers = []
        for _ in range(pp_size):
            sender, saver = _Transfer(producer=True), _Transfer()
            worker = MultiConnector.__new__(MultiConnector)
            worker._connectors = [sender, saver]
            worker.is_producer = True
            self.workers.append(worker)
            self.savers.append(saver)
        cls = PPEngineCoreProc if pp_size > 1 else EngineCore
        self.engine = cls.__new__(cls)
        self.engine.kv_transfer_enabled = True
        self.engine.scheduler = self.sched
        self.engine._next_idle_kv_drain = 0
        self.engine.runner_mgr = SimpleNamespace(
            call_func_with_aggregation=lambda name: self.workers[0].get_finished(),
            call_func=lambda name, meta=None, **kwargs: (
                self.workers[0].start_load_kv(meta)
                if name == "process_kvconnector_output"
                else None
            ),
        )
        if pp_size > 1:
            self.engine.pp_size = pp_size
            self.engine._pp_kv_aggregator = None
            self.engine._in_flight = deque()
            self.engine.pp_transport = SimpleNamespace(
                recv_kv_status=lambda timeout_ms: [
                    (rank, worker.get_finished())
                    for rank, worker in enumerate(self.workers[1:], 1)
                ],
                send_metadata=lambda batch: [
                    worker.start_load_kv(batch.connector_meta_output)
                    for worker in self.workers[1:]
                ],
            )

    def step(self):
        batch, seqs = self.sched.schedule()
        for worker in self.workers:
            worker.start_load_kv(batch.connector_meta_output)
        deferred_prefill = (
            not self.sched.advance_on_schedule
            and batch.total_seqs_num_prefill > 0
            and batch.produces_output()
        )
        if not self.sched.advance_on_schedule or batch.produces_output():
            self.sched.postprocess(
                list(seqs.values()),
                ScheduledBatchOutput(
                    req_ids=[] if deferred_prefill else list(batch.req_ids),
                    token_ids=[] if deferred_prefill else [(42,)],
                    num_rejected=np.zeros(1, dtype=np.int32),
                    num_bonus=np.zeros(1, dtype=np.int32),
                    draft_token_ids=None,
                    is_deferred_out=not self.sched.advance_on_schedule,
                ),
                batch=batch,
            )
        if deferred_prefill:
            # Non-PP returns the first generated token on the following step.
            self.step()
        return batch

    def poll(self):
        self.engine._poll_kv_transfer_progress()

    def complete_send(self):
        self.workers[0]._connectors[0].output.finished_sending.add(self.seq.id)
        self.poll()

    def complete_save(self):
        operations = [saver.complete_save() for saver in self.savers]
        assert len(set(operations)) == 1
        self.poll()
        return operations[0]

    def idle(self):
        self.engine._next_idle_kv_drain = 0
        assert not self.sched.running and not self.sched.waiting
        assert not self.sched.is_finished()  # deferred blocks keep it active
        if self.sched.advance_on_schedule:
            self.engine._pp_head_step()
        else:
            self.engine._process_engine_step_inner()

    def assert_freed(self):
        assert not self.seq.block_table
        assert not self.sched.deferred_free_blocks
        assert not self.off._save_tracker
        assert not self.engine.has_pending_kv_work()


@pytest.mark.parametrize("pp_size", [1, 4])
@pytest.mark.parametrize("send_first", [False, True])
def test_chunked_prefill_saves_every_chunk_before_releasing(pp_size, send_first):
    run = _Lifecycle(pp_size)
    run.step()  # compute [0,64)
    run.step()  # save [0,64), compute [64,128)
    run.complete_save()  # must reach scheduler before the final P/D send
    assert str(run.seq.id) not in run.off._save_inflight
    run.step()  # save [64,128), compute [128,192), finish request
    assert run.seq._awaiting_kv_send
    assert run.seq.id in run.sched.deferred_free_blocks
    assert not run.sched.running
    if send_first:
        run.complete_send()
        assert run.seq.block_table  # second save is still reading
    run.complete_save()
    assert run.seq.block_table  # final suffix has not even been dispatched
    assert not any(saver.pending for saver in run.savers)
    run.idle()  # last request: idle drain must dispatch [128,192)
    assert all(saver.pending for saver in run.savers)
    run.complete_save()
    if not send_first:
        assert run.seq.block_table  # all saves done, but RDMA still owns it
        run.complete_send()
    assert all(saver.saved == [(0, 64), (64, 128), (128, 192)] for saver in run.savers)
    run.assert_freed()


@pytest.mark.parametrize("pp_size", [1, 4])
def test_send_before_first_save_completion_preserves_undispatched_suffix(pp_size):
    run = _Lifecycle(pp_size)
    run.step()
    run.step()
    run.step()  # first save still pending while the final chunk computes
    run.complete_send()
    assert run.seq.block_table
    run.complete_save()
    assert run.seq.block_table
    run.idle()
    run.complete_save()
    assert all(saver.saved == [(0, 64), (64, 192)] for saver in run.savers)
    run.assert_freed()


def test_send_cannot_free_blocks_before_last_pp_stage_finishes_save():
    run = _Lifecycle(4)
    run.step()
    run.step()
    run.step()
    run.complete_send()
    run.complete_save()
    run.idle()
    # Three stages complete the suffix; the fourth is still reading GPU KV.
    for saver in run.savers[:3]:
        saver.complete_save()
    run.poll()
    assert run.seq.block_table
    assert run.engine.has_pending_kv_work()
    run.savers[3].complete_save()
    run.poll()
    run.assert_freed()


@pytest.mark.parametrize("pp_size", [1, 4])
def test_old_save_completion_does_not_release_a_new_save(pp_size):
    run = _Lifecycle(pp_size)
    run.step()
    run.step()
    old_operation = run.complete_save()
    run.step()
    run.complete_send()
    for saver in run.savers:
        saver.output.finished_saving.add(old_operation)
    run.poll()
    assert run.seq.block_table
    run.complete_save()
    run.idle()
    run.complete_save()
    run.assert_freed()


@pytest.mark.parametrize("pp_size", [1, 4])
def test_prompt_below_offload_chunk_releases_on_send_without_a_save(pp_size):
    run = _Lifecycle(pp_size, prompt=32)
    run.step()
    assert not any(saver.pending for saver in run.savers)
    run.complete_send()
    run.assert_freed()
