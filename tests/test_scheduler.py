"""Scheduler logic tests with a fake executor (no torch needed).

The driver mimics engine.step(): schedule -> "execute" (advance
num_computed, sample fake tokens) -> update_after_exec.
"""

from dataclasses import dataclass, field

from servelab.config import CacheConfig, SchedulerConfig
from servelab.engine.sampling_params import SamplingParams
from servelab.engine.scheduler import Scheduler
from servelab.engine.sequence import Sequence
from servelab.kv_cache.manager import BlockManager


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@dataclass
class FakeSpec:
    output_len: int = 8


class Driver:
    """Executes whatever the scheduler decides, with fake model work."""

    def __init__(self, num_blocks=16, block_size=4, max_batched=32,
                 max_seqs=8, policy="fcfs", chunked=True, enable_prefix=True):
        self.clock = FakeClock()
        self.bm = BlockManager(num_blocks=num_blocks, block_size=block_size,
                               enable_prefix_caching=enable_prefix)
        self.scheduler = Scheduler(
            SchedulerConfig(max_num_seqs=max_seqs, max_num_batched_tokens=max_batched,
                            chunked_prefill=chunked, policy=policy),
            CacheConfig(block_size=block_size), self.bm, self.clock, eos_token_id=-1)
        self.specs: dict = {}
        self.finished = []

    def submit(self, req_id, prompt_len, output_len, prefix_ids=None, priority=0):
        seq = Sequence(req_id, list(range(1000, 1000 + prompt_len)),
                       SamplingParams(max_tokens=output_len, ignore_eos=True),
                       priority=priority, arrival_time=self.clock.t)
        self.specs[req_id] = output_len
        self.scheduler.add_request(seq)
        return seq

    def run(self, max_steps=1000):
        steps = 0
        while self.scheduler.has_unfinished() and steps < max_steps:
            steps += 1
            out = self.scheduler.schedule()
            progressed = False
            sampled = {}
            for ps in out.prefills:
                ps.seq.num_computed_tokens += ps.num_new_tokens
                progressed = True
                if ps.seq.is_prefill_done():
                    sampled[ps.seq.request_id] = 7
            for seq in out.decodes:
                sampled[seq.request_id] = 7
                progressed = True
            self.clock.t += 0.01
            self.finished += self.scheduler.update_after_exec(sampled)
            if not progressed and not out.is_empty():
                raise RuntimeError("scheduled but nothing executed")
            if out.is_empty() and self.scheduler.has_unfinished():
                # could be a transient LATER; force one more round by failing fast
                stuck = steps
                if stuck > max_steps - 2:
                    raise RuntimeError("scheduler deadlock")
        assert all(s.is_finished() for s in self.finished)


def test_all_finish_basic():
    d = Driver()
    for i in range(5):
        d.submit(f"r{i}", prompt_len=10, output_len=5)
    d.run()
    assert len(d.finished) == 5
    outs = {s.request_id: s.num_output_tokens for s in d.finished}
    assert outs == {f"r{i}": 5 for i in range(5)}


def test_chunked_prefill_respects_token_budget():
    d = Driver(max_batched=8)
    d.submit("big", prompt_len=40, output_len=1)
    out = d.scheduler.schedule()
    assert len(out.prefills) == 1
    assert out.prefills[0].num_new_tokens <= 8
    assert out.prefills[0].num_computed == 0


def test_fcfs_admission_order():
    d = Driver(max_batched=12)   # small budget: one seq per step roughly
    for i in range(4):
        d.submit(f"r{i}", prompt_len=20, output_len=1)
    first = d.scheduler.schedule()
    assert first.prefills[0].seq.request_id == "r0"


def test_priority_order():
    d = Driver(max_batched=12)
    d.submit("low", prompt_len=20, output_len=1, priority=5)
    d.submit("high", prompt_len=20, output_len=1, priority=0)
    first = d.scheduler.schedule()
    assert first.prefills[0].seq.request_id == "high"


def test_prefix_cache_hit_two_requests():
    d = Driver(enable_prefix=True)
    prompt = list(range(500, 540))       # 10 blocks (bs=4)
    s1 = d.submit("a", prompt_len=40, output_len=2)
    s1.prompt_token_ids = prompt
    s1.token_ids = prompt
    s1.num_prompt_tokens = 40
    d.run()
    s2 = d.submit("b", prompt_len=40, output_len=2)
    s2.prompt_token_ids = prompt
    s2.token_ids = prompt
    s2.num_prompt_tokens = 40
    d.run()
    assert d.bm.stat_prefix_hit_tokens > 0


def test_preemption_under_memory_pressure():
    # 8 blocks * 4 tokens = 32 tokens of KV; 4 seqs need 12 tokens each
    d = Driver(num_blocks=8, block_size=4, max_batched=64, max_seqs=4)
    for i in range(4):
        d.submit(f"r{i}", prompt_len=8, output_len=10)
    d.run(max_steps=500)
    assert len(d.finished) == 4
    total_preempts = sum(s.metrics.num_preemptions for s in d.finished)
    assert total_preempts > 0


def test_max_num_seqs_cap():
    d = Driver(max_seqs=2, max_batched=64)
    for i in range(5):
        d.submit(f"r{i}", prompt_len=4, output_len=3)
    out = d.scheduler.schedule()
    assert len(out.prefills) <= 2
    d.run()


def test_abort():
    d = Driver()
    d.submit("x", 10, 10)
    assert d.scheduler.abort_request("x")
    assert not d.scheduler.has_unfinished()
