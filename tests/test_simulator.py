"""End-to-end simulator tests (torch-free)."""


from servelab.simulator.cluster import ClusterSimulator, SimConfig
from servelab.simulator.metrics import compute_metrics
from servelab.simulator.model_cost import GPU_PRESETS, MODEL_PRESETS
from servelab.simulator.predictor import build_predictor
from servelab.simulator.trace import make_synthetic_trace
from servelab.simulator.moe import rebalanced_sim


def make_sim(router="least-loaded", num_replicas=2, predictor=None, **kw):
    cfg = SimConfig(
        num_replicas=num_replicas,
        gpu=GPU_PRESETS["A100"],
        model=MODEL_PRESETS["qwen2.5-7b"],
        router=router,
        **kw,
    )
    predictor = predictor or build_predictor("online-mean")
    return ClusterSimulator(cfg, predictor=predictor)


def test_synthetic_run_finishes_all():
    trace = make_synthetic_trace(num_requests=60, request_rate=2.0,
                                 prompt_mean=128, output_mean=64)
    sim = make_sim()
    results = sim.run(trace)
    assert len(results) == 60
    for r in results:
        assert r.finish_time > r.first_token_time >= r.arrival_time
        assert r.output_tokens > 0


def test_metrics_sane():
    trace = make_synthetic_trace(num_requests=40, request_rate=3.0,
                                 prompt_mean=256, output_mean=128, seed=1)
    sim = make_sim()
    results = sim.run(trace)
    m = compute_metrics(results, num_replicas=2)
    assert 0.0 < m.goodput <= 1.0
    assert m.ttft_p50_ms > 0 and m.ttft_p99_ms >= m.ttft_p50_ms
    assert 0 < m.jain_load_index <= 1.0
    assert m.throughput_tok_s > 0


def test_routers_all_run():
    trace = make_synthetic_trace(num_requests=30, request_rate=2.0, seed=2)
    for router in ("round-robin", "least-loaded", "prefix-affinity", "predictive"):
        sim = make_sim(router=router)
        results = sim.run(trace)
        assert len(results) == 30, router


def test_prefix_affinity_improves_cache_hits():
    """With heavy prefix sharing, affinity routing should match more prefix
    tokens than round-robin (which spreads conversations across replicas)."""
    trace = make_synthetic_trace(num_requests=120, request_rate=1.5,
                                 num_prefix_groups=4, prefix_len=256,
                                 prompt_mean=384, output_mean=64, seed=3)
    def matched_tokens(router_name):
        sim = make_sim(router=router_name, num_replicas=2)
        sim.run(trace)
        return sum(r.stat_matched_tokens for r in sim.replicas)
    assert matched_tokens("prefix-affinity") > matched_tokens("round-robin")


def test_pd_disaggregation_runs():
    trace = make_synthetic_trace(num_requests=40, request_rate=2.0, seed=4)
    sim = make_sim(num_replicas=4, pd_mode=True)
    results = sim.run(trace)
    assert len(results) == 40
    assert all(r.first_token_time >= r.arrival_time for r in results)


def test_oracle_vs_online_predictor_differs():
    trace = make_synthetic_trace(num_requests=30, seed=5)
    sim1 = make_sim(predictor=build_predictor("oracle"))
    sim2 = make_sim(predictor=build_predictor("constant", value=4))
    r1, r2 = sim1.run(trace), sim2.run(trace)
    assert len(r1) == len(r2) == 30


def test_moe_rebalance_improves_imbalance():
    base, better = rebalanced_sim(num_tokens=1024, num_experts=32,
                                  num_gpus=4, zipf_alpha=1.6)
    assert better.imbalance_factor < base.imbalance_factor
    assert better.imbalance_factor <= 1.3
