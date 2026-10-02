"""SLO-oriented serving metrics (torch-free).

Goodput = fraction of requests meeting BOTH the TTFT and TPOT SLO
(cf. DistServe's goodput formulation). Jain index measures load balance
across replicas.
"""

from dataclasses import dataclass
from typing import Dict, List

from tabulate import tabulate

from ..utils.common import pct
from .cluster import RequestResult


@dataclass
class ClusterMetrics:
    num_finished: int
    makespan_s: float
    throughput_tok_s: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    ttft_mean_ms: float
    tpot_p50_ms: float
    tpot_p99_ms: float
    goodput: float
    jain_load_index: float
    mean_cache_hit_rate: float
    total_preemptions: int

    def to_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)

    def print_table(self, title: str = "serving metrics") -> None:
        rows = [[k, f"{v:.4g}" if isinstance(v, float) else v]
                for k, v in self.to_dict().items()]
        print(f"\n=== {title} ===")
        print(tabulate(rows, headers=["metric", "value"], tablefmt="github"))


def tpot_ms(r: RequestResult) -> float:
    if r.output_tokens <= 1:
        return 0.0
    return (r.finish_time - r.first_token_time) / (r.output_tokens - 1) * 1000.0


def compute_metrics(results: List[RequestResult],
                    ttft_slo_ms: float = 2000.0,
                    tpot_slo_ms: float = 50.0,
                    num_replicas: int = 1) -> ClusterMetrics:
    if not results:
        raise ValueError("empty results")
    ttfts = [(r.first_token_time - r.arrival_time) * 1000.0 for r in results]
    tpots = [tpot_ms(r) for r in results if r.output_tokens > 1]
    makespan = max(r.finish_time for r in results) - min(r.arrival_time for r in results)
    total_out = sum(r.output_tokens for r in results)
    good = sum(1 for r, t in zip(results, ttfts)
               if t <= ttft_slo_ms and tpot_ms(r) <= tpot_slo_ms) / len(results)
    per_replica = [0.0] * num_replicas
    for r in results:
        per_replica[min(r.replica_id, num_replicas - 1)] += r.output_tokens
    s = sum(per_replica)
    jain = (s ** 2 / (num_replicas * sum(x * x for x in per_replica))
            if s > 0 and any(per_replica) else 1.0)
    return ClusterMetrics(
        num_finished=len(results),
        makespan_s=makespan,
        throughput_tok_s=total_out / makespan if makespan > 0 else 0.0,
        ttft_p50_ms=pct(ttfts, 50),
        ttft_p90_ms=pct(ttfts, 90),
        ttft_p99_ms=pct(ttfts, 99),
        ttft_mean_ms=sum(ttfts) / len(ttfts),
        tpot_p50_ms=pct(tpots, 50),
        tpot_p99_ms=pct(tpots, 99),
        goodput=good,
        jain_load_index=jain,
        mean_cache_hit_rate=0.0,   # filled by the caller from replica stats
        total_preemptions=sum(r.preemptions for r in results),
    )
