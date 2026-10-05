"""Output-length predictors.

Predictors feed SJF scheduling and predictive routing. Their accuracy/
latency trade-off is a first-class research axis (roadmap #4): the oracle is
an upper bound; anything smarter than the online baselines below is
publishable material (cf. S3: NeurIPS'23, Andes, response-length perception).
"""


from ..engine.sequence import Sequence


class OutputLengthPredictor:
    name = "base"

    def predict_remaining(self, seq: Sequence) -> int:
        raise NotImplementedError

    def observe(self, seq: Sequence) -> None:
        """Called by the simulator/engine when a request finishes."""


class OraclePredictor(OutputLengthPredictor):
    """Uses the true output length (trace-driven simulation only)."""
    name = "oracle"

    def __init__(self, truth=None):
        self.truth = truth or {}          # request_id -> total output tokens

    def register(self, req_id: str, output_tokens: int) -> None:
        self.truth[req_id] = output_tokens

    def predict_remaining(self, seq: Sequence) -> int:
        total = self.truth.get(seq.request_id, seq.num_output_tokens + 8)
        return max(1, total - seq.num_output_tokens)


class ConstantPredictor(OutputLengthPredictor):
    name = "constant"

    def __init__(self, value: int = 128):
        self.value = value

    def predict_remaining(self, seq: Sequence) -> int:
        return max(1, self.value - seq.num_output_tokens)


class OnlineMeanPredictor(OutputLengthPredictor):
    """Running mean of finished output lengths (EMA for adaptivity)."""
    name = "online-mean"

    def __init__(self, warm_start: float = 128.0, momentum: float = 0.1):
        self.mean = warm_start
        self.momentum = momentum

    def predict_remaining(self, seq: Sequence) -> int:
        return max(1, int(round(self.mean)) - seq.num_output_tokens)

    def observe(self, seq: Sequence) -> None:
        n = seq.num_output_tokens
        self.mean = (1 - self.momentum) * self.mean + self.momentum * n


class PromptRegressionPredictor(OutputLengthPredictor):
    """Online linear regression output_len ~ a * prompt_len + b, fit on
    finished requests (S3-style 'stateful' baseline, kept tiny)."""
    name = "prompt-reg"

    def __init__(self, warm_start: float = 128.0):
        self.mean = warm_start
        self.slope = 0.0
        self.n = 0
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_xy = 0.0
        self._sum_xx = 0.0

    def predict_remaining(self, seq: Sequence) -> int:
        est = self.slope * seq.num_prompt_tokens + self.mean
        return max(1, int(round(est)) - seq.num_output_tokens)

    def observe(self, seq: Sequence) -> None:
        x, y = float(seq.num_prompt_tokens), float(seq.num_output_tokens)
        self.n += 1
        self._sum_x += x; self._sum_y += y
        self._sum_xy += x * y; self._sum_xx += x * x
        denom = self.n * self._sum_xx - self._sum_x ** 2
        if abs(denom) > 1e-6:
            self.slope = (self.n * self._sum_xy - self._sum_x * self._sum_y) / denom
            self.mean = (self._sum_y - self.slope * self._sum_x) / self.n
        else:
            self.mean = self._sum_y / self.n


def build_predictor(name: str, **kwargs) -> OutputLengthPredictor:
    registry = {
        "oracle": OraclePredictor,
        "constant": ConstantPredictor,
        "online-mean": OnlineMeanPredictor,
        "prompt-reg": PromptRegressionPredictor,
    }
    if name not in registry:
        raise ValueError(f"unknown predictor '{name}', choose from {list(registry)}")
    return registry[name](**kwargs)
