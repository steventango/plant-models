import inspect

import metrax
import metrax.classification_metrics


def merge(self, other):
    if other.beta == self.beta:
        return type(self)(
            true_positives=self.true_positives + other.true_positives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
            beta=self.beta,
        )
    else:
        raise ValueError('The "Beta" values between the two are not equal.')


def _make_robust_wrapper(original_method):
    original_func = original_method.__func__
    sig = inspect.signature(original_func)

    @classmethod
    def wrapper(cls, **kwargs):
        valid_params = sig.parameters
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in valid_params.values()
        )

        if has_var_keyword:
            filtered_kwargs = kwargs
        else:
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        return original_func(cls, **filtered_kwargs)

    return wrapper


def patch_metrax():
    # NotImplementedError: Must override merge()
    # https://github.com/google/metrax/issues/131
    metrax.classification_metrics.FBetaScore.merge = merge

    # Patch all metrics to accept arbitrary kwargs in from_model_output
    # This fixes TypeError when using MultiMetric with different args
    # https://github.com/google/metrax/issues/132
    for name in metrax.__all__:
        metric_cls = getattr(metrax, name)
        if isinstance(metric_cls, type) and hasattr(metric_cls, "from_model_output"):
            metric_cls.from_model_output = _make_robust_wrapper(
                metric_cls.from_model_output
            )
