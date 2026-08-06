from mechanistic.global_regression import fit_global_phase_model
from mechanistic.proxies import PAPER_PROXY_COLUMNS


def test_global_regression_pools_seeds_and_uses_paper_columns():
    records = []
    for method, offset in (("typiclust", 0.0), ("coreset", 0.1)):
        for seed in (1, 2):
            for budget in range(10, 171, 10):
                risk = 0.8 - budget / 500 + offset
                records.append({
                    "method": method, "seed": seed, "annotation_budget": budget,
                    "test_accuracy": 100 * (1 - risk),
                    **{name: (budget / 100) * (index + 1) + offset for index, name in enumerate(PAPER_PROXY_COLUMNS)},
                })
    analysis, pooled = fit_global_phase_model(records, maximum_segments=2, minimum_segment_points=8)
    assert analysis.features == PAPER_PROXY_COLUMNS
    assert analysis.segment_count in (1, 2)
    assert len(pooled) == 2 * 17
