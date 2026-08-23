from src.benchmarks.cresci import isolation_forest_evaluation as evaluation


def test_resolve_contamination_caps_at_sklearn_limit():
    assert evaluation.resolve_contamination(0.6846064064323694) == 0.5
    assert evaluation.resolve_contamination(0.3) == 0.3
    assert evaluation.resolve_contamination(0.01) == 0.01
