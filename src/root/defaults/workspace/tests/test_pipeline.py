from pipeline import normalize_scores, rank


def test_normalize_sums_to_one():
    assert sum(normalize_scores([1, 3])) == 1


def test_rank_returns_indices():
    assert rank([5, 1, 9]) == [2, 0, 1]
