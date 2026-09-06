def normalize_scores(scores):
    total = sum(scores)
    return [score / total for score in scores] if total else scores


def rank(scores, limit=3):
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
