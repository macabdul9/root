import math

DEFAULT_TEMPERATURE = 0.7
MAX_CANDIDATES = 500


def softmax(scores):
    largest = max(scores)
    exponentials = [math.exp(score - largest) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def temperature_scale(scores, temperature=DEFAULT_TEMPERATURE):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return [score / temperature for score in scores]
