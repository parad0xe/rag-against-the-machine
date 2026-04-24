from src.domain.models.search import MinimalSearchResults


class MinimalAnswer(MinimalSearchResults):
    answer: str
