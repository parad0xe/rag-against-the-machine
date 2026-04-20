def pluralize(word: str, count: int) -> str:
    if count <= 1:
        return word
    return word + "s"
