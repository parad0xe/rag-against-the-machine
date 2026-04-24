from typing import TypedDict


class ChunkMetadata(TypedDict):
    text: str
    hash: str
    file_path: str
    first_character_index: int
    last_character_index: int
