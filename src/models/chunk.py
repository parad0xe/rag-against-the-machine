from pydantic import BaseModel, NonNegativeInt


class ChunkMetadata(BaseModel):
    text: str
    file_path: str
    first_character_index: NonNegativeInt
    last_character_index: NonNegativeInt
