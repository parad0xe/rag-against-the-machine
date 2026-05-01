from src.domain.models.base import Chunk


def parse_llm_thought(full_text: str) -> tuple[str, str]:
    if "<think>" not in full_text:
        return "", full_text.strip("\n")

    parts = full_text.split("<think>", 1)
    after_think = parts[1]

    if "</think>" in after_think:
        think_parts = after_think.split("</think>", 1)
        return think_parts[0].strip("\n"), think_parts[1].strip("\n")

    return after_think.strip("\n"), ""


def build_context_from_chunks(chunks: list[Chunk]) -> str:
    context = []
    for i, chunk in enumerate(chunks):
        context.append(
            f"---  SOURCE #{i + 1} ---\n"
            f"File: {chunk.get('file_path')} "
            f"(Chars: {chunk.get('first_character_index')}-"
            f"{chunk.get('last_character_index')})\n"
            f"Content: {chunk.get('text')}"
        )
    return "\n".join(context)
