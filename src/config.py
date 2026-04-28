from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    repo_path: Path = Path("data/raw/vllm-0.10.1")

    processed_dir: Path = Path("data/processed")
    bm25_dir: Path = processed_dir / "bm25_index"
    chroma_dir: Path = processed_dir / "chroma_index"
    manifest_path: Path = processed_dir / "manifest.json"
    chunks_path: Path = processed_dir / "chunks.json"

    dataset_dir: Path = Path("data/datasets")
    unanswered_path: Path = (
        dataset_dir / "UnansweredQuestions/dataset_code_public.json"
    )
    answered_path: Path = (
        dataset_dir / "AnsweredQuestions/dataset_code_public.json"
    )

    output_dir: Path = Path("data/output")
    search_output: Path = output_dir / "search_results.json"
    answer_output: Path = output_dir / "answer_results.json"

    translator_model: str = "Helsinki-NLP/opus-mt-mul-en"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "Qwen/Qwen3-0.6B"

    default_k: int = 10
    max_chunk_size: int = 2000
    overlap_threshold: float = 0.05

    index_batch_size: int = 32


settings = Settings()
