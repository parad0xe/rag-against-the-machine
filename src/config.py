from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    data_dir: Path = Path("data")

    repo_path: Path = data_dir / "raw/vllm-0.10.1"

    # PROCESSED
    processed_dir: Path = data_dir / "processed"
    bm25_dir: Path = processed_dir / "bm25_index"
    chroma_dir: Path = processed_dir / "chroma_index"
    manifest_path: Path = processed_dir / "manifest.json"
    chunks_path: Path = processed_dir / "chunks.json"

    # DATASETS
    dataset_dir: Path = data_dir / "datasets"
    unanswered_path: Path = (
        dataset_dir / "UnansweredQuestions/dataset_code_public.json"
    )
    answered_path: Path = (
        dataset_dir / "AnsweredQuestions/dataset_code_public.json"
    )

    # OUTPUT
    search_output_dir_path: Path = data_dir / "output/search_results"
    answer_output_dir_path: Path = (
        data_dir / "output/search_results_and_answer"
    )

    # MODELS
    llm_model: str = "Qwen/Qwen3-0.6B"
    cross_encoder_model: str = "BAAI/bge-reranker-base"
    translator_model: str = "Helsinki-NLP/opus-mt-mul-en"
    embedding_model: str = "all-MiniLM-L6-v2"

    # VALUES
    default_k: int = 5
    max_chunk_size: int = 2000
    overlap_threshold: float = 0.05
    index_batch_size: int = 32


settings = Settings()
