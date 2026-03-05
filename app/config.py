from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash-latest"

    whisper_model: str = "small"
    whisper_language: str = "auto"
    whisper_beam_size: int = 5
    whisper_best_of: int = 5
    whisper_temperature: float = 0.0

    min_corner_sec: float = 20.0
    merge_gap_sec: float = 2.0

    database_url: str = "sqlite:///./app/data/video_analyzer.db"

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
    )


settings = Settings()
