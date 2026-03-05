from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str = ""
    whisper_model: str = "base"
    database_url: str = "sqlite:///./app/data/video_analyzer.db"

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
    )


settings = Settings()
