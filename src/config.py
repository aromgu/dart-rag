from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    dart_api_key: str = ""

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "dart_rag"
    postgres_user: str = "dart_rag"
    postgres_password: str = "dart_rag_dev_password"

    embedding_model: str = "BAAI/bge-m3"
    generation_model: str = "Qwen/Qwen2.5-3B-Instruct"
    hf_home: str = "./model_cache"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
