from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "AI Perception Engine"
    app_version: str = "0.1.0"
    environment: str = "development"

    # Database
    # DATABASE_URL  – Transaction-mode pooler (port 6543).  Used by the app.
    # DIRECT_URL    – Session connection (port 5432). Used by Alembic only;
    #                 the transaction pooler does not support DDL migrations.
    database_url: str = (
        "postgresql+psycopg://user:password@localhost:5432/perception?sslmode=disable"
    )
    direct_url: str = (
        "postgresql+psycopg://user:password@localhost:5432/perception?sslmode=disable"
    )


settings = Settings()
