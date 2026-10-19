from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='AGENTIC_', env_file='.env', extra='ignore')

    environment: str = 'local'
    database_url: SecretStr = SecretStr('postgresql://postgres:postgres@localhost:5432/agentic')

    @property
    def sqlalchemy_url(self) -> str:
        """Database URL with the psycopg v3 driver selected for SQLAlchemy."""
        url = self.database_url.get_secret_value()
        return url.replace('postgresql://', 'postgresql+psycopg://', 1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
