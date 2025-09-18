from pydantic_settings import BaseSettings


PLACES = "Savojbolagh County, Alborz Province, Iran"#"Tehran Province, Iran"
PLACE = "Savojbolagh County, Alborz Province, Iran"

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./rideshare.db"
    SECRET_KEY: str = "your-secret-key-here"  # Change in production
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 300
    DEBUG: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
