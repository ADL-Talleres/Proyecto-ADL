from pydantic import BaseSettings
import os

class Settings(BaseSettings):
    PROJECT_NAME = "Kidney Tumor Segmentation"
    DB_NAME = "kidney_tumor_segmentation"
    DB_USER = "postgres"
    DB_PASSWORD = '123456'
    DB_HOST = "localhost"
    DB_PORT = '5432'

class AuthSettings(BaseSettings):
    authjwt_secret_key: str = "secret"

    class Config:
        case_sensitive = True

settings = Settings()