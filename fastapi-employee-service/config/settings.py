import os

APP_NAME = os.getenv("APP_NAME", "Employee Service")
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
