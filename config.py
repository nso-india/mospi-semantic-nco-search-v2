import os
from typing import List


class Config:
    """Base configuration"""
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(32).hex()
    DEBUG = False
    TESTING = False
    
    # Security
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB max request size (increased for metadata updates)
    JSON_SORT_KEYS = False
    
    # CORS - Whitelist your domains
    CORS_ORIGINS: List[str] = os.environ.get('CORS_ORIGINS', '').split(',') or [
        "http://localhost:3000",
        "http://localhost:5000",
    ]
    
    # API Security
    API_KEY_REQUIRED = os.environ.get('API_KEY_REQUIRED', 'true').lower() == 'true'
    API_KEYS: List[str] = os.environ.get('API_KEYS', '').split(',') or []
    
    # Rate Limiting
    RATELIMIT_ENABLED = True
    RATELIMIT_DEFAULT = "100 per hour"
    RATELIMIT_STORAGE_URL = "memory://"
    RATELIMIT_STRATEGY = "fixed-window"
    
    # LLM Configuration
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'http://localhost:8001/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'openai/gpt-oss-20b')
    LLM_TIMEOUT = 30  # seconds
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.path.join('logs', 'app.log')
    QUERY_LOG_FILE = os.path.join('logs', 'queries.jsonl')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = False  # Never enable debug in production!
    API_KEY_REQUIRED = False  # Can disable for local testing
    RATELIMIT_ENABLED = False  # Can disable for testing


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    API_KEY_REQUIRED = True
    RATELIMIT_ENABLED = True
    
    # Strict CORS in production
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '').split(',')
    
    # Require API keys in production
    if not Config.API_KEYS:
        raise ValueError("API_KEYS environment variable must be set in production!")


# Configuration selector
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': ProductionConfig
}


def get_config(env: str = None) -> Config:
    """Get configuration based on environment"""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'production')
    return config.get(env, config['default'])
