"""
DS2API Python Version - DeepSeek Web to OpenAI API Converter
"""
from .config import config_store
from .deepseek_client import deepseek_client
from .account_pool import account_pool

__version__ = "1.0.0"
__all__ = ["config_store", "deepseek_client", "account_pool"]
