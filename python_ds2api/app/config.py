"""
Configuration management for DS2API Python version.
"""
import os
import json
import base64
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AccountConfig(BaseModel):
    """DeepSeek account configuration."""
    email: Optional[str] = None
    mobile: Optional[str] = None
    password: str
    token: Optional[str] = None


class CompatConfig(BaseModel):
    """Compatibility settings."""
    wide_input_strict_output: bool = True
    strip_reference_markers: bool = True


class ResponsesConfig(BaseModel):
    """Responses API settings."""
    store_ttl_seconds: int = 900


class EmbeddingsConfig(BaseModel):
    """Embeddings settings."""
    provider: str = "deterministic"


class ClaudeMappingConfig(BaseModel):
    """Claude model mapping."""
    fast: str = "deepseek-chat"
    slow: str = "deepseek-reasoner"


class AdminConfig(BaseModel):
    """Admin settings."""
    jwt_expire_hours: int = 24


class RuntimeConfig(BaseModel):
    """Runtime settings."""
    account_max_inflight: int = 2
    account_max_queue: int = 0
    global_max_inflight: int = 0
    token_refresh_interval_hours: int = 6


class AutoDeleteConfig(BaseModel):
    """Auto delete settings."""
    mode: str = "none"  # none, single, all


class Settings(BaseSettings):
    """Application settings from environment variables."""
    port: int = Field(default=5001, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    admin_key: str = Field(default="admin", alias="DS2API_ADMIN_KEY")
    jwt_secret: str = Field(default="", alias="DS2API_JWT_SECRET")
    jwt_expire_hours: int = Field(default=24, alias="DS2API_JWT_EXPIRE_HOURS")
    config_path: str = Field(default="config.json", alias="DS2API_CONFIG_PATH")
    config_json: Optional[str] = Field(default=None, alias="DS2API_CONFIG_JSON")

    model_config = {"env_file": ".env", "extra": "ignore"}


class AppConfig(BaseModel):
    """Full application configuration."""
    keys: List[str] = Field(default_factory=list)
    accounts: List[AccountConfig] = Field(default_factory=list)
    model_aliases: Dict[str, str] = Field(default_factory=dict)
    compat: CompatConfig = Field(default_factory=CompatConfig)
    responses: ResponsesConfig = Field(default_factory=ResponsesConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    claude_mapping: Dict[str, str] = Field(default_factory=lambda: {
        "fast": "deepseek-chat",
        "slow": "deepseek-reasoner"
    })
    admin: AdminConfig = Field(default_factory=AdminConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    auto_delete: AutoDeleteConfig = Field(default_factory=AutoDeleteConfig)


class ConfigStore:
    """Configuration store with runtime state management."""

    def __init__(self):
        self.settings = Settings()
        self.config = AppConfig()
        self._account_tokens: Dict[str, str] = {}  # account_id -> token
        self._account_inflight: Dict[str, int] = {}  # account_id -> count

    def load_config(self) -> None:
        """Load configuration from file or environment."""
        config_data = {}

        # Try to load from environment variable first
        if self.settings.config_json:
            try:
                # Try base64 decode first
                try:
                    decoded = base64.b64decode(self.settings.config_json)
                    config_data = json.loads(decoded)
                except Exception:
                    # Try plain JSON
                    config_data = json.loads(self.settings.config_json)
            except Exception as e:
                print(f"Failed to parse DS2API_CONFIG_JSON: {e}")

        # Fall back to file
        if not config_data and os.path.exists(self.settings.config_path):
            try:
                with open(self.settings.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            except Exception as e:
                print(f"Failed to load config file: {e}")

        # Parse configuration
        if config_data:
            # Handle accounts
            accounts = []
            for acc in config_data.get("accounts", []):
                # Skip comment-only entries
                if "_comment" in acc and not acc.get("email") and not acc.get("mobile"):
                    continue
                accounts.append(AccountConfig(
                    email=acc.get("email"),
                    mobile=acc.get("mobile"),
                    password=acc.get("password", ""),
                    token=acc.get("token")
                ))

            self.config = AppConfig(
                keys=config_data.get("keys", []),
                accounts=accounts,
                model_aliases=config_data.get("model_aliases", {}),
                compat=CompatConfig(**config_data.get("compat", {})),
                responses=ResponsesConfig(**config_data.get("responses", {})),
                embeddings=EmbeddingsConfig(**config_data.get("embeddings", {})),
                claude_mapping=config_data.get("claude_mapping", config_data.get("claude_model_mapping", {})),
                admin=AdminConfig(**config_data.get("admin", {})),
                runtime=RuntimeConfig(**config_data.get("runtime", {})),
                auto_delete=AutoDeleteConfig(**config_data.get("auto_delete", {}))
            )

    def get_jwt_secret(self) -> str:
        """Get JWT secret, falling back to admin key."""
        return self.settings.jwt_secret or self.settings.admin_key

    def is_valid_key(self, key: str) -> bool:
        """Check if the API key is valid."""
        return key in self.config.keys

    def get_account_token(self, account_id: str) -> Optional[str]:
        """Get stored token for an account."""
        return self._account_tokens.get(account_id)

    def set_account_token(self, account_id: str, token: str) -> None:
        """Store token for an account."""
        self._account_tokens[account_id] = token

    def get_account_inflight(self, account_id: str) -> int:
        """Get current in-flight request count for an account."""
        return self._account_inflight.get(account_id, 0)

    def acquire_account_slot(self, account_id: str) -> bool:
        """Try to acquire a request slot for an account."""
        current = self._account_inflight.get(account_id, 0)
        if current >= self.config.runtime.account_max_inflight:
            return False
        self._account_inflight[account_id] = current + 1
        return True

    def release_account_slot(self, account_id: str) -> None:
        """Release a request slot for an account."""
        current = self._account_inflight.get(account_id, 0)
        if current > 0:
            self._account_inflight[account_id] = current - 1

    def resolve_model(self, model: str) -> str:
        """Resolve model alias to actual model name."""
        return self.config.model_aliases.get(model, model)


# Global config store instance
config_store = ConfigStore()
