"""Configuration for the Academic Explorer MVP."""



from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
    """Small runtime configuration loaded from environment variables."""

    openalex_mailto: str | None
    semantic_scholar_api_key: str | None
    provider_timeout_seconds: int
    agent_model_id: str
    agent_max_new_tokens: int
    agent_max_length: int
    agent_temperature: float
    agent_torch_dtype: str
    agent_device_map: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""

        return cls(
            openalex_mailto=os.getenv("OPENALEX_MAILTO") or None,
            semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None,
            provider_timeout_seconds=_int_env("ACADEMIC_EXPLORER_PROVIDER_TIMEOUT_SECONDS", 20),
            agent_model_id=os.getenv(
                "ACADEMIC_EXPLORER_AGENT_MODEL_ID",
                "Qwen/Qwen2.5-1.5B-Instruct",
            ),
            agent_max_new_tokens=_int_env("ACADEMIC_EXPLORER_AGENT_MAX_NEW_TOKENS", 512),
            agent_max_length = _int_env("ACADEMIC_EXPLORER_AGENT_MAX_LENGTH", 20),
            agent_temperature=_float_env("ACADEMIC_EXPLORER_AGENT_TEMPERATURE", 0.0),
            agent_torch_dtype=os.getenv("ACADEMIC_EXPLORER_AGENT_TORCH_DTYPE", "auto"),
            agent_device_map=os.getenv("ACADEMIC_EXPLORER_AGENT_DEVICE_MAP", "auto"),
        )


def load_config() -> AppConfig:
    """Load application configuration."""

    return AppConfig.from_env()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default
