from __future__ import annotations

from instructor import Instructor, Mode
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
import instructor
import os

def merge_configs(user_config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user configuration with default configuration.
    User config values override defaults, but missing keys are filled with defaults.
    """
    merged = default_config.copy()
    merged.update(user_config)
    return merged

def get_openrouter_default_config() -> Dict[str, Any]:
    """Get default configuration for OpenRouter client."""
    load_dotenv()
    return {
        "base_url": os.getenv("OPENROUTER_BASE_URL"),
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "model": os.getenv("OPENROUTER_MODEL_ID"),
        "mode": Mode.TOOLS,
    }

def get_lmstudio_default_config() -> Dict[str, Any]:
    """Get default configuration for LMStudio client."""
    load_dotenv()
    return {
        "base_url": os.getenv("LMSTUDIO_BASE_URL"),
        "api_key": os.getenv("LMSTUDIO_API_KEY", '123'),
        "model": os.getenv("LMSTUDIO_MODEL_ID"),
        "mode": Mode.JSON_SCHEMA,
    }

def build_openrouter_client(config: Optional[Dict[str, Any]] = {}) -> Instructor:
    """Build OpenRouter client with configuration."""
    default_config = get_openrouter_default_config()
    final_config = merge_configs(config, default_config)
    
    provider = f'openrouter/{final_config["model"]}'
    return instructor.from_provider(provider, api_key=final_config["api_key"], mode=final_config["mode"], base_url=final_config["base_url"])

def build_lmstudio_client(config: Optional[Dict[str, Any]] = {}) -> Instructor:
    """Build LMStudio client with configuration."""
    default_config = get_lmstudio_default_config()
    final_config = merge_configs(config, default_config)

    openai_client = OpenAI(api_key=final_config["api_key"], base_url=final_config["base_url"])
    return instructor.from_openai(openai_client, mode=final_config["mode"], model=final_config["model"])

def build_client(use_lmstudio: Optional[bool] = None, config: Optional[Dict[str, Any]] = {}) -> Instructor:
    load_dotenv()
    use_lmstudio_final = use_lmstudio or os.getenv("USE_LMSTUDIO", "false").lower() == "true"
    if use_lmstudio_final:
        return build_lmstudio_client(config)
    return build_openrouter_client(config)

__all__ = ["build_openrouter_client", "build_lmstudio_client", "build_client"]