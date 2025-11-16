#!/usr/bin/env python3
"""
Client management for AI model connections.

This module provides factory functions for building AI model clients
for different providers (OpenRouter, LMStudio) with standardized configuration.
"""

from __future__ import annotations

import os
from typing import Any

import instructor
from dotenv import load_dotenv
from instructor import Instructor, Mode
from openai import OpenAI


def merge_configs(user_config: dict[str, Any], default_config: dict[str, Any]) -> dict[str, Any]:
    """
    Merge user configuration with default configuration.

    User config values override defaults, but missing keys are filled with defaults.

    Args:
        user_config: User-provided configuration dictionary.
        default_config: Default configuration dictionary.

    Returns:
        Merged configuration dictionary.
    """
    merged = default_config.copy()
    merged.update(user_config)
    return merged


def get_openrouter_default_config() -> dict[str, Any]:
    """
    Get default configuration for OpenRouter client.

    Returns:
        Configuration dictionary for OpenRouter provider.
    """
    load_dotenv()
    return {
        "base_url": os.getenv("OPENROUTER_BASE_URL"),
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "model": os.getenv("OPENROUTER_MODEL_ID"),
        "mode": Mode.JSON,
    }


def get_lmstudio_default_config() -> dict[str, Any]:
    """
    Get default configuration for LMStudio client.

    Returns:
        Configuration dictionary for LMStudio provider.
    """
    load_dotenv()
    return {
        "base_url": os.getenv("LMSTUDIO_BASE_URL"),
        "api_key": os.getenv("LMSTUDIO_API_KEY", "123"),
        "model": os.getenv("LMSTUDIO_MODEL_ID"),
        "mode": Mode.JSON_SCHEMA,
    }


def build_openrouter_client(config: dict[str, Any] | None = None) -> Instructor:
    """
    Build OpenRouter client with configuration.

    Args:
        config: Optional user configuration to override defaults.

    Returns:
        Configured OpenRouter client instance.
    """
    if config is None:
        config = {}

    default_config = get_openrouter_default_config()
    final_config = merge_configs(config, default_config)

    provider = f'openrouter/{final_config["model"]}'
    return instructor.from_provider(
        provider,
        api_key=final_config["api_key"],
        mode=final_config["mode"],
        base_url=final_config["base_url"],
    )


def build_lmstudio_client(config: dict[str, Any] | None = None) -> Instructor:
    """
    Build LMStudio client with configuration.

    Args:
        config: Optional user configuration to override defaults.

    Returns:
        Configured LMStudio client instance.
    """
    if config is None:
        config = {}

    default_config = get_lmstudio_default_config()
    final_config = merge_configs(config, default_config)

    openai_client = OpenAI(
        api_key=final_config["api_key"], base_url=final_config["base_url"]
    )
    return instructor.from_openai(
        openai_client, mode=final_config["mode"], model=final_config["model"]
    )


def build_client(use_lmstudio: bool | None = None, config: dict[str, Any] | None = None) -> Instructor:
    """
    Build AI client based on configuration or environment variables.

    Args:
        use_lmstudio: Force use of LMStudio client. If None, uses USE_LMSTUDIO env var.
        config: Optional configuration dictionary.

    Returns:
        Configured AI client instance.
    """
    if config is None:
        config = {}

    load_dotenv()
    use_lmstudio_final = (
        use_lmstudio or os.getenv("USE_LMSTUDIO", "false").lower() == "true"
    )

    if use_lmstudio_final:
        return build_lmstudio_client(config)
    return build_openrouter_client(config)


# Define a global LLM_CLIENT variable for easy access
LLM_CLIENT = build_client()


# Define the public API for this module
__all__ = [
    "build_client",
    "build_lmstudio_client",
    "build_openrouter_client",
    "get_lmstudio_default_config",
    "get_openrouter_default_config",
    "merge_configs",
]