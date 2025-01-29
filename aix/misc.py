"""Misc AIX functions."""

from tabled import get_tables_from_url


def get_llm_leaderboards():
    """Get the LLM leaderboards."""
    return get_tables_from_url("https://www.vellum.ai/llm-leaderboard")
