"""Misc AIX functions."""


def get_llm_leaderboards():
    """Get the LLM leaderboards.

    Requires the optional ``tabled`` dependency (``pip install aix[tables]``).
    """
    from tabled import get_tables_from_url

    return get_tables_from_url("https://www.vellum.ai/llm-leaderboard")
