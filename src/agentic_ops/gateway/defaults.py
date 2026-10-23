from agentic_ops.gateway.models import Alias, ModelSpec
from agentic_ops.gateway.ratelimit import RateLimit
from agentic_ops.gateway.registry import ModelRegistry
from agentic_ops.gateway.router import limiter_key

GROQ_20B = ModelSpec(provider='groq', model_id='openai/gpt-oss-20b', supports_tools=True)
GROQ_120B = ModelSpec(provider='groq', model_id='openai/gpt-oss-120b', supports_tools=True)
GEMINI_LITE = ModelSpec(provider='gemini', model_id='gemini-3.5-flash-lite', supports_tools=True)
OPENROUTER_120B = ModelSpec(
    provider='openrouter', model_id='openai/gpt-oss-120b:free', supports_tools=True
)

_GROQ_FREE = RateLimit(
    requests_per_minute=30,
    tokens_per_minute=8000,
    requests_per_day=1000,
    tokens_per_day=200000,
)


def default_registry() -> ModelRegistry:
    """Alias routes ordered by the procurement_v1 tool-calling benchmark (2026-10-22).

    Groq 20b and 120b tied on accuracy, so 20b leads FAST for its cheaper token budget.
    OpenRouter serves the same 120b weights, so it is the like-for-like fallback.
    Gemini is last: it invented write calls on underspecified requests.
    """
    return ModelRegistry(
        {
            Alias.FAST: [GROQ_20B, OPENROUTER_120B, GEMINI_LITE],
            Alias.DEFAULT: [GROQ_120B, OPENROUTER_120B, GEMINI_LITE],
            Alias.REASONING: [GROQ_120B, OPENROUTER_120B],
        }
    )


def default_limits() -> dict[str, RateLimit]:
    """Free-tier budgets per route. Verify against each provider console before relying on them.

    Groq and OpenRouter figures were verified against published pages on 2026-10-21.
    Gemini is PROVISIONAL: Google does not publish per-model free limits, so replace these
    values with the figures shown in AI Studio.
    """
    return {
        limiter_key(GROQ_20B): _GROQ_FREE,
        limiter_key(GROQ_120B): _GROQ_FREE,
        limiter_key(GEMINI_LITE): RateLimit(requests_per_minute=15, requests_per_day=1500),
        limiter_key(OPENROUTER_120B): RateLimit(requests_per_minute=20, requests_per_day=50),
    }
