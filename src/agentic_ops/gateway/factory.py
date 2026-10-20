from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agentic_ops.config import Settings
from agentic_ops.gateway.errors import GatewayConfigError
from agentic_ops.gateway.models import ModelSpec
from agentic_ops.gateway.providers import PROVIDERS


def build_chat_model(
    spec: ModelSpec, settings: Settings, *, temperature: float = 0.0
) -> ChatOpenAI:
    """Build a chat client for one route.

    SDK-level retries are disabled (max_retries=0): the gateway owns retry and fallback
    policy, so hidden retries cannot multiply requests or burn free-tier daily quotas.
    """
    endpoint = PROVIDERS.get(spec.provider)
    if endpoint is None:
        raise GatewayConfigError(f'Unknown provider {spec.provider!r}')
    api_key: SecretStr | None = getattr(settings, endpoint.key_field)
    if api_key is None:
        raise GatewayConfigError(f'No API key configured for provider {spec.provider!r}')
    return ChatOpenAI(
        model=spec.model_id,
        base_url=endpoint.base_url,
        api_key=api_key,
        temperature=temperature,
        max_retries=0,
    )
