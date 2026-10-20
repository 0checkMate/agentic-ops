from pydantic import BaseModel, ConfigDict


class ProviderEndpoint(BaseModel):
    """Where a provider's OpenAI-compatible API lives and which setting holds its key."""

    model_config = ConfigDict(frozen=True)

    base_url: str
    key_field: str


PROVIDERS: dict[str, ProviderEndpoint] = {
    'groq': ProviderEndpoint(
        base_url='https://api.groq.com/openai/v1',
        key_field='groq_api_key',
    ),
    'cerebras': ProviderEndpoint(
        base_url='https://api.cerebras.ai/v1',
        key_field='cerebras_api_key',
    ),
    'openrouter': ProviderEndpoint(
        base_url='https://openrouter.ai/api/v1',
        key_field='openrouter_api_key',
    ),
    'gemini': ProviderEndpoint(
        base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
        key_field='gemini_api_key',
    ),
}
