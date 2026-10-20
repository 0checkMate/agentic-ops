import pytest
from pydantic import SecretStr

from agentic_ops.config import Settings
from agentic_ops.gateway.errors import GatewayConfigError
from agentic_ops.gateway.factory import build_chat_model
from agentic_ops.gateway.models import ModelSpec
from agentic_ops.gateway.providers import PROVIDERS


def _settings(*, groq: str | None = None, openrouter: str | None = None) -> Settings:
    return Settings(
        groq_api_key=SecretStr(groq) if groq else None,
        openrouter_api_key=SecretStr(openrouter) if openrouter else None,
    )


def test_client_targets_provider_endpoint_with_its_key() -> None:
    spec = ModelSpec(provider='groq', model_id='model-a', supports_tools=True)
    model = build_chat_model(spec, _settings(groq='test-key'))
    assert model.model_name == 'model-a'
    assert model.openai_api_base == PROVIDERS['groq'].base_url
    api_key = model.openai_api_key
    assert isinstance(api_key, SecretStr)
    assert api_key.get_secret_value() == 'test-key'


def test_sdk_retries_are_disabled() -> None:
    spec = ModelSpec(provider='openrouter', model_id='model-c', supports_tools=True)
    model = build_chat_model(spec, _settings(openrouter='test-key'))
    assert model.max_retries == 0


def test_temperature_defaults_to_zero_and_can_be_overridden() -> None:
    spec = ModelSpec(provider='groq', model_id='model-a', supports_tools=True)
    settings = _settings(groq='test-key')
    assert build_chat_model(spec, settings).temperature == 0.0
    assert build_chat_model(spec, settings, temperature=0.7).temperature == 0.7


def test_missing_api_key_is_a_configuration_error() -> None:
    spec = ModelSpec(provider='groq', model_id='model-a', supports_tools=True)
    with pytest.raises(GatewayConfigError):
        build_chat_model(spec, _settings())


def test_unknown_provider_is_a_configuration_error() -> None:
    spec = ModelSpec(provider='nowhere', model_id='model-x', supports_tools=True)
    with pytest.raises(GatewayConfigError):
        build_chat_model(spec, _settings(groq='test-key'))


@pytest.mark.parametrize('name', sorted(PROVIDERS))
def test_provider_table_is_consistent_with_settings(name: str) -> None:
    endpoint = PROVIDERS[name]
    assert endpoint.base_url.startswith('https://')
    assert endpoint.key_field in Settings.model_fields
