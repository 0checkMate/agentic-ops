"""The production chat model is built from the default routes and limits."""

from agentic_ops.api.main import gateway_chat_model
from agentic_ops.config import Settings
from agentic_ops.gateway.models import Alias


def test_default_routes_and_limits_are_consistent() -> None:
    # Gateway.__init__ raises if any default route lacks a limiter key, so building
    # the model proves registry and limits agree before any request is served.
    model = gateway_chat_model(Settings())
    assert model.route_alias is Alias.DEFAULT
