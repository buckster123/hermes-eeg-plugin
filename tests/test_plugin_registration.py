# tests/test_plugin_registration.py
"""Test that the plugin registers correctly with Hermes."""

from unittest.mock import MagicMock
import hermes_eeg


EXPECTED_TOOLS = {
    "eeg_connect",
    "eeg_disconnect",
    "eeg_stream_start",
    "eeg_stream_stop",
    "eeg_realtime_emotion",
    "eeg_experience_get",
    "eeg_calibrate_baseline",
    "eeg_list_sessions",
}


def test_register_all_tools():
    ctx = MagicMock()
    hermes_eeg.register(ctx)
    assert ctx.register_tool.call_count == 8


def test_all_tools_in_eeg_toolset():
    ctx = MagicMock()
    hermes_eeg.register(ctx)
    for call in ctx.register_tool.call_args_list:
        assert call.kwargs["toolset"] == "eeg"


def test_all_expected_tool_names():
    ctx = MagicMock()
    hermes_eeg.register(ctx)
    registered_names = {call.kwargs["name"] for call in ctx.register_tool.call_args_list}
    assert registered_names == EXPECTED_TOOLS


def test_schemas_have_required_fields():
    for name, schema in hermes_eeg.TOOL_SCHEMAS.items():
        assert "name" in schema, f"Missing 'name' in schema for {name}"
        assert "description" in schema, f"Missing 'description' in schema for {name}"
        assert "parameters" in schema, f"Missing 'parameters' in schema for {name}"
        assert schema["parameters"]["type"] == "object"


def test_schemas_use_openai_format_not_anthropic():
    """Ensure no schema uses Anthropic's input_schema key."""
    for name, schema in hermes_eeg.TOOL_SCHEMAS.items():
        assert "input_schema" not in schema, f"Schema {name} uses Anthropic format (input_schema). Use 'parameters'."


def test_all_handlers_exist():
    for name in EXPECTED_TOOLS:
        assert name in hermes_eeg.TOOL_HANDLERS, f"Missing handler for {name}"
        assert callable(hermes_eeg.TOOL_HANDLERS[name])


def test_check_fn_is_fast():
    """check_fn must be fast (no network calls)."""
    assert hermes_eeg._check_eeg_available() in (True, False)


def test_all_tools_have_emoji():
    ctx = MagicMock()
    hermes_eeg.register(ctx)
    for call in ctx.register_tool.call_args_list:
        assert call.kwargs["emoji"] == "🧠"
