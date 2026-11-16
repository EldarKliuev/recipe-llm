import json
import logging
import sys

from app.logging import JsonFormatter, request_id_var


def make_record(
    message: str, *args: object, extra: dict[str, object] | None = None
) -> logging.LogRecord:
    return logging.getLogger("app.test").makeRecord(
        "app.test", logging.INFO, __file__, 1, message, args, None, extra=extra
    )


def test_log_line_is_json_with_request_id_and_extra_fields():
    token = request_id_var.set("core-api-42")
    try:
        line = JsonFormatter().format(
            make_record("recipe %s", "adapted", extra={"retries": 1, "latency_ms": 14})
        )
    finally:
        request_id_var.reset(token)

    payload = json.loads(line)
    assert payload["message"] == "recipe adapted"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert payload["request_id"] == "core-api-42"
    assert payload["retries"] == 1
    assert payload["latency_ms"] == 14
    assert payload["ts"]
    # Internal LogRecord attributes stay out of the line.
    assert not {"args", "msg", "pathname", "lineno", "created"} & payload.keys()


def test_log_line_outside_a_request_has_no_request_id():
    payload = json.loads(JsonFormatter().format(make_record("llm-service started")))

    assert "request_id" not in payload


def test_exception_and_unserialisable_values_still_give_one_json_line():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = make_record("failed", extra={"path": object()})
        record.exc_info = sys.exc_info()

    line = JsonFormatter().format(record)

    assert "\n" not in line
    payload = json.loads(line)
    assert "RuntimeError: boom" in payload["exc"]
    assert payload["path"].startswith("<object object")
