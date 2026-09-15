from tests.conftest import FakeProvider

LINES = ["2 large ripe tomatoes, diced", "1/2 cup olive oil", "salt"]

PARSED = {
    "items": [
        {
            "raw": "2 large ripe tomatoes, diced",
            "name": "tomato",
            "quantity": 2,
            "unit": None,
            "preparation": "diced",
        },
        {
            "raw": "1/2 cup olive oil",
            "name": "olive oil",
            "quantity": 0.5,
            "unit": "cup",
            "preparation": None,
        },
        {"raw": "salt", "name": "salt", "quantity": None, "unit": None, "preparation": None},
    ]
}


def test_parse_ingredients_returns_one_item_per_line(make_client):
    provider = FakeProvider([PARSED])
    response = make_client(provider).post("/v1/parse-ingredients", json={"lines": LINES})

    assert response.status_code == 200
    body = response.json()
    assert [item["name"] for item in body["items"]] == ["tomato", "olive oil", "salt"]
    assert body["items"][1]["quantity"] == 0.5
    assert body["meta"]["retries"] == 0
    assert provider.calls[0]["schema"]["name"] == "parsed_ingredients"


def test_dropped_line_is_repaired(make_client):
    short = {"items": PARSED["items"][:2]}
    provider = FakeProvider([short, PARSED])
    response = make_client(provider).post("/v1/parse-ingredients", json={"lines": LINES})

    assert response.status_code == 200
    assert response.json()["meta"]["retries"] == 1
    assert "expected 3 items, got 2" in provider.calls[1]["user"]


def test_rewritten_raw_line_is_repaired(make_client):
    tampered = {"items": [{**PARSED["items"][0], "raw": "tomatoes"}, *PARSED["items"][1:]]}
    provider = FakeProvider([tampered, PARSED])
    response = make_client(provider).post("/v1/parse-ingredients", json={"lines": LINES})

    assert response.status_code == 200
    assert "item 0 `raw` must repeat the input line verbatim" in provider.calls[1]["user"]


def test_batch_is_capped(make_client):
    response = make_client(FakeProvider([PARSED])).post(
        "/v1/parse-ingredients", json={"lines": ["salt"] * 51}
    )

    assert response.status_code == 400
