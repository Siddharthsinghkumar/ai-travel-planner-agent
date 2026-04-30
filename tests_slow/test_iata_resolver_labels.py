from core.iata_resolver import city_for_iata, label_for_iata, resolve_location


def test_city_for_iata_returns_city_when_known():
    city = city_for_iata("DEL")
    assert isinstance(city, str)
    assert city.strip() != ""


def test_label_for_iata_formats_city_and_code():
    label = label_for_iata("BOM")
    assert isinstance(label, str)
    assert label.endswith("(BOM)")


def test_iata_label_helpers_return_none_for_unknown():
    assert city_for_iata("XXX") is None
    assert label_for_iata("XXX") is None


def test_resolve_location_prefers_real_city_name_over_plainword_iata_collision():
    # "new" can collide with a valid 3-letter IATA token; resolver should still
    # map the full location phrase correctly.
    assert resolve_location("new delhi") == "DEL"
