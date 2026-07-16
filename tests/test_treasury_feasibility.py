from urllib.parse import urlparse

import pandas as pd

from tlm.treasury_feasibility import (
    audit_treasury_probe,
    build_treasury_year_url,
    parse_treasury_csv,
    run_treasury_feasibility,
)


def _config(tmp_path) -> dict:
    return {
        "treasury_feasibility": {
            "csv_url_template": "https://example.test/{year}.csv",
            "start": "2022-01-03", "end": "2022-01-14",
            "methodology_start": "2021-12-06", "timeout_seconds": 1,
            "required_columns": ["Date", "2 Yr", "10 Yr"],
            "minimum_weekday_coverage": 1.0,
            "execution_lag_calendar_days": 3, "maximum_carry_days": 7,
            "evidence_urls": [
                "https://example.test/1", "https://example.test/2",
                "https://example.test/3",
            ],
        },
        "output_dir": str(tmp_path / "audit"),
    }


def _csv() -> bytes:
    lines = ["Date,2 Yr,10 Yr"]
    for index, date in enumerate(pd.bdate_range("2022-01-03", "2022-01-14")):
        lines.append(f"{date.strftime('%m/%d/%Y')},{1.0 + index / 100:.2f},{2.0 + index / 100:.2f}")
    return ("\n".join(lines) + "\n").encode()


def test_year_url_and_parser_preserve_registered_tenors(tmp_path):
    config = _config(tmp_path)
    assert build_treasury_year_url(config, 2022) == "https://example.test/2022.csv"
    frame = parse_treasury_csv(_csv(), 2022, ["Date", "2 Yr", "10 Yr"])
    assert list(frame.columns) == ["2 Yr", "10 Yr"]
    assert len(frame) == 10


def test_probe_audit_enforces_strict_t_plus_three_contract(tmp_path):
    frame = parse_treasury_csv(_csv(), 2022, ["Date", "2 Yr", "10 Yr"])
    probe = audit_treasury_probe(frame, _config(tmp_path))
    assert probe["passed"]
    assert probe["weekday_coverage"] == 1.0
    assert probe["checks"]["source_final_strictly_precedes_execution"]


def test_feasibility_selects_source_without_building_features(tmp_path):
    config = _config(tmp_path)

    def fetcher(url: str, timeout: float) -> bytes:
        assert urlparse(url).path.endswith("2022.csv")
        assert timeout == 1
        return _csv()

    result = run_treasury_feasibility(config, fetch_bytes=fetcher)
    output = tmp_path / "audit"
    assert result["selected"]
    assert result["audit"]["passed"]
    assert result["decision"] == "authorize_v18_treasury_curve_data_layer_only"
    assert (output / "annual_probes" / "2022.csv").is_file()
    assert (output / "probe.csv").is_file()
    assert (output / "report.md").is_file()
