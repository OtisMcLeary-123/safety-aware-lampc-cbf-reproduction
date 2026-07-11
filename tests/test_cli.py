import json

import pytest

from lampc_cbf.cli import main


def test_cli_outputs_dry_run_manifest(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--gamma", "0.05", "--steps", "3"]) == 0

    manifest = json.loads(capsys.readouterr().out)
    assert manifest["mode"] == "dry-run"
    assert manifest["external_api_calls"] is False
    assert manifest["steps"] == 3
    assert manifest["config"]["gamma"] == 0.05


def test_cli_rejects_gamma_outside_experimental_range() -> None:
    with pytest.raises(SystemExit):
        main(["--gamma", "0.2"])


def test_cli_can_explicitly_use_theoretical_range(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--gamma", "1", "--allow-theoretical-gamma"]) == 0
    assert json.loads(capsys.readouterr().out)["config"]["gamma"] == 1.0
