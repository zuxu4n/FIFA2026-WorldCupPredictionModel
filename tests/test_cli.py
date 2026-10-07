"""Run the real CLI against a temporary project containing the synthetic dataset."""

import shutil

import pandas as pd
import pytest
from conftest import COLUMNS

from wcpred import config as C
from wcpred.cli import main


@pytest.fixture
def project(tmp_path, monkeypatch, results, wc_format):
    shutil.copytree(C.PATHS.reference_dir, tmp_path / "data" / "reference")
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    out = results.loc[:, COLUMNS].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(raw / "results.csv", index=False)
    monkeypatch.setattr(C, "PATHS", C.Paths(tmp_path))
    return tmp_path


def test_predict_before_training_explains_what_to_do(project, capsys):
    assert main(["predict", "--home", "Spain", "--away", "Japan"]) == 1
    assert "wcpred train" in capsys.readouterr().err


def test_train_predict_simulate(project, wc_format, capsys):
    home, away = wc_format.teams[0], wc_format.teams[-1]
    assert main(["train"]) == 0
    assert (project / "artifacts" / "model" / "booster.json").exists()

    assert main(["predict", "--home", home, "--away", away, "--knockout", "--explain"]) == 0
    out = capsys.readouterr().out
    assert f"{home} win" in out and "To advance" in out and "SHAP" in out

    assert main(["predict", "--home", "Atlantis", "--away", away]) == 1
    assert "unknown team" in capsys.readouterr().err

    assert main(["simulate", "--runs", "200", "--top", "5"]) == 0
    table = pd.read_csv(project / "outputs" / "simulation.csv")
    assert len(table) == 48
    assert table["P_champion"].sum() == pytest.approx(1.0)


def test_asof_models_are_kept_separate(project, capsys):
    assert main(["train", "--asof", "2025-06-01"]) == 0
    assert (project / "artifacts" / "model-asof-2025-06-01" / "meta.json").exists()
    assert main(["simulate", "--asof", "2025-06-01", "--runs", "100"]) == 0
    assert (project / "outputs" / "simulation_asof_2025-06-01.csv").exists()
    # a different as-of date needs its own model
    assert main(["simulate", "--asof", "2025-01-01", "--runs", "100"]) == 1
    assert "wcpred train --asof 2025-01-01" in capsys.readouterr().err


def test_bad_arguments_exit_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["train", "--exclude", "not-a-group"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        main(["predict", "--home", "A", "--away", "B", "--date", "yesterday-ish"])
