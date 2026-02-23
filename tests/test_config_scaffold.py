from pathlib import Path

from llm_orchestrator.config import load_config, scaffold_default_config


def test_scaffold_default_config_creates_yaml(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    target = scaffold_default_config()

    assert target == Path("orchestrator.yaml")
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert "providers:" in content
    assert "models:" in content


def test_scaffold_default_config_does_not_overwrite_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target = Path("orchestrator.yaml")
    target.write_text("custom: true\n", encoding="utf-8")

    scaffold_default_config(overwrite=False)

    assert target.read_text(encoding="utf-8") == "custom: true\n"


def test_scaffold_default_config_overwrites_when_forced(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target = Path("orchestrator.yaml")
    target.write_text("custom: true\n", encoding="utf-8")

    scaffold_default_config(overwrite=True)

    content = target.read_text(encoding="utf-8")
    assert "providers:" in content
    assert "models:" in content


def test_load_config_autogenerates_default_yaml_when_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert not Path("orchestrator.yaml").exists()

    cfg = load_config()

    assert Path("orchestrator.yaml").exists()
    assert cfg is not None
    assert len(cfg.models) > 0