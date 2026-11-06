"""The shipped station configs load, and mistakes name the key to fix."""

import pathlib

import pytest
import yaml
from conftest import filled_template

from dream_hackathon import config

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "configs"


def test_templates_refuse_to_load_until_filled_in(tmp_path: pathlib.Path) -> None:
    with pytest.raises(config.ConfigError, match=r"'cameras\.overhead\.device' still says REPLACE"):
        config.load(CONFIGS / "yam.yaml")
    with pytest.raises(config.ConfigError, match=r"'arms\.left\.port' still says REPLACE"):
        config.load(CONFIGS / "maker.yaml")


def test_filled_yam_and_maker_configs_load(tmp_path: pathlib.Path) -> None:
    yam = config.load(filled_template("yam", tmp_path))
    assert (yam.family, yam.control_hz // yam.fps) == ("yam", 2)
    maker = config.load(filled_template("maker", tmp_path))
    assert maker.family == "makermods"
    assert maker.arms["left"].tcp_xyz_m is None  # shipped unmeasured on purpose


def test_metal_config_needs_its_measured_gripper_angles(tmp_path: pathlib.Path) -> None:
    with pytest.raises(config.ConfigError, match="gripper_open_deg"):
        config.load(filled_template("metal", tmp_path))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw.update(fps=25), "fps"),
        (lambda raw: raw.update(record_width=640, record_height=480), "16:9"),
        (lambda raw: raw["cameras"].pop("wrist_l"), "cameras"),
        (lambda raw: raw["arms"]["left"].update(quest_to_base=[[1, 0, 0], [0, 1, 0], [0, 0, -1]]), "rotation"),
        (lambda raw: raw["arms"]["right"].update(workspace_min_m=[1, 1, 1]), "workspace_min_m"),
    ],
)
def test_mistakes_name_the_key(tmp_path: pathlib.Path, change, message: str) -> None:
    raw = yaml.safe_load(filled_template("yam", tmp_path).read_text())
    change(raw)
    path = tmp_path / "station.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(config.ConfigError, match=message):
        config.load(path)


def test_a_missing_file_says_how_to_make_it(tmp_path):
    with pytest.raises(config.ConfigError, match="does not exist: copy your arm's template"):
        config.load(tmp_path / "station.yaml")
