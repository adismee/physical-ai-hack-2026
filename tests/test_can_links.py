"""The YAM adapter's CAN link check, on a fake /sys/class/net. No arms needed."""

import pathlib

from dream_hackathon.arms import yam


def _link(sys_net: pathlib.Path, name: str, flags: str, operstate: str) -> None:
    link = sys_net / name
    link.mkdir(parents=True)
    (link / "flags").write_text(flags + "\n")
    (link / "operstate").write_text(operstate + "\n")


def test_a_link_that_is_up_is_ready(tmp_path):
    _link(tmp_path, "can_follower_l", "0x40081", "up")
    assert yam.can_link_problem("can_follower_l", tmp_path) is None


def test_a_link_left_down_after_a_reboot_gets_its_command(tmp_path):
    _link(tmp_path, "can_follower_l", "0x80", "down")
    problem = yam.can_link_problem("can_follower_l", tmp_path)
    assert "sudo ip link set can_follower_l up type can bitrate 1000000" in problem


def test_a_link_up_without_carrier_points_at_the_arm(tmp_path):
    _link(tmp_path, "can_follower_r", "0x40081", "down")
    assert "powered" in yam.can_link_problem("can_follower_r", tmp_path)


def test_unnamed_links_point_at_the_naming_step(tmp_path):
    _link(tmp_path, "can0", "0x80", "down")
    _link(tmp_path, "can1", "0x80", "down")
    problem = yam.can_link_problem("can_follower_l", tmp_path)
    assert "can0, can1" in problem and "CAN names" in problem
