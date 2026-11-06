"""The Quest's adb check: what `adb devices` says, and the advice it turns into. No headset needed."""

from dream_hackathon import quest


def test_a_ready_headset_has_no_problem():
    assert quest.adb_problem("List of devices attached\n1WMHH0000000AB\tdevice\n\n") is None


def test_nothing_listed_is_left_to_oculus_reader():
    assert quest.adb_problem("List of devices attached\n\n") is None
    assert quest.adb_problem("") is None


def test_no_permissions_points_at_the_udev_rule():
    output = (
        "List of devices attached\n"
        "1WMHH0000000AB\tno permissions (missing udev rules? user is in the plugdev group); see [http://x]\n"
    )
    assert "udev rule" in quest.adb_problem(output)


def test_unauthorized_asks_for_the_prompt_in_the_headset():
    assert "Allow USB debugging" in quest.adb_problem("List of devices attached\n1WMHH0000000AB\tunauthorized\n")


def test_a_device_on_wifi_is_not_judged():
    assert quest.adb_problem("List of devices attached\n10.0.0.5:5555\tunauthorized\n") is None
