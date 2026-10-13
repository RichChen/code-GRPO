import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "closeout" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_select_checkpoint_on_the_phase3_dev_counts():
    vt = load("verify_training")
    solved = dict(zip(vt.DEV_STEPS, vt.DEV_SOLVED))
    choice = vt.select_checkpoint(solved, 800)
    assert choice["best_step"] == 175 and choice["within_1_point"] == [150, 175, 200] and choice["chosen"] == 150


def test_select_checkpoint_exactly_one_point_is_not_within_and_not_chained():
    vt = load("verify_training")
    # 125 is exactly 1 point (8 of 800) below 150 but 14 below the best: not chosen.
    assert vt.select_checkpoint({125: 543, 150: 551, 175: 557}, 800)["chosen"] == 150
    assert vt.select_checkpoint({100: 549, 175: 557}, 800)["chosen"] == 175
    assert vt.select_checkpoint({100: 550, 175: 557}, 800)["chosen"] == 100
    # Tied best: the earlier step.
    assert vt.select_checkpoint({50: 10, 75: 10}, 100)["best_step"] == 50


def test_rounding_match_and_half_unit_tie():
    vt = load("verify_training")
    assert vt.matches(2.0492, 2.0, 1) and not vt.matches(2.0492, 2.1, 1)
    assert vt.matches(0.6675, 0.668, 3) and vt.on_tie(0.6675, 0.668, 3)
    assert not vt.on_tie(0.6674, 0.667, 3)
    assert vt.step_value({"reward_mean": 0.4922, "episodes": 128}, "reward_mean") == 63 / 128


def test_round_half_up_rounds_once_from_the_exact_value():
    from fractions import Fraction

    vt = load("verify_training")
    assert vt.round_half_up(Fraction(2136, 3200), 3) == "0.668"
    assert vt.round_half_up(Fraction("2.0492"), 1) == "2.0"
    assert vt.round_half_up(Fraction("0.845"), 2) == "0.85"
    assert vt.round_half_up(Fraction("0.0007190"), 4) == "0.0007"
    assert vt.round_half_up(Fraction("398.412"), 0) == "398"
    vh = load("verify_heldout")
    assert vh.round_half_up(Fraction(213 * 100, 6000), 1) == "3.6"
    assert vh.round_half_up(Fraction(-1, 8), 2) == "-0.13"
