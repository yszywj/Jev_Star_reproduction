"""Offline integration tests: JEV selection, SkillManager lifecycle, CLI and suite wiring
for CoverRetreat. No SC2, no paid calls."""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_micro.client import JevError
from jev_micro.skill_manager import SkillManager
from jev_micro.skills.cover_retreat import allocate_cover_retreat
from jev_micro.skill_policy import (COVER_RETREAT, DIRECT_CONTROL,
                                    decode_skill, skill_request, validate_skill_response)
from test_jev_cover_retreat import build_state, make_ally, make_enemy


def payload(health=10, threat_position=(13, 16), safe=False):
    allies = {**make_ally(0, (10, 16), health=health), **make_ally(1, (9, 16)),
              **make_ally(2, (8, 16))}
    enemies = {**make_enemy(0, threat_position)}
    state = build_state(allies, enemies)
    if safe:  # No visible threat anywhere near.
        state["visible_enemies"] = {}
        state["relations"] = {a: {} for a in allies}
        for actor, moves in state["movement_candidates"].items():
            for entry in moves.values():
                entry["distances_to_visible_enemies"] = {}
    state["objective"] = "Eliminate all opposing units before the episode step limit."
    state["map"], state["episode"], state["game_loop"] = "3m", 1, 80
    state["steps_remaining"] = 50
    return {"model": "jev-1.13.0", "state": state,
            "questions": {"u0": {"criteria": {"6": "Attack e0"}}}}


def choice_answer(choice, keys):
    probabilities = {key: (0.9 if key == choice else 0.1 / max(1, len(keys) - 1)) for key in keys}
    return {"team_skill": {"type": "choice", "choice": choice, "confidence": 0.9,
                           "probabilities": probabilities}}


class JevSelectionTests(unittest.TestCase):
    def test_jev_can_choose_cover_retreat(self):
        request = skill_request(payload(), "jev-1.13.0", 4)
        keys = tuple(request["questions"]["team_skill"]["criteria"])
        self.assertIn("2", keys)
        answer = choice_answer("2", keys)
        validated = validate_skill_response({"model": "jev-1.13.0", "answers": answer,
                                             "usage": {"input_tokens": 5}}, request)
        mode, priority = decode_skill(request, validated["answers"])
        self.assertEqual(mode, COVER_RETREAT)
        self.assertIsNone(priority)

    def test_direct_control_choice_keeps_original_flow(self):
        request = skill_request(payload(), "jev-1.13.0", 4)
        keys = tuple(request["questions"]["team_skill"]["criteria"])
        mode, _ = decode_skill(request, choice_answer("0", keys))
        self.assertEqual(mode, DIRECT_CONTROL)
        manager = SkillManager(4)
        manager.select(mode)
        self.assertFalse(manager.active)  # Per-unit JEV keeps deciding; no local skill actions.

    def test_illegal_choice_rejected_by_protocol_validation(self):
        request = skill_request(payload(), "jev-1.13.0", 4)
        keys = tuple(request["questions"]["team_skill"]["criteria"])
        bad = {"model": "jev-1.13.0",
               "answers": choice_answer("99", keys), "usage": {}}
        with self.assertRaises(JevError):
            validate_skill_response(bad, request)


class ManagerLifecycleTests(unittest.TestCase):
    def test_selection_enters_cover_execution(self):
        manager = SkillManager(4)
        manager.select(COVER_RETREAT)
        self.assertTrue(manager.active)
        actions, telemetry = manager.step(payload(), 3)
        self.assertEqual(len(actions), 3)
        self.assertEqual(telemetry["skill"], "cover_retreat")
        self.assertEqual(telemetry["retreat_units"], ["u0"])
        self.assertIn(actions[0], (2, 3, 4, 5))

    def test_safe_streak_ends_skill(self):
        manager = SkillManager(4, safe_streak_required=2)
        manager.select(COVER_RETREAT)
        reasons = []
        for step in range(3):
            actions, telemetry = manager.step(payload(safe=True), 3)  # Threat disappears.
            reasons.append(telemetry.get("termination_reason"))
            if telemetry.get("termination_reason"):
                break  # run.py stops stepping a terminated skill.
        self.assertEqual(reasons, [None, "safe_streak"])
        self.assertEqual(manager.stats["terminations"], {"safe_streak": 1})
        self.assertFalse(manager.active)

    def test_safe_streak_resets_when_threat_returns(self):
        manager = SkillManager(4, safe_streak_required=2)
        manager.select(COVER_RETREAT)
        _, t1 = manager.step(payload(safe=True), 3)
        self.assertEqual(manager.safe_streak, 1)
        _, t2 = manager.step(payload(), 3)  # Threat back: streak resets, no oscillation exit.
        self.assertEqual(manager.safe_streak, 0)
        self.assertIsNone(t2.get("termination_reason"))

    def test_max_steps_termination(self):
        manager = SkillManager(2)
        manager.select(COVER_RETREAT)
        reasons = []
        for _ in range(3):
            _, telemetry = manager.step(payload(), 3)
            reasons.append(telemetry.get("termination_reason"))
            if telemetry.get("termination_reason"):
                break  # run.py stops stepping a terminated skill.
        self.assertEqual(reasons, [None, "max_steps"])

    def test_role_reassignment_counted(self):
        manager = SkillManager(4)
        manager.select(COVER_RETREAT)
        manager.step(payload(), 3)
        # u0 recovers to full health: still safe-streak window, but roles change.
        _, telemetry = manager.step(payload(health=45), 3)
        self.assertEqual(manager.stats["role_reassignments"], 1)
        self.assertEqual(telemetry["retreat_units"], [])

    def test_episode_reset_clears_cover_state(self):
        manager = SkillManager(4)
        manager.select(COVER_RETREAT)
        manager.step(payload(), 3)
        self.assertTrue(manager.retreat_units or manager.cover_units)
        manager.reset()
        self.assertIsNone(manager.current_skill)
        self.assertEqual(manager.retreat_units, [])
        self.assertEqual(manager.cover_units, [])
        self.assertEqual(manager.safe_streak, 0)
        self.assertFalse(manager.active)

    def test_risk_threshold_is_configurable(self):
        manager = SkillManager(4, risk_threshold=0.1)
        manager.select(COVER_RETREAT)
        _, telemetry = manager.step(payload(health=20), 3)  # Ratio 0.44 > 0.1: not endangered.
        self.assertEqual(telemetry["retreat_units"], [])

    def test_cover_execution_never_invokes_the_paid_client(self):
        from jev_micro.client import JevClient
        manager = SkillManager(4)
        manager.select(COVER_RETREAT)
        with patch.object(JevClient, "choose", side_effect=AssertionError("paid call")):
            for _ in range(4):
                manager.step(payload(), 3)
        self.assertEqual(manager.stats["steps"], 4)


class CliWiringTests(unittest.TestCase):
    def test_defaults_keep_original_behavior_and_skills_disabled(self):
        from jev_micro.run import build_parser
        args = build_parser().parse_args([])
        self.assertEqual(args.skill_mode, "off")
        self.assertEqual(args.skill_name, "focus_fire_budget")
        self.assertEqual((args.cover_risk_threshold, args.cover_safe_streak), (0.35, 2))

    def test_cover_arguments_parse(self):
        from jev_micro.run import build_parser
        args = build_parser().parse_args(["--skill-mode", "always", "--skill-name",
                                          "cover_retreat", "--cover-risk-threshold", "0.2",
                                          "--cover-safe-streak", "3"])
        self.assertEqual((args.skill_name, args.cover_risk_threshold, args.cover_safe_streak),
                         ("cover_retreat", 0.2, 3))

    def test_invalid_cover_arguments_rejected(self):
        from jev_micro.run import build_parser
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--cover-risk-threshold", "1.5"])
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--cover-safe-streak", "0"])

    def test_skill_mode_jev_requires_jev_policy(self):
        import jev_micro.run as run_module
        with patch.object(run_module, "load_api_key", return_value=None), \
             patch.object(sys, "argv", ["run", "--policy", "nearest", "--skill-mode", "jev"]):
            with self.assertRaises(SystemExit) as raised:
                run_module.main()
        self.assertEqual(raised.exception.code, 2)


class SuiteForwardingTests(unittest.TestCase):
    def run_suite(self, tmp, argv, existing_suite=None):
        from jev_micro import suite
        output = Path(tmp) / "suite-output"
        if existing_suite:
            output.mkdir(parents=True, exist_ok=True)
            (output / "suite.json").write_text(json.dumps(existing_suite), encoding="utf-8")
        commands = []

        class FakeProcess:
            stdout = iter(["Output: fake\n"])
            def wait(self):
                return 0
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        def fake_popen(command, **kwargs):
            commands.append(command)
            case_output = Path(command[command.index("--output-dir") + 1])
            case_output.mkdir(parents=True, exist_ok=True)
            (case_output / "summary.json").write_text(json.dumps({
                "episodes": [{"status": "win", "steps": 1, "initial": {"enemies_alive": 3},
                              "final": {"enemies_alive": 0, "allies_alive": 3,
                                        "enemy_health": 0, "enemy_shield": 0,
                                        "allied_health": 40, "allied_shield": 0}}],
                "skill_mode": "always",
                "skill": {"activations": 1, "steps": 2, "terminations": {}}}),
                encoding="utf-8")
            return FakeProcess()

        with patch.object(subprocess, "Popen", fake_popen), \
             patch.object(sys, "argv", ["suite", *argv, "--output-dir", str(output)]):
            try:
                suite.main()
            except SystemExit:
                pass
        return output, commands

    def test_suite_forwards_cover_arguments(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            output, commands = self.run_suite(
                tmp, ["--maps", "3m", "--policies", "nearest", "--episodes", "1",
                      "--skill-mode", "always", "--skill-name", "cover_retreat",
                      "--cover-risk-threshold", "0.25", "--cover-safe-streak", "3"])
            forwarded = commands[0]
            self.assertEqual(forwarded[forwarded.index("--skill-name") + 1], "cover_retreat")
            self.assertEqual(forwarded[forwarded.index("--cover-risk-threshold") + 1], "0.25")
            self.assertEqual(forwarded[forwarded.index("--cover-safe-streak") + 1], "3")
            settings = json.loads((output / "suite.json").read_text(encoding="utf-8"))
            self.assertEqual(settings["skill_name"], "cover_retreat")

    def test_resume_rejects_mismatched_skill_name(self):
        import tempfile
        from jev_micro import suite
        with tempfile.TemporaryDirectory() as tmp:
            output, _ = self.run_suite(
                tmp, ["--maps", "3m", "--policies", "nearest", "--episodes", "1",
                      "--skill-mode", "always", "--skill-name", "cover_retreat"])
            with patch.object(sys, "argv", ["suite", "--maps", "3m", "--policies", "nearest",
                                            "--episodes", "1", "--skill-mode", "always",
                                            "--skill-name", "focus_fire_budget",
                                            "--output-dir", str(output), "--resume"]):
                with self.assertRaises(SystemExit) as raised:
                    suite.main()
                self.assertEqual(raised.exception.code, 2)


class FirstStepThresholdTests(unittest.TestCase):
    def test_manager_first_step_uses_entry_threshold(self):
        # P0 regression: first execution step must not use the tightened
        # continue threshold. u0 ratio 0.30 with a nearby threat retreats.
        allies = {**make_ally(0, (10, 16), health=13.5), **make_ally(1, (9, 16))}
        enemies = {**make_enemy(0, (13, 16))}
        state = build_state(allies, enemies)
        state["objective"] = "win"
        state["map"], state["episode"], state["game_loop"] = "3m", 1, 80
        state["steps_remaining"] = 50
        manager = SkillManager(4)
        manager.select(COVER_RETREAT)
        actions, telemetry = manager.step({"model": "m", "state": state}, 2)
        self.assertEqual(telemetry["retreat_units"], ["u0"])
        self.assertIn(actions[0], (2, 3, 4, 5))

    def test_units_keep_individual_retreat_history(self):
        # u0 (0.31) retreating, u1 (0.18) newly endangered: step two judges
        # each unit by its own history — u0 may drop out, u1 stays.
        allies = {**make_ally(0, (10, 16), health=14), **make_ally(1, (11, 16), health=8)}
        enemies = {**make_enemy(0, (13, 16))}
        state = build_state(allies, enemies)
        state["objective"] = "win"
        state["map"], state["episode"], state["game_loop"] = "3m", 1, 80
        state["steps_remaining"] = 50
        manager = SkillManager(4)
        manager.select(COVER_RETREAT)
        _, first = manager.step({"model": "m", "state": state}, 2)
        self.assertEqual(first["retreat_units"], ["u0", "u1"])
        _, second = manager.step({"model": "m", "state": state}, 2)
        self.assertEqual(second["retreat_units"], ["u1"])  # u0 below continue bar.

    def test_alive_unit_without_any_candidate_gets_stop_not_zero(self):
        allies = {**make_ally(0, (10, 16), health=13.5), **make_ally(1, (9, 16))}
        state = build_state(allies, {**make_enemy(0, (13, 16))})
        state["relations"]["u0"]["e0"]["smac_attack_available"] = False
        state["movement_candidates"]["u0"] = {}  # No move, no attack, no heal.
        result = allocate_cover_retreat(state, 2)
        self.assertEqual(result["actions"][0], 1)  # Explicit Stop, never 0.
        self.assertEqual(result["stop_fallbacks"], ["u0"])


if __name__ == "__main__":
    unittest.main()
