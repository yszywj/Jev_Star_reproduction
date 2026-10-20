"""Offline integration tests: manager lifecycle, JEV choice, CLI/suite wiring
for AdaptiveFocusFire. No SC2, no paid calls."""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_micro.client import JevError
from jev_micro.skill_manager import SkillManager
from jev_micro.skill_policy import (ADAPTIVE_FOCUS_FIRE, ADAPTIVE_POSTURES,
                                    COVER_RETREAT, DIRECT_CONTROL,
                                    FOCUS_FIRE_BUDGET, HEAL_ASSIGN,
                                    decode_skill, skill_request, validate_skill_response)
from test_jev_adaptive_focus_fire import scene, wounded_scene


def payload(state=None):
    state = state or scene(n_allies=4, n_enemies=3, enemy_health=30)
    state["objective"] = "Eliminate all opposing units."
    state["map"], state["episode"], state["game_loop"] = "8m", 1, 80
    state["steps_remaining"] = 50
    return {"model": "jev-1.13.0", "state": state,
            "questions": {"u0": {"criteria": {"6": "Attack e0"}}}}


def choice_answer(choice, keys):
    probabilities = {key: (0.9 if key == choice else 0.1 / max(1, len(keys) - 1)) for key in keys}
    return {"team_skill": {"type": "choice", "choice": choice, "confidence": 0.9,
                           "probabilities": probabilities}}


class JevSelectionTests(unittest.TestCase):
    def test_adaptive_posture_candidates_appended_and_backward_compatible(self):
        request = skill_request(payload(), "jev-1.13.0", 4)
        criteria = request["questions"]["team_skill"]["criteria"]
        keys = sorted(criteria, key=int)
        # Existing ids unchanged: 0 direct, 1 focus, 2 cover, 3 heal (healer-less
        # snapshot shifts explicit targets to 3..6), then adaptive postures.
        self.assertEqual(criteria["0"].split(":")[0], "direct_control")
        adaptive_keys = keys[-len(ADAPTIVE_POSTURES):]
        for key, posture in zip(adaptive_keys, ADAPTIVE_POSTURES):
            self.assertIn(f"adaptive_focus_fire ({posture} posture)", criteria[key])

    def test_jev_can_choose_adaptive_with_posture(self):
        request = skill_request(payload(), "jev-1.13.0", 4)
        keys = tuple(request["questions"]["team_skill"]["criteria"])
        adaptive_key = sorted(keys, key=int)[-2]  # balanced posture.
        answer = choice_answer(adaptive_key, keys)
        validated = validate_skill_response({"model": "jev-1.13.0", "answers": answer,
                                             "usage": {"input_tokens": 5}}, request)
        mode, param = decode_skill(request, validated["answers"])
        self.assertEqual(mode, ADAPTIVE_FOCUS_FIRE)
        self.assertEqual(param, "balanced")

    def test_existing_skills_do_not_degrade(self):
        request = skill_request(payload(), "jev-1.13.0", 4)
        keys = tuple(request["questions"]["team_skill"]["criteria"])
        modes = {decode_skill(request, choice_answer(key, keys))[0] for key in keys}
        self.assertEqual(modes, {DIRECT_CONTROL, FOCUS_FIRE_BUDGET, COVER_RETREAT,
                                 ADAPTIVE_FOCUS_FIRE})

    def test_malformed_response_rejected(self):
        request = skill_request(payload(), "jev-1.13.0", 4)
        keys = tuple(request["questions"]["team_skill"]["criteria"])
        bad = {"model": "jev-1.13.0", "answers": choice_answer("99", keys), "usage": {}}
        with self.assertRaises(JevError):
            validate_skill_response(bad, request)


class ManagerLifecycleTests(unittest.TestCase):
    def test_activation_executes_adaptive_with_full_actions(self):
        manager = SkillManager(4)
        manager.select(ADAPTIVE_FOCUS_FIRE, posture="balanced")
        self.assertTrue(manager.active)
        actions, telemetry = manager.step(payload(), 4)
        self.assertEqual(len(actions), 4)
        self.assertEqual(telemetry["skill"], "adaptive_focus_fire")
        self.assertEqual(telemetry["posture"], "balanced")
        self.assertEqual(telemetry["phase"], "engage")
        self.assertEqual(len(telemetry["attack_units"]), 4)

    def test_role_history_survives_skill_window_reactivation(self):
        manager = SkillManager(2)
        manager.select(ADAPTIVE_FOCUS_FIRE)
        manager.step(payload(wounded_scene()), 8)  # u6/u7 retreat.
        manager.step(payload(wounded_scene()), 8)
        self.assertEqual(manager.unit_states["u6"]["retreat_steps"], 2)
        # Window ends; re-activation must NOT reset the retreat caps.
        manager.select(ADAPTIVE_FOCUS_FIRE)
        self.assertEqual(manager.unit_states["u6"]["retreat_steps"], 2)
        self.assertGreaterEqual(manager.unit_states["u6"].get("retreat_cycles", 0), 0)

    def test_stall_and_contact_tracking(self):
        manager = SkillManager(4)
        manager.select(ADAPTIVE_FOCUS_FIRE)
        p = payload()
        manager.step(p, 4)  # Establishes baseline enemy health.
        p2 = payload()
        p2["state"]["game_loop"] = 88
        for enemy in p2["state"]["visible_enemies"].values():
            enemy["health"] -= 10  # Observable progress.
        _, telemetry = manager.step(p2, 4)
        self.assertTrue(manager.contacted)
        self.assertEqual(manager.stall_steps, 0)
        p3 = payload()
        p3["state"]["game_loop"] = 96
        _, telemetry = manager.step(p3, 4)  # No change: stall increments.
        self.assertEqual(manager.stall_steps, 1)

    def test_all_in_sticky_across_steps(self):
        manager = SkillManager(4)
        manager.select(ADAPTIVE_FOCUS_FIRE)
        p = payload()
        manager.step(p, 4)
        # Team collapses + no progress: next step should trigger All-in.
        p2 = payload()
        p2["state"]["game_loop"] = 88
        for unit in p2["state"]["allies"].values():
            unit["health"] = 5
        manager.stall_steps = 6
        _, telemetry = manager.step(p2, 4)
        self.assertTrue(telemetry["all_in"])
        reason = telemetry["all_in_started_now"]
        self.assertIn("no_damage_progress", reason)
        # Later steps stay All-in and never fall back to survival retreats.
        p3 = payload(wounded_scene())
        p3["state"]["game_loop"] = 96
        _, telemetry3 = manager.step(p3, 8)
        self.assertTrue(telemetry3["all_in"])
        self.assertEqual(telemetry3["all_in_reason"], "sticky")
        self.assertEqual(telemetry3["retreat_units"], [])

    def test_terminated_skill_releases_mode_window(self):
        manager = SkillManager(4)
        manager.select(ADAPTIVE_FOCUS_FIRE)
        manager.step(payload(), 4)
        manager._terminate("unsupported_state")
        self.assertEqual(manager.mode_remaining, 0)  # JEV may re-choose next step.

    def test_episode_reset_clears_all_adaptive_state(self):
        manager = SkillManager(4)
        manager.select(ADAPTIVE_FOCUS_FIRE, posture="defensive")
        manager.step(payload(wounded_scene()), 8)
        manager.stall_steps, manager.contacted, manager.all_in = 5, True, True
        manager.reset()
        self.assertEqual(manager.unit_states, {})
        self.assertFalse(manager.all_in)
        self.assertEqual(manager.stall_steps, 0)
        self.assertFalse(manager.contacted)
        self.assertEqual(manager.posture, "balanced")
        self.assertFalse(manager.active)

    def test_adaptive_execution_never_invokes_the_paid_client(self):
        from jev_micro.client import JevClient
        manager = SkillManager(4)
        manager.select(ADAPTIVE_FOCUS_FIRE)
        with patch.object(JevClient, "choose", side_effect=AssertionError("paid call")):
            for _ in range(4):
                manager.step(payload(), 4)
        self.assertEqual(manager.stats["steps"], 4)


class CliWiringTests(unittest.TestCase):
    def test_defaults_and_new_arguments(self):
        from jev_micro.run import build_parser
        args = build_parser().parse_args([])
        self.assertEqual((args.skill_mode, args.skill_name), ("off", "focus_fire_budget"))
        self.assertEqual((args.adaptive_posture, args.adaptive_max_retreat_steps,
                          args.adaptive_stall_steps), ("balanced", 2, 6))
        args = build_parser().parse_args(
            ["--skill-mode", "always", "--skill-name", "adaptive_focus_fire",
             "--adaptive-posture", "aggressive", "--adaptive-max-retreat-steps", "3",
             "--adaptive-stall-steps", "4"])
        self.assertEqual((args.skill_name, args.adaptive_posture,
                          args.adaptive_max_retreat_steps, args.adaptive_stall_steps),
                         ("adaptive_focus_fire", "aggressive", 3, 4))

    def test_invalid_adaptive_arguments_rejected(self):
        from jev_micro.run import build_parser
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--adaptive-posture", "reckless"])
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--adaptive-max-retreat-steps", "0"])

    def test_off_mode_backward_compatible(self):
        from jev_micro.run import build_parser
        self.assertEqual(build_parser().parse_args([]).skill_mode, "off")


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

    def test_suite_forwards_adaptive_arguments(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            output, commands = self.run_suite(
                tmp, ["--maps", "3m", "--policies", "nearest", "--episodes", "1",
                      "--skill-mode", "always", "--skill-name", "adaptive_focus_fire",
                      "--adaptive-posture", "defensive",
                      "--adaptive-max-retreat-steps", "3", "--adaptive-stall-steps", "5"])
            forwarded = commands[0]
            self.assertEqual(forwarded[forwarded.index("--skill-name") + 1],
                             "adaptive_focus_fire")
            self.assertEqual(forwarded[forwarded.index("--adaptive-posture") + 1], "defensive")
            self.assertEqual(forwarded[forwarded.index("--adaptive-max-retreat-steps") + 1], "3")
            self.assertEqual(forwarded[forwarded.index("--adaptive-stall-steps") + 1], "5")
            settings = json.loads((output / "suite.json").read_text(encoding="utf-8"))
            self.assertEqual(settings["adaptive_posture"], "defensive")

    def test_resume_rejects_mismatched_adaptive_configuration(self):
        import tempfile
        from jev_micro import suite
        with tempfile.TemporaryDirectory() as tmp:
            output, _ = self.run_suite(
                tmp, ["--maps", "3m", "--policies", "nearest", "--episodes", "1",
                      "--skill-mode", "always", "--skill-name", "adaptive_focus_fire"])
            with patch.object(sys, "argv", ["suite", "--maps", "3m", "--policies", "nearest",
                                            "--episodes", "1", "--skill-mode", "always",
                                            "--skill-name", "adaptive_focus_fire",
                                            "--adaptive-stall-steps", "9",
                                            "--output-dir", str(output), "--resume"]):
                with self.assertRaises(SystemExit) as raised:
                    suite.main()
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
