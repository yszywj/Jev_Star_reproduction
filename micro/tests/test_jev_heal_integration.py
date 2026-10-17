"""Offline integration tests: JEV selection, SkillManager lifecycle, CLI/suite wiring
for HealAssign. No SC2, no paid calls."""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_micro.client import JevError
from jev_micro.skill_manager import SkillManager
from jev_micro.skill_policy import (DIRECT_CONTROL, HEAL_ASSIGN,
                                    decode_skill, skill_request, validate_skill_response)
from test_jev_cover_retreat import build_state, make_ally, make_enemy
from test_jev_heal_assign import healer, heal_entry


def payload(with_healer=True, wounded=True):
    allies = {**make_ally(0, (10, 16), health=20 if wounded else 45, loss=10 if wounded else 0),
              **make_ally(1, (12, 16), health=45)}
    if with_healer:
        allies.update(healer(2, (11, 16)))
    enemies = {**make_enemy(0, (18, 18))}
    state = build_state(allies, enemies)
    state["heal_candidates"] = ({"u2": {"6": heal_entry("u0", 2.0, 25)}}
                               if with_healer and wounded else {})
    state["objective"] = "Eliminate all opposing units before the episode step limit."
    state["map"], state["episode"], state["game_loop"] = "MMM", 1, 80
    state["steps_remaining"] = 50
    return {"model": "jev-1.13.0", "state": state,
            "questions": {"u0": {"criteria": {"6": "Attack e0"}}}}


def choice_answer(choice, keys):
    probabilities = {key: (0.9 if key == choice else 0.1 / max(1, len(keys) - 1)) for key in keys}
    return {"team_skill": {"type": "choice", "choice": choice, "confidence": 0.9,
                           "probabilities": probabilities}}


class JevSelectionTests(unittest.TestCase):
    def test_heal_candidate_offered_only_when_healers_exist(self):
        with_healer = skill_request(payload(True), "jev-1.13.0", 4)
        without = skill_request(payload(False), "jev-1.13.0", 4)
        keys_with = with_healer["questions"]["team_skill"]["criteria"]
        keys_without = without["questions"]["team_skill"]["criteria"]
        self.assertIn("heal_assign", " ".join(keys_with.values()))
        self.assertNotIn("heal_assign", " ".join(keys_without.values()))
        self.assertFalse(without["state"]["healers_present"])  # No unexecutable candidate.

    def test_jev_can_choose_heal_assign(self):
        request = skill_request(payload(True), "jev-1.13.0", 4)
        keys = tuple(request["questions"]["team_skill"]["criteria"])
        answer = choice_answer("3", keys)
        validated = validate_skill_response({"model": "jev-1.13.0", "answers": answer,
                                             "usage": {"input_tokens": 5}}, request)
        mode, priority = decode_skill(request, validated["answers"])
        self.assertEqual(mode, HEAL_ASSIGN)
        self.assertIsNone(priority)

    def test_no_candidate_maps_to_heal_without_healers(self):
        request = skill_request(payload(False), "jev-1.13.0", 4)
        keys = tuple(request["questions"]["team_skill"]["criteria"])
        for key in keys:
            mode, _ = decode_skill(request, choice_answer(key, keys))
            self.assertNotEqual(mode, HEAL_ASSIGN)

    def test_direct_control_keeps_original_flow(self):
        request = skill_request(payload(True), "jev-1.13.0", 4)
        keys = tuple(request["questions"]["team_skill"]["criteria"])
        mode, _ = decode_skill(request, choice_answer("0", keys))
        self.assertEqual(mode, DIRECT_CONTROL)
        manager = SkillManager(4)
        manager.select(mode)
        self.assertFalse(manager.active)

    def test_malformed_response_rejected_by_protocol(self):
        request = skill_request(payload(True), "jev-1.13.0", 4)
        keys = tuple(request["questions"]["team_skill"]["criteria"])
        bad = {"model": "jev-1.13.0", "answers": choice_answer("42", keys), "usage": {}}
        with self.assertRaises(JevError):
            validate_skill_response(bad, request)


class ManagerLifecycleTests(unittest.TestCase):
    def test_selection_enters_heal_execution_with_partial_actions(self):
        manager = SkillManager(4)
        manager.select(HEAL_ASSIGN)
        self.assertTrue(manager.active)
        partial, telemetry = manager.step(payload(), 3)
        self.assertEqual(telemetry["skill"], "heal_assign")
        self.assertEqual(telemetry["assignments"], {"u2": "u0"})
        self.assertEqual(partial, {2: 6})  # Healer slot only; u0/u1 untouched.

    def test_no_wounded_targets_terminates(self):
        manager = SkillManager(4)
        manager.select(HEAL_ASSIGN)
        partial, telemetry = manager.step(payload(wounded=False), 3)
        self.assertEqual(telemetry["termination_reason"], "no_heal_demand")
        self.assertFalse(manager.active)
        self.assertNotIn(partial.get(2), (6, 7))  # Never a stale heal command.

    def test_max_steps_termination(self):
        manager = SkillManager(2)
        manager.select(HEAL_ASSIGN)
        reasons = []
        for _ in range(3):
            _, telemetry = manager.step(payload(), 3)
            reasons.append(telemetry.get("termination_reason"))
            if telemetry.get("termination_reason"):
                break  # run.py stops stepping a terminated skill.
        self.assertEqual(reasons, [None, "max_steps"])

    def test_switch_counted_and_stability_kept(self):
        manager = SkillManager(4)
        manager.select(HEAL_ASSIGN)
        manager.step(payload(), 3)
        # Target u0 heals to full: assignment must be released, not clung to.
        partial, telemetry = manager.step(payload(wounded=False), 3)
        self.assertEqual(telemetry["assignments"], {})
        self.assertGreaterEqual(manager.stats["heal_target_switches"], 0)

    def test_healer_death_reassigns(self):
        manager = SkillManager(4)
        manager.select(HEAL_ASSIGN)
        p = payload()
        p["state"]["allies"].update(healer(3, (11.5, 16)))
        p["state"]["heal_candidates"]["u3"] = {"6": heal_entry("u0", 2.5, 25)}
        manager.step(p, 4)
        dead = payload()
        dead["state"]["allies"].update(healer(2, (11, 16), alive=False))
        dead["state"]["heal_candidates"]["u2"] = {}
        dead["state"]["allies"].update(healer(3, (11.5, 16)))
        dead["state"]["heal_candidates"]["u3"] = {"6": heal_entry("u0", 2.5, 25)}
        partial, telemetry = manager.step(dead, 4)
        self.assertEqual(telemetry["assignments"], {"u3": "u0"})

    def test_episode_reset_clears_heal_state(self):
        manager = SkillManager(4)
        manager.select(HEAL_ASSIGN)
        manager.step(payload(), 3)
        manager.reset()
        self.assertIsNone(manager.current_skill)
        self.assertEqual(manager.target_assignments, {})
        self.assertFalse(manager.active)

    def test_switch_margin_configurable(self):
        manager = SkillManager(4, heal_switch_margin=1.5)  # Huge margin: never switch.
        manager.select(HEAL_ASSIGN)
        manager.step(payload(), 3)
        p = payload()
        p["state"]["allies"]["u1"]["health"] = 5  # u1 now critical.
        p["state"]["allies"]["u1"]["health_lost_since_previous_decision"] = 40
        p["state"]["heal_candidates"]["u2"]["7"] = heal_entry("u1", 2.0, 40)
        _, telemetry = manager.step(p, 3)
        self.assertEqual(telemetry["assignments"], {"u2": "u0"})  # Kept by margin.

    def test_heal_execution_never_invokes_the_paid_client(self):
        from jev_micro.client import JevClient
        manager = SkillManager(4)
        manager.select(HEAL_ASSIGN)
        with patch.object(JevClient, "choose", side_effect=AssertionError("paid call")):
            for _ in range(4):
                manager.step(payload(), 3)
        self.assertEqual(manager.stats["steps"], 4)

    def test_merging_partial_actions_keeps_one_action_per_unit(self):
        from jev_micro.policy import nearest_actions
        manager = SkillManager(4)
        manager.select(HEAL_ASSIGN)
        partial, telemetry = manager.step(payload(), 3)
        base = nearest_actions(payload(), 3)
        merged = [partial.get(i, action) for i, action in enumerate(base)]
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[2], 6)   # Healer slot from HealAssign.
        self.assertEqual(merged[:2], base[:2])  # Non-healers keep the base policy.


class CliWiringTests(unittest.TestCase):
    def test_defaults_keep_original_behavior(self):
        from jev_micro.run import build_parser
        args = build_parser().parse_args([])
        self.assertEqual(args.skill_mode, "off")
        self.assertEqual((args.skill_name, args.heal_switch_margin),
                         ("focus_fire_budget", 0.15))

    def test_heal_arguments_parse(self):
        from jev_micro.run import build_parser
        args = build_parser().parse_args(["--skill-mode", "always", "--skill-name",
                                          "heal_assign", "--heal-switch-margin", "0.3"])
        self.assertEqual((args.skill_name, args.heal_switch_margin), ("heal_assign", 0.3))

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

    def test_suite_forwards_heal_arguments(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            output, commands = self.run_suite(
                tmp, ["--maps", "3m", "--policies", "nearest", "--episodes", "1",
                      "--skill-mode", "always", "--skill-name", "heal_assign",
                      "--heal-switch-margin", "0.25"])
            forwarded = commands[0]
            self.assertEqual(forwarded[forwarded.index("--skill-name") + 1], "heal_assign")
            self.assertEqual(forwarded[forwarded.index("--heal-switch-margin") + 1], "0.25")
            settings = json.loads((output / "suite.json").read_text(encoding="utf-8"))
            self.assertEqual(settings["skill_name"], "heal_assign")

    def test_resume_rejects_mismatched_heal_configuration(self):
        import tempfile
        from jev_micro import suite
        with tempfile.TemporaryDirectory() as tmp:
            output, _ = self.run_suite(
                tmp, ["--maps", "3m", "--policies", "nearest", "--episodes", "1",
                      "--skill-mode", "always", "--skill-name", "heal_assign"])
            with patch.object(sys, "argv", ["suite", "--maps", "3m", "--policies", "nearest",
                                            "--episodes", "1", "--skill-mode", "always",
                                            "--skill-name", "heal_assign",
                                            "--heal-switch-margin", "0.4",
                                            "--output-dir", str(output), "--resume"]):
                with self.assertRaises(SystemExit) as raised:
                    suite.main()
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
