"""Offline integration tests for JEV skill selection, the SkillManager and CLI wiring."""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jev_micro.client import JevError
from jev_micro.skill_manager import SkillManager
from jev_micro.skill_policy import (COVER_RETREAT, DIRECT_CONTROL, FOCUS_FIRE_BUDGET,
                                    decode_skill, skill_request, validate_skill_response)
from test_jev_focus_fire import make_state


def payload(n_allies=3, **kwargs):
    state = make_state(n_allies, {f"e{j}": None for j in range(2)}, **kwargs)
    state["visible_enemies"] = {}
    for j, health in ((0, 10), (1, 25)):
        from test_jev_focus_fire import enemy
        state["visible_enemies"].update(enemy(j, health))
    state["relations"] = {actor: {e: {"distance": 4.0, "in_weapon_range": True,
                                       "smac_attack_available": True}
                                  for e in state["visible_enemies"]} for actor in state["allies"]}
    state["objective"] = "Eliminate all opposing units before the episode step limit."
    state["map"], state["episode"], state["game_loop"] = "3m", 1, 80
    state["steps_remaining"] = 50
    return {"model": "jev-1.13.0", "state": state,
            "questions": {"u0": {"criteria": {"6": "Attack e0", "7": "Attack e1"}}}}


def choice_answer(choice, keys=("0", "1")):
    probabilities = {key: (0.9 if key == choice else 0.1 / max(1, len(keys) - 1)) for key in keys}
    return {"team_skill": {"type": "choice", "choice": choice, "confidence": 0.9,
                           "probabilities": probabilities}}


class SkillPolicyTests(unittest.TestCase):
    def test_request_is_one_team_question_without_unit_decoding(self):
        request = skill_request(payload(), "jev-1.13.0", 4)
        self.assertEqual(list(request["questions"]), ["team_skill"])
        # Dynamic binding: 0/1/2 fixed modes, explicit targets, adaptive postures appended.
        self.assertEqual(set(request["questions"]["team_skill"]["criteria"]),
                         {"0", "1", "2", "3", "4", "5", "6", "7"})
        self.assertNotIn("questions", request["state"])
        # Only shared team-visible data: no raw hidden state, no per-unit criteria.
        body = json.dumps(request)
        self.assertNotIn("deathless", body)

    def test_decode_skill_accepts_valid_and_rejects_illegal_choices(self):
        request = skill_request(payload(), "jev-1.13.0", 4)
        self.assertEqual(decode_skill(request, choice_answer("0")), (DIRECT_CONTROL, None))
        self.assertEqual(decode_skill(request, choice_answer("1")), (FOCUS_FIRE_BUDGET, None))
        self.assertEqual(decode_skill(request, choice_answer("2")), (COVER_RETREAT, None))
        self.assertEqual(decode_skill(request, choice_answer("3")), (FOCUS_FIRE_BUDGET, "e0"))
        self.assertEqual(decode_skill(request, choice_answer("4")), (FOCUS_FIRE_BUDGET, "e1"))
        with self.assertRaises(ValueError):
            decode_skill(request, choice_answer("9"))
        with self.assertRaises(ValueError):
            decode_skill(request, {"u0": choice_answer("1")})  # unit answer, not team_skill.

    def test_explicit_target_candidates_bind_dynamically(self):
        many = payload()
        # Seven visible enemies: only the first four get explicit candidates.
        from test_jev_focus_fire import enemy
        for j in range(2, 7):
            many["state"]["visible_enemies"].update(enemy(j, 20))
        many["state"]["relations"] = {a: {e: {"distance": 4.0, "in_weapon_range": True,
                                              "smac_attack_available": True}
                                         for e in many["state"]["visible_enemies"]}
                                      for a in many["state"]["allies"]}
        request = skill_request(many, "jev-1.13.0", 4)
        criteria = request["questions"]["team_skill"]["criteria"]
        self.assertEqual(sorted(criteria, key=int),
                         ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])  # + adaptive postures.
        self.assertEqual(decode_skill(request, choice_answer("6")), (FOCUS_FIRE_BUDGET, "e3"))
        # A choice beyond the dynamically bound set is illegal even if numeric.
        with self.assertRaises(ValueError):
            decode_skill(request, choice_answer("10"))

    def test_team_answer_is_never_routed_through_unit_decoding(self):
        from jev_micro.policy import decode_actions
        request = skill_request(payload(), "jev-1.13.0", 4)
        with self.assertRaises(ValueError):
            decode_actions(request, {"team_skill": {"choice": "1"}}, 3)

    def test_choice_validation_rejects_malformed_responses(self):
        request = skill_request(payload(), "jev-1.13.0", 4)
        keys = tuple(request["questions"]["team_skill"]["criteria"])
        clean = {"model": "jev-1.13.0", "answers": choice_answer("1", keys),
                 "usage": {"input_tokens": 10}}
        validated = validate_skill_response(clean, request)
        self.assertEqual(validated["answers"]["team_skill"]["choice"], "1")
        bad = {"model": "jev-1.13.0", "answers": {"team_skill": {"type": "choice", "choice": "9",
                                                                 "confidence": 1, "probabilities": {"9": 1}}},
               "usage": {}}
        with self.assertRaises(JevError):
            validate_skill_response(bad, request)

    def test_attached_plan_is_only_a_digest(self):
        p = payload()
        p["state"]["strategic_plan"] = {"plan_sha256": "abc", "objective": "win",
                                        "target_selection": "weakest", "contingencies": ["secret"]}
        request = skill_request(p, "jev-1.13.0", 4)
        self.assertEqual(request["state"]["strategic_plan"]["objective"], "win")
        self.assertNotIn("contingencies", request["state"]["strategic_plan"])


class SkillManagerTests(unittest.TestCase):
    def test_focus_fire_selection_enters_skill_execution(self):
        manager = SkillManager(max_steps=2)
        manager.select(FOCUS_FIRE_BUDGET)
        self.assertTrue(manager.active)
        actions, telemetry = manager.step(payload(), 3)
        self.assertEqual(len(actions), 3)
        self.assertEqual(telemetry["skill"], "focus_fire_budget")
        self.assertEqual(telemetry["skill_step"], 1)

    def test_skill_runs_multiple_steps_then_terminates(self):
        manager = SkillManager(max_steps=3)
        manager.select(FOCUS_FIRE_BUDGET)
        reasons = []
        for step in range(1, 5):
            actions, telemetry = manager.step(payload(), 3)
            if "termination_reason" in telemetry:
                reasons.append(telemetry["termination_reason"])
                break  # run.py stops stepping a terminated skill.
        self.assertEqual(reasons, ["max_steps"])
        self.assertEqual(manager.stats["steps"], 3)
        self.assertFalse(manager.active)

    def test_direct_control_keeps_original_per_unit_path(self):
        manager = SkillManager(max_steps=4)
        manager.select(DIRECT_CONTROL)
        self.assertFalse(manager.active)  # No local skill actions: JEV decides per unit.
        for _ in range(4):
            manager.spend_mode_step()
        self.assertEqual(manager.mode_remaining, 0)  # Window over: re-ask JEV.

    def test_early_termination_when_no_targets_remain(self):
        manager = SkillManager(max_steps=4)
        manager.select(FOCUS_FIRE_BUDGET)
        empty = payload()
        empty["state"]["visible_enemies"] = {}
        empty["state"]["relations"] = {a: {} for a in empty["state"]["allies"]}
        _, telemetry = manager.step(empty, 3)
        self.assertEqual(telemetry["termination_reason"], "no_visible_targets")
        self.assertFalse(manager.active)

    def test_reassignment_counted_across_steps(self):
        manager = SkillManager(max_steps=4)
        manager.select(FOCUS_FIRE_BUDGET)
        manager.step(payload(), 3)
        weakened = payload()
        weakened["state"]["visible_enemies"]["e0"]["health"] = 0
        weakened["state"]["visible_enemies"]["e0"]["effective_health"] = 0
        manager.step(weakened, 3)
        self.assertGreaterEqual(manager.stats["reassignments"], 1)

    def test_reset_prevents_cross_episode_leak(self):
        manager = SkillManager(max_steps=4)
        manager.select(FOCUS_FIRE_BUDGET)
        manager.step(payload(), 3)
        stats = dict(manager.stats)
        manager.reset()
        self.assertIsNone(manager.current_skill)
        self.assertFalse(manager.active)
        self.assertEqual(manager.target_assignments, {})
        self.assertEqual(manager.mode_remaining, 0)
        # Cumulative stats survive; per-episode diffs are computed against a baseline copy.
        self.assertEqual(manager.stats, stats)

    def test_pending_damage_ledger_stays_disabled_across_steps(self):
        # P0 regression: the manager no longer carries predicted damage into
        # the next step's budget; observations are the single source of truth.
        manager = SkillManager(max_steps=4)
        manager.select(FOCUS_FIRE_BUDGET)
        _, first = manager.step(payload(), 3)
        self.assertGreater(first["predicted_damage"]["e0"], 0)
        _, second = manager.step(payload(), 3)
        self.assertEqual(second["pending_damage_applied"], {})
        self.assertEqual(first["pending_damage_applied"], {})

    def test_explicit_priority_target_flows_into_execution(self):
        manager = SkillManager(max_steps=4)
        manager.select(FOCUS_FIRE_BUDGET, priority_target="e1")  # e1 has more HP than e0.
        self.assertEqual(manager.priority_target, "e1")
        _, telemetry = manager.step(payload(), 3)
        self.assertEqual(telemetry["target_priority"][0], "e1")

    def test_observation_step_and_reassignment_flag(self):
        manager = SkillManager(max_steps=4)
        manager.select(FOCUS_FIRE_BUDGET)
        p = payload()
        p["state"]["game_loop"] = 100
        _, first = manager.step(p, 3)
        self.assertEqual(manager.last_observation_step, 100)
        self.assertFalse(first["reassigned"])
        changed = payload()
        changed["state"]["game_loop"] = 108
        changed["state"]["allies"]["u0"]["alive"] = False  # one attacker dies.
        _, second = manager.step(changed, 3)
        self.assertEqual(manager.last_observation_step, 108)
        self.assertTrue(second["reassigned"])

    def test_skill_execution_never_invokes_the_paid_client(self):
        from jev_micro.client import JevClient
        manager = SkillManager(max_steps=4)
        manager.select(FOCUS_FIRE_BUDGET)
        with patch.object(JevClient, "choose", side_effect=AssertionError("paid call")):
            for _ in range(4):
                manager.step(payload(), 3)  # Purely local: request counters stay untouched.
        self.assertEqual(manager.stats["steps"], 4)


class CliWiringTests(unittest.TestCase):
    def test_run_defaults_keep_original_behavior(self):
        from jev_micro.run import build_parser
        args = build_parser().parse_args([])
        self.assertEqual((args.skill_mode, args.skill_max_steps), ("off", 4))

    def test_skill_mode_jev_requires_jev_policy(self):
        import jev_micro.run as run_module
        with patch.object(run_module, "load_api_key", return_value=None), \
             patch.object(sys, "argv", ["run", "--policy", "nearest", "--skill-mode", "jev"]):
            with self.assertRaises(SystemExit) as raised:
                run_module.main()
        self.assertEqual(raised.exception.code, 2)

    def test_skill_mode_always_accepts_nearest_policy(self):
        from jev_micro.run import build_parser
        self.assertEqual(build_parser().parse_args(["--skill-mode", "always"]).skill_mode, "always")


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
                "skill_mode": "always", "skill": {"activations": 1, "steps": 2, "terminations": {}}}),
                encoding="utf-8")
            return FakeProcess()

        with patch.object(subprocess, "Popen", fake_popen), \
             patch.object(sys, "argv", ["suite", *argv, "--output-dir", str(output)]):
            try:
                suite.main()
            except SystemExit:
                pass
        return output, commands

    def test_suite_forwards_and_records_skill_arguments(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            output, commands = self.run_suite(
                tmp, ["--maps", "3m", "--policies", "nearest", "--episodes", "1",
                      "--skill-mode", "always", "--skill-max-steps", "2"])
            self.assertTrue(any("--skill-mode" in c and "always" in c for c in commands))
            self.assertTrue(any("2" == c[c.index("--skill-max-steps") + 1] for c in commands))
            settings = json.loads((output / "suite.json").read_text(encoding="utf-8"))
            self.assertEqual(settings["skill_mode"], "always")
            self.assertEqual(settings["skill_max_steps"], 2)
            report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["skill_activations"], 1)
            self.assertEqual(report["results"][0]["skill_mode"], "always")

    def test_resume_rejects_mismatched_skill_configuration(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            output, _ = self.run_suite(tmp, ["--maps", "3m", "--policies", "nearest",
                                             "--episodes", "1", "--skill-mode", "always"])
            with patch.object(sys, "argv", ["suite", "--maps", "3m", "--policies", "nearest",
                                            "--episodes", "1", "--skill-mode", "jev",
                                            "--output-dir", str(output), "--resume"]):
                from jev_micro import suite
                with self.assertRaises(SystemExit) as raised:
                    suite.main()
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
