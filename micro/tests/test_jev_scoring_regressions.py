"""Regressions for opt-in scoring, standing baseline and persisted diagnostics."""
import json
import unittest
from unittest.mock import patch

from jev_micro.skills.focus_fire import _attacker_order
from jev_micro.skills.movement_scoring import select_movement
from jev_micro.skills.adaptive_focus_fire import allocate_adaptive_focus_fire
from jev_micro.skills.cover_retreat import select_retreat_action
from jev_micro.skill_manager import SkillManager
import test_jev_combat_timing as timing_tests
from test_jev_adaptive_focus_fire import wounded_scene
from test_jev_adaptive_integration import payload


class ScoringRegressionTests(unittest.TestCase):
    def test_tactical_rejects_all_worse_moves_and_emergency_can_leave(self):
        state = timing_tests.MovementScoringTests().movement_state(
            enemy_positions=((12, 16), (8, 16), (10, 18), (10, 14)))
        action, reason, info = select_movement(
            state, "u0", "e0", "tactical", parameters={"support_weight": 0})
        self.assertIsNone(action)
        self.assertEqual(reason, "keep_attack_position")
        self.assertTrue(all(s > info["standing_score"]
                            for s in info["movement_candidate_scores"].values()))
        action, _, info = select_movement(
            state, "u0", "e0", "emergency", parameters={"support_weight": 0})
        self.assertIn(str(action), state["movement_candidates"]["u0"])
        self.assertEqual(info["selected_action"], action)

    def test_ties_preserve_attack_and_range_diagnostic_is_geometric(self):
        state = timing_tests.MovementScoringTests().movement_state()
        weights = dict.fromkeys(("threat_weight", "danger_penalty", "support_weight",
                                 "fire_opportunity_weight", "reentry_cost",
                                 "direction_change_cost"), 0)
        action, reason, _ = select_movement(state, "u0", "e0", "tactical",
                                            parameters=weights)
        self.assertIsNone(action)
        self.assertEqual(reason, "keep_attack_position")
        state["relations"]["u0"]["e0"]["native_weapon_range"] = 2.5
        state["movement_candidates"]["u0"] = {
            "5": state["movement_candidates"]["u0"]["5"]}
        _, _, info = select_movement(state, "u0", "e0", "emergency")
        self.assertTrue(info["standing_fire_opportunity"])
        self.assertFalse(info["keeps_fire_opportunity"])

    def test_default_order_exactly_preserves_hysteresis_before_readiness(self):
        state = timing_tests.MovementScoringTests().movement_state(cooldown=12)
        relation = state["relations"]["u0"]["e0"]
        previous = {"u0": "e0"}
        self.assertEqual(_attacker_order(state, "u0", "e0", relation, 6, previous),
                         (0, 0, 1, -6, 0))
        self.assertEqual(_attacker_order(state, "u0", "e0", relation, 6, previous,
                                         {"range_preserving_move": True}),
                         (0, 0, 1, -6, 0))
        self.assertNotEqual(_attacker_order(state, "u0", "e0", relation, 6, previous,
                                            {"window_fire": True}),
                            (0, 0, 1, -6, 0))

    def test_adaptive_passes_flags_history_and_manager_serializes_diagnostics(self):
        state = wounded_scene()
        state["step_mul"] = 8
        params = {"range_preserving_move": True, "window_fire": True,
                  "target_scoring": True}
        with patch("jev_micro.skills.adaptive_focus_fire.select_retreat_action",
                   wraps=select_retreat_action) as select:
            result = allocate_adaptive_focus_fire(
                state, 8, parameters=params, move_history={"u6": 5})
        self.assertTrue(select.called)
        call = next(c for c in select.call_args_list if c.args[1] == "u6")
        self.assertTrue(call.args[3]["range_preserving_move"])
        self.assertEqual(call.kwargs["previous_action"], 5)
        self.assertTrue(result["movement_diagnostics"])
        manager = SkillManager(4, combat_params=params)
        manager.select("adaptive_focus_fire", posture="balanced")
        _, telemetry = manager.step(payload(state), 8)
        event = json.loads(json.dumps({"event": "skill_step", **telemetry}))
        for key in ("window_fire_enabled", "range_preserving_move_enabled",
                    "target_scoring_enabled"):
            self.assertTrue(event[key])
        for key in ("immediate_damage", "window_only_damage", "movement_diagnostics"):
            self.assertIn(key, event)
        self.assertTrue(event["movement_diagnostics"])
        self.assertGreater(manager.stats["movement_evaluations"], 0)


if __name__ == "__main__":
    unittest.main()
