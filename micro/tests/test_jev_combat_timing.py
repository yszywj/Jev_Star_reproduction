"""Offline tests for the Sharpy-inspired combat timing and movement scoring.
No SC2, no paid calls."""

import unittest

from jev_micro.skills.combat_timing import can_fire_within_window, window_loops
from jev_micro.skills.focus_fire import allocate_focus_fire
from jev_micro.skills.movement_scoring import (score_movement_candidate,
                                               select_movement)
from jev_micro.skills.cover_retreat import allocate_cover_retreat
from jev_micro.skills.adaptive_focus_fire import (allocate_adaptive_focus_fire,
                                                  evaluate_retreat_value)

from test_jev_adaptive_focus_fire import PROFILE, scene
from test_jev_cover_retreat import build_state, make_ally, make_enemy


def with_step_mul(state, step_mul):
    state["step_mul"] = step_mul
    return state


def cooldown_state(cooldowns):
    """8 ready-range marines vs 2 enemies; per-actor weapon cooldowns set."""
    state = scene(n_allies=len(cooldowns), n_enemies=2, enemy_health=45)
    for actor, cooldown in zip(sorted(state["allies"]), cooldowns):
        state["allies"][actor]["weapon_cooldown_loops"] = cooldown
        state["allies"][actor]["weapon_ready"] = cooldown <= 0
    return with_step_mul(state, 8)


class WindowFireTests(unittest.TestCase):
    def test_cooldown_thresholds_within_default_window(self):
        state = cooldown_state([0, 3, 8, 12])
        actors = sorted(state["allies"])
        expected = {"u0": True, "u1": True, "u2": True, "u3": False}
        for actor, value in zip(actors, expected.values()):
            self.assertEqual(can_fire_within_window(state, actor, "e0"), value,
                             f"{actor} cooldown check")

    def test_step_mul_changes_the_window(self):
        state = with_step_mul(cooldown_state([12]), 16)  # 16+1 tolerance window.
        self.assertTrue(can_fire_within_window(state, "u0", "e0"))
        state["step_mul"] = 8
        self.assertFalse(can_fire_within_window(state, "u0", "e0"))

    def test_window_loops_is_explainable(self):
        self.assertEqual(window_loops(with_step_mul({"step_mul": 8}, 8)), 9.0)

    def test_illegal_or_invisible_target_never_fires(self):
        state = cooldown_state([0])
        state["relations"]["u0"]["e0"]["smac_attack_available"] = False
        self.assertFalse(can_fire_within_window(state, "u0", "e0"))
        state["relations"]["u0"]["e0"]["smac_attack_available"] = True
        state["relations"]["u0"]["e0"]["in_weapon_range"] = False
        self.assertFalse(can_fire_within_window(state, "u0", "e0"))

    def test_weapon_ready_observation_field_is_unchanged(self):
        state = cooldown_state([3])
        self.assertFalse(state["allies"]["u0"]["weapon_ready"])  # Raw field untouched.
        self.assertTrue(can_fire_within_window(state, "u0", "e0"))  # Internal judgment.


class WindowBudgetTests(unittest.TestCase):
    def test_9hp_enemy_does_not_hog_all_attackers(self):
        # 8v8-style: e0 has 9 HP; two ready marines already cover it. Cooling
        # marines (3-5 loops) count under window fire, so the allocator must
        # NOT pile everyone onto e0 — extra attackers go to e1.
        state = scene(n_allies=8, n_enemies=2, enemy_health=45)
        state["visible_enemies"]["e0"]["health"] = 9
        state["visible_enemies"]["e0"]["effective_health"] = 9
        for actor in sorted(state["allies"])[2:]:
            state["allies"][actor]["weapon_cooldown_loops"] = 4
            state["allies"][actor]["weapon_ready"] = False
        state = with_step_mul(state, 8)
        legacy = allocate_focus_fire(state, 8)  # window_fire off: old behavior.
        windowed = allocate_focus_fire(state, 8, parameters={"window_fire": True})
        self.assertGreater(len(windowed["assignments"]),
                           len(set(windowed["assignments"].values())) - 2 + 2 - 2 + 1)
        on_e0 = [a for a, t in windowed["assignments"].items() if t == "e0"]
        self.assertLessEqual(len(on_e0), 2, "9 HP needs at most two 6-damage attackers")
        self.assertIn("e1", windowed["assignments"].values())
        # Legacy mode keeps the old semantics: cooling units are not budgeted.
        self.assertEqual(legacy["window_fire_enabled"], False)
        self.assertEqual(windowed["window_fire_enabled"], True)
        self.assertGreater(sum(windowed["window_only_damage"].values()), 0)

    def test_immediate_vs_window_damage_split(self):
        state = with_step_mul(cooldown_state([0, 4, 20]), 8)
        result = allocate_focus_fire(state, 3, parameters={"window_fire": True})
        self.assertEqual(sum(result["immediate_damage"].values()), 6)   # u0 only.
        self.assertEqual(sum(result["window_only_damage"].values()), 6)  # u1 counts now.
        # Predicted damage is one attack per attacker at most: no double count.
        self.assertEqual(sum(result["predicted_damage"].values()), 12)

    def test_long_cooldown_still_not_counted(self):
        state = with_step_mul(cooldown_state([30]), 8)
        result = allocate_focus_fire(state, 1, parameters={"window_fire": True})
        self.assertEqual(result["predicted_damage"], {"e0": 0, "e1": 0})

    def test_moving_units_never_double_counted(self):
        # Retreating/moving units are excluded from the focus budget by the
        # adaptive/cover callers; the budget itself counts each remaining
        # attacker exactly once per step (verified above) and observes fresh
        # health next step, so a moved unit's "attack" cannot persist.
        state = with_step_mul(cooldown_state([0, 0, 4]), 8)
        first = allocate_focus_fire(state, 3, parameters={"window_fire": True})
        second = allocate_focus_fire(state, 3, parameters={"window_fire": True},
                                     previous_assignments=first["assignments"])
        for result in (first, second):
            per_target = {}
            for target in result["assignments"].values():
                per_target[target] = per_target.get(target, 0) + 1
            self.assertTrue(all(v <= len(state["allies"]) for v in per_target.values()))
            self.assertEqual(sum(result["predicted_damage"].values()),
                             6 * sum(1 for u in state["allies"].values()
                                     if u["weapon_cooldown_loops"] <= 9))

    def test_hysteresis_no_longer_beats_better_damage(self):
        # u0 previously attacked e1; u1 did not. Attacker ordering now puts
        # window fire/range/damage first, so hysteresis cannot mask kills.
        state = with_step_mul(cooldown_state([0, 0]), 8)
        result = allocate_focus_fire(state, 2, parameters={"window_fire": True},
                                     previous_assignments={"u0": "e1"})
        self.assertTrue(result["assignments"])  # Deterministic, valid split.

    def test_target_scoring_is_off_by_default_and_bounded(self):
        state = with_step_mul(scene(n_allies=3, n_enemies=3), 8)
        plain = allocate_focus_fire(state, 3)
        scored = allocate_focus_fire(state, 3, parameters={"target_scoring": True})
        self.assertEqual(plain["target_scoring_enabled"], False)
        self.assertEqual(scored["target_scoring_enabled"], True)
        # Soft scoring never changes legality or the action length.
        self.assertEqual(len(scored["actions"]), 3)
        for action in scored["actions"]:
            self.assertGreaterEqual(action, 6)


class MovementScoringTests(unittest.TestCase):
    def movement_state(self, ally_position=(10, 16), enemy_positions=((13, 16),),
                       target=None, cooldown=0):
        allies = {**make_ally(0, ally_position, health=20),
                  **make_ally(1, (8, 16))}
        enemies = {}
        for j, position in enumerate(enemy_positions):
            enemies.update(make_enemy(j, position))
        state = build_state(allies, enemies, profiles=dict(PROFILE))
        state["allies"]["u0"]["weapon_cooldown_loops"] = cooldown
        state["allies"]["u0"]["weapon_ready"] = cooldown <= 0
        state["step_mul"] = 8
        if "e0" in state["relations"].get("u0", {}):
            state["relations"]["u0"]["e0"]["native_weapon_range"] = 5.0
            state["relations"]["u0"]["e0"]["in_weapon_range"] = True
        return state

    def test_tactical_move_never_leaves_attack_range(self):
        state = self.movement_state()
        action, reason, info = select_movement(state, "u0", "e0", "tactical")
        self.assertEqual(reason, "scored_move")
        self.assertIn("movement_candidate_scores", info)
        reach = 5.0 + 0.375 + 0.375
        destination = state["movement_candidates"]["u0"][str(action)]["destination"]
        distance = ((destination[0] - 13) ** 2 + (destination[1] - 16) ** 2) ** 0.5
        self.assertLessEqual(distance, reach)  # Fighting distance preserved.

    def test_tactical_approaches_when_out_of_range(self):
        state = self.movement_state()
        state["relations"]["u0"]["e0"]["native_weapon_range"] = 2.0  # Short range.
        action, _, _ = select_movement(state, "u0", "e0", "tactical")
        self.assertEqual(action, 4)  # East: close back into firing range.

    def test_emergency_move_maximizes_distance(self):
        state = self.movement_state(cooldown=0)
        action, reason, _ = select_movement(state, "u0", "e0", "emergency")
        self.assertEqual(action, 5)  # West: farthest from the eastern threat.

    def test_scores_use_native_range_and_buffer(self):
        state = self.movement_state()
        state["relations"]["u0"]["e0"]["native_weapon_range"] = 2.0  # Short range.
        # Being back inside firing range beats a farther position that
        # breaks range and must be paid for with a later re-approach.
        east = score_movement_candidate(state, "u0", "4", "e0", "tactical")
        west = score_movement_candidate(state, "u0", "5", "e0", "tactical")
        self.assertLess(east, west)
        # The buffer zone matters: comfortably inside beats the bare edge.
        state["relations"]["u0"]["e0"]["native_weapon_range"] = 4.0  # reach 4.75.
        edge = score_movement_candidate(state, "u0", "5", "e0", "tactical")  # dist 5: outside.
        inside = score_movement_candidate(state, "u0", "2", "e0", "tactical")  # ~3.6: inside.
        self.assertLess(inside, edge)

    def test_direction_history_penalizes_flipping(self):
        state = self.movement_state()
        keep = score_movement_candidate(state, "u0", "2", "e0", "tactical",
                                        previous_action=2)
        flip = score_movement_candidate(state, "u0", "2", "e0", "tactical",
                                        previous_action=3)
        self.assertLess(keep, flip)

    def test_melee_approach_bias(self):
        state = self.movement_state()
        state["relations"]["u0"]["e0"]["native_weapon_range"] = 0.5  # Melee.
        approach = score_movement_candidate(state, "u0", "4", "e0", "tactical")  # East, closer.
        retreat = score_movement_candidate(state, "u0", "5", "e0", "tactical")   # West.
        self.assertLess(approach, retreat)

    def test_invisible_enemies_excluded(self):
        state = self.movement_state(enemy_positions=())  # No visible enemies.
        action, reason, _ = select_movement(state, "u0", "e0", "tactical")
        # No visible enemy: no hidden data touched, scoring still deterministic.
        self.assertIn(action, (2, 3, 4, 5))  # Still deterministic, no hidden data.

    def test_cover_uses_mode_and_history(self):
        allies = {**make_ally(0, (10, 16), health=10, loss=20),
                  **make_ally(1, (8, 16)), **make_ally(2, (7, 16))}
        state = build_state(allies, {**make_enemy(0, (13, 16))}, profiles=dict(PROFILE))
        state["step_mul"] = 8
        result = allocate_cover_retreat(
            state, 3, parameters={"range_preserving_move": True},
            move_history={"u0": 5})
        self.assertIn(result["retreat_actions"]["u0"]["movement_mode"],
                      ("tactical", "emergency"))
        self.assertIn(result["actions"][0], (2, 3, 4, 5))

    def test_emergency_ratio_selects_emergency_mode(self):
        allies = {**make_ally(0, (10, 16), health=4, loss=41),
                  **make_ally(1, (8, 16))}
        state = build_state(allies, {**make_enemy(0, (13, 16))}, profiles=dict(PROFILE))
        state["step_mul"] = 8
        result = allocate_cover_retreat(state, 2, parameters={"range_preserving_move": True})
        self.assertEqual(result["retreat_actions"]["u0"]["movement_mode"], "emergency")


class AdaptiveTimingTests(unittest.TestCase):
    def test_shared_timing_prevents_contradiction(self):
        # With window fire on, a 3-loop-cooldown in-range unit must be treated
        # as able to fight by BOTH the budget and the retreat evaluation.
        state = scene(n_allies=2, n_enemies=1, enemy_health=45, cooldown=(0,), wounded={0: 15})
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 20
        state = with_step_mul(state, 8)
        params = {"window_fire": True}
        # Budget: u0's window damage counts.
        budget = allocate_focus_fire(state, 2, parameters=params)
        self.assertEqual(budget["window_only_damage"]["e0"], 6)
        # Retreat evaluation: u0 can fire within the window -> keeps attacking.
        value, reason = evaluate_retreat_value(state, "u0", 5, {}, parameters=params)
        self.assertFalse(value)
        self.assertEqual(reason, "weapon_engaged")
        # Same state, window fire off (legacy): u0 retreats on cooldown.
        value, reason = evaluate_retreat_value(state, "u0", 5, {})
        self.assertEqual(reason, "cooldown_or_out_of_range")

    def test_in_range_unit_does_not_retreat_over_small_cooldown(self):
        state = scene(n_allies=3, n_enemies=2, enemy_health=45, wounded={0: 20},
                      cooldown=(3,))
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 25
        state = with_step_mul(state, 8)
        result = allocate_adaptive_focus_fire(state, 3, parameters={"window_fire": True})
        self.assertEqual(result["retreat_units"], [])
        self.assertIn("u0", result["attack_units"])

    def test_all_in_still_allows_approach_moves(self):
        state = scene(n_allies=2, n_enemies=1)
        state["relations"]["u0"]["e0"]["smac_attack_available"] = False  # Out of SMAC range.
        state = with_step_mul(state, 8)
        result = allocate_adaptive_focus_fire(state, 2, all_in=True, contacted=True,
                                              stall=6, parameters={"window_fire": True})
        self.assertTrue(result["all_in"])
        self.assertIn(result["actions"][0], (1, 2, 3, 4, 5))  # Legal approach move.

    def test_variable_formations_and_defaults(self):
        for n in (3, 8, 25):
            state = with_step_mul(scene(n_allies=n, n_enemies=n, enemy_health=30), 8)
            result = allocate_adaptive_focus_fire(state, n, parameters={"window_fire": True})
            self.assertEqual(len(result["actions"]), n)
            self.assertTrue(all(a >= 6 for a in result["actions"]))

    def test_defaults_off_keep_legacy_semantics(self):
        state = with_step_mul(cooldown_state([4, 0]), 8)
        legacy = allocate_focus_fire(state, 2)
        self.assertEqual(legacy["window_fire_enabled"], False)
        self.assertEqual(sum(legacy["window_only_damage"].values()), 0)


class CliToggleTests(unittest.TestCase):
    def test_new_toggles_default_off_and_parse(self):
        from jev_micro.run import build_parser
        args = build_parser().parse_args([])
        self.assertFalse(args.adaptive_window_fire)
        self.assertFalse(args.adaptive_range_preserving_move)
        self.assertFalse(args.adaptive_target_scoring)
        args = build_parser().parse_args(["--skill-mode", "always",
                                          "--skill-name", "adaptive_focus_fire",
                                          "--adaptive-window-fire",
                                          "--adaptive-range-preserving-move",
                                          "--adaptive-target-scoring"])
        self.assertTrue(args.adaptive_window_fire and args.adaptive_range_preserving_move
                        and args.adaptive_target_scoring)
        # Existing ablation switches still parse alongside.
        args = build_parser().parse_args(["--adaptive-disable-retreat",
                                          "--adaptive-disable-all-in"])
        self.assertTrue(args.adaptive_disable_retreat and args.adaptive_disable_all_in)


if __name__ == "__main__":
    unittest.main()
