"""Regression cases distilled from the paired Protoss trajectory analysis."""
import copy
import math
import unittest

from jev_micro.skill_manager import SkillManager
from jev_micro.skills.adaptive_focus_fire import allocate_adaptive_focus_fire
from jev_micro.skills.focus_fire import allocate_focus_fire, fallback_action
from jev_micro.skills.movement_scoring import predicted_move, select_movement
from test_jev_adaptive_integration import payload
from test_jev_adaptive_focus_fire import scene
from test_jev_cover_retreat import make_ally, make_enemy, build_state

FLAGS = {"window_fire": True, "range_preserving_move": True, "target_scoring": True}


def stalker(distance=5.7, cooldown=20):
    allies = make_ally(0, (10, 10), health=37, max_health=80, max_shield=80, type_id=74)
    allies["u0"].update(type="Stalker", radius=.625,
                        weapon_cooldown_loops=cooldown, weapon_ready=cooldown == 0)
    enemies = make_enemy(0, (10 + distance, 10), health=80, type_id=74)
    enemies["e0"].update(type="Stalker", radius=.625, max_health=80, max_shield=80)
    state = build_state(allies, enemies, profiles={"74": {
        "known": True, "armor": 1, "movement_speed_native": 2.953125,
        "weapons": [{"target_type": "any", "range": 6, "base_damage": 13,
                     "hits_per_attack": 1}]}})
    state["step_mul"] = 8
    state["relations"]["u0"]["e0"].update(
        native_weapon_range=6, smac_attack_distance_limit=6,
        smac_attack_available=distance <= 6, in_weapon_range=distance <= 7.25)
    return state


class TrajectoryFixTests(unittest.TestCase):
    def test_reachable_destination_uses_speed_and_window(self):
        state = stalker()
        prediction = predicted_move(state, "u0", 5)
        self.assertAlmostEqual(math.dist(state["allies"]["u0"]["position"],
                                        prediction["destination"]), 2.953125 * 8 / 16)
        self.assertEqual(state["movement_candidates"]["u0"]["5"]["destination"], [8, 10])
        state["step_mul"] = 16
        self.assertEqual(predicted_move(state, "u0", 5)["destination"], [8, 10])

    def test_tactical_does_not_cross_mask_while_still_inside_enemy_range(self):
        state = stalker(distance=5.1)
        state["movement_candidates"]["u0"] = {"5": state["movement_candidates"]["u0"]["5"]}
        action, reason, diagnostic = select_movement(state, "u0", "e0", "tactical")
        self.assertIsNone(action)
        self.assertEqual(reason, "keep_attack_position")
        self.assertEqual(diagnostic["rejected_candidates"]["5"],
                         "loses_attack_without_reducing_exposure")

    def test_recovery_is_bounded_non_reversing_then_reenters_when_ready(self):
        state = stalker(distance=6.3)
        history = {"u0": {"role": "retreat", "retreat_steps": 1, "retreat_cycles": 0}}
        result = allocate_adaptive_focus_fire(
            state, 1, unit_states=history, parameters=FLAGS, move_history={"u0": 5})
        self.assertEqual(result["roles"]["u0"], "recover")
        self.assertNotEqual(result["actions"][0], 4)  # No instant eastward reversal.
        history["u0"].update(role="recover", recover_steps=2)
        capped = allocate_adaptive_focus_fire(
            state, 1, unit_states=history, parameters=FLAGS, move_history={"u0": 5})
        self.assertEqual(capped["roles"]["u0"], "reposition")
        state["allies"]["u0"]["weapon_cooldown_loops"] = 0
        history["u0"].update(role="retreat", recover_steps=0)
        ready = allocate_adaptive_focus_fire(
            state, 1, unit_states=history, parameters=FLAGS, move_history={"u0": 5})
        self.assertEqual(ready["roles"]["u0"], "reposition")

    def test_six_hp_finisher_precedes_target_with_more_attackers(self):
        state = scene(n_allies=2, n_enemies=2, enemy_health=40)
        state["visible_enemies"]["e1"].update(health=6, effective_health=6)
        state["relations"]["u0"]["e1"]["smac_attack_available"] = False
        legacy = allocate_focus_fire(state, 2)
        improved = allocate_focus_fire(state, 2, parameters=FLAGS)
        self.assertEqual(legacy["assignments"]["u1"], "e0")
        self.assertEqual(improved["assignments"]["u1"], "e1")
        self.assertEqual(improved["assignments"]["u0"], "e0")

    def test_delayed_damage_clears_cycles_before_all_in_on_recovery_frame(self):
        manager = SkillManager(combat_params=FLAGS)
        manager.select("adaptive_focus_fire")
        state = stalker(distance=6.3)
        manager._last_enemy_health = {"e0": 98}
        manager.contacted = True
        manager.unit_states = {"u0": {"role": "retreat", "retreat_steps": 1,
                                       "retreat_cycles": 3}}
        manager._last_move = {"u0": 5}
        _, telemetry = manager.step(payload(state), 1)  # e0 actually lost 18 HP.
        self.assertTrue(telemetry["observed_team_damage_progress"])
        self.assertFalse(telemetry["all_in"])
        self.assertEqual(telemetry["roles"]["u0"], "recover")
        self.assertEqual(manager.unit_states["u0"]["retreat_cycles"], 0)

    def test_last_seen_search_never_fabricates_an_attack_and_expires(self):
        manager = SkillManager(combat_params=FLAGS)
        manager.select("adaptive_focus_fire")
        state = stalker()
        manager.step(payload(state), 1)
        hidden = copy.deepcopy(state)
        hidden["visible_enemies"], hidden["relations"]["u0"] = {}, {}
        for move in hidden["movement_candidates"]["u0"].values():
            move["distances_to_visible_enemies"] = {}
        request = payload(hidden)
        request["state"]["game_loop"] = 88
        actions, telemetry = manager.step(request, 1)
        self.assertEqual(telemetry["search_hint"]["enemy"], "e0")
        self.assertLess(actions[0], 6)
        self.assertEqual(request["state"]["visible_enemies"], {})
        request["state"]["game_loop"] = 120
        _, telemetry = manager.step(request, 1)
        self.assertIsNone(telemetry["search_hint"])
        manager.reset()
        self.assertEqual(manager._last_seen_enemies, {})

    def test_adaptive_window_reactivation_preserves_focus(self):
        manager = SkillManager(max_steps=1, combat_params=FLAGS)
        manager.select("adaptive_focus_fire")
        manager.step(payload(), 4)
        previous = dict(manager.target_assignments)
        manager.select("adaptive_focus_fire")
        self.assertEqual(manager.target_assignments, previous)
        manager.reset()
        self.assertEqual(manager.target_assignments, {})


if __name__ == "__main__":
    unittest.main()
