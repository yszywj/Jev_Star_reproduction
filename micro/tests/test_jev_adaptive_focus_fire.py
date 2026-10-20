"""Offline unit tests for AdaptiveFocusFire. No SC2, no API keys."""

import unittest
from unittest.mock import patch

from jev_micro.skills import adaptive_focus_fire
from jev_micro.skills.adaptive_focus_fire import (allocate_adaptive_focus_fire,
                                                  evaluate_retreat_value,
                                                  should_enter_all_in)

from test_jev_cover_retreat import build_state, make_ally, make_enemy

MARINE = 48
PROFILE = {str(MARINE): {"known": True, "armor": 0, "weapons": [
    {"target_type": "ground", "range": 5, "base_damage": 6, "hits_per_attack": 1}]}}


def marine(index, position, health=45, loss=0, ready=True, alive=True):
    return {"u%d" % index: {
        "tag": str(100 + index), "type": "Marine", "alive": alive,
        "health": health, "max_health": 45, "shield": 0, "max_shield": 0,
        "effective_health": health, "unit_type_id": MARINE,
        "position": list(position), "radius": 0.375, "is_flying": False,
        "weapon_ready": ready, "weapon_cooldown_loops": 0 if ready else 5,
        "health_lost_since_previous_decision": loss,
        "shield_lost_since_previous_decision": 0}}


def medivac(index, position, health=150):
    record = marine(index, position, health=health)
    unit = record[f"u{index}"]
    unit["type"] = "Medivac"
    unit["unit_type_id"] = 54
    unit["max_health"] = 150
    return record


def scene(n_allies=8, n_enemies=8, ally_health=45, ally_loss=0, wounded=(),
          cooldown=(), enemy_health=45, ready_default=True):
    allies = {}
    for i in range(n_allies):
        allies.update(marine(i, (10.0 - i * 0.4, 16.0),
                             health=ally_health if i not in wounded else wounded[i],
                             loss=ally_loss if i not in wounded else 0,
                             ready=(i not in cooldown)))
    enemies = {}
    for j in range(n_enemies):
        enemies.update(make_enemy(j, (13.0 + j * 0.3, 16.0), health=enemy_health))
    state = build_state(allies, enemies, profiles=dict(PROFILE))
    for actor in allies:
        for enemy in enemies:
            state["relations"][actor][enemy]["in_weapon_range"] = True
    return state


def wounded_scene():
    """u6/u7 hurt, recently damaged and on cooldown; the rest fully ready."""
    return scene(wounded={6: 12, 7: 10}, cooldown=(6, 7))


class FocusFireDefaultTests(unittest.TestCase):
    def test_no_retreaters_still_uses_focus_fire_budget(self):
        state = scene()  # Everyone healthy: nobody retreats.
        with patch.object(adaptive_focus_fire, "allocate_focus_fire",
                          wraps=adaptive_focus_fire.allocate_focus_fire) as spy:
            result = allocate_adaptive_focus_fire(state, 8)
        self.assertEqual(spy.call_count, 1)  # Never degrades to nearest fallback.
        self.assertEqual(result["phase"], "engage")
        self.assertEqual(result["retreat_units"], [])
        self.assertEqual(len(result["attack_units"]), 8)
        for actor, target in result["attack_assignments"].items():
            self.assertEqual(result["actions"][int(actor[1:])], 6 + int(target[1:]))

    def test_variable_formations_3_8_25(self):
        for n in (3, 8, 25):
            state = scene(n_allies=n, n_enemies=n, enemy_health=30)
            result = allocate_adaptive_focus_fire(state, n)
            self.assertEqual(len(result["actions"]), n)
            self.assertTrue(all(a >= 6 for a in result["actions"]))

    def test_only_one_endangered_unit_rest_keep_focus_fire(self):
        state = wounded_scene()
        result = allocate_adaptive_focus_fire(state, 8)
        self.assertEqual(result["retreat_units"], ["u6", "u7"])
        for actor in ("u6", "u7"):
            self.assertIn(result["actions"][int(actor[1:])], (2, 3, 4, 5))  # Move, not attack.
        self.assertEqual(len(result["attack_units"]), 6)
        # Retreated units never appear in the attack budget.
        self.assertNotIn("u6", result["attack_assignments"])
        self.assertNotIn("u7", result["attack_assignments"])

    def test_budget_spills_to_next_target(self):
        state = scene(n_allies=8, n_enemies=2, enemy_health=12)
        result = allocate_adaptive_focus_fire(state, 8)
        targets = list(result["attack_assignments"].values())
        self.assertIn("e0", targets)
        self.assertIn("e1", targets)  # Budget met on e0, remaining attackers spill.

    def test_dead_units_get_zero_alive_get_legal_actions(self):
        state = scene(n_allies=4, n_enemies=2, enemy_health=30)
        state["allies"]["u3"]["alive"] = False
        state["allies"]["u3"]["health"] = 0
        result = allocate_adaptive_focus_fire(state, 4)
        self.assertEqual(result["actions"][3], 0)
        for index in range(3):
            action = result["actions"][index]
            allowed = ({1} | {int(a) for a in state["movement_candidates"][f"u{index}"]}
                       | {6 + int(e[1:]) for e, r in state["relations"][f"u{index}"].items()
                          if r["smac_attack_available"]})
            self.assertIn(action, allowed)

    def test_medivac_is_support_never_attacker(self):
        state = scene(n_allies=3, n_enemies=2, enemy_health=30)
        state["allies"].update(medivac(2, (9.0, 16.0)))
        state["relations"]["u2"]["e0"]["smac_attack_available"] = False
        state["relations"]["u2"]["e1"]["smac_attack_available"] = False
        state["heal_candidates"] = {"u2": {"6": {"target": "u0", "distance": 2.0,
                                                 "missing_health": 5}}}
        result = allocate_adaptive_focus_fire(state, 3)
        self.assertNotIn("u2", result["attack_units"])
        self.assertEqual(result["roles"]["u2"], "support")
        self.assertEqual(result["actions"][2], 6)  # Heal action, not an attack on e*.

    def test_invisible_enemies_never_enter_tactics(self):
        state = scene(n_allies=2, n_enemies=1)
        state["visible_enemies"] = {}
        state["relations"] = {a: {} for a in state["allies"]}
        for actor, moves in state["movement_candidates"].items():
            for entry in moves.values():
                entry["distances_to_visible_enemies"] = {}
        result = allocate_adaptive_focus_fire(state, 2)
        self.assertEqual(result["attack_assignments"], {})
        self.assertEqual(result["phase"], "engage")
        self.assertTrue(all(a in (1, 2, 3, 4, 5) for a in result["actions"]))


class RetreatValueTests(unittest.TestCase):
    def test_kill_opportunity_keeps_attacker(self):
        state = scene(n_allies=2, n_enemies=1, enemy_health=5)  # 6 damage kills e0.
        value, reason = evaluate_retreat_value(state, "u0", 5, {})
        self.assertFalse(value)
        self.assertEqual(reason, "kill_opportunity")

    def test_cooldown_unit_with_safe_move_may_retreat(self):
        state = scene(n_allies=2, n_enemies=1, cooldown=(0,), wounded={0: 15})
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 20
        value, reason = evaluate_retreat_value(state, "u0", 5,
                                               {"retreat_steps": 0})
        self.assertTrue(value)
        self.assertEqual(reason, "cooldown_or_out_of_range")

    def test_ready_weapon_engaged_keeps_attacker(self):
        state = scene(n_allies=2, n_enemies=1, wounded={0: 20})
        value, reason = evaluate_retreat_value(state, "u0", 5, {})
        self.assertFalse(value)
        self.assertEqual(reason, "weapon_engaged")

    def test_retreat_cap_blocks_further_retreat(self):
        state = scene(n_allies=2, n_enemies=1, cooldown=(0,), wounded={0: 15})
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 20
        value, reason = evaluate_retreat_value(
            state, "u0", 5, {"retreat_steps": 2})
        self.assertFalse(value)
        self.assertEqual(reason, "retreat_cap_reached")

    def test_no_safe_move_blocks_retreat(self):
        state = scene(n_allies=2, n_enemies=1, cooldown=(0,), wounded={0: 15})
        value, reason = evaluate_retreat_value(state, "u0", None, {})
        self.assertFalse(value)
        self.assertEqual(reason, "no_safe_move")

    def test_postures_change_thresholds_not_template(self):
        state = scene(n_allies=2, n_enemies=1, wounded={0: 12})  # ratio ~0.27.
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 16  # 0.36 loss.
        # Defensive: critical ratio 0.35 AND loss 0.15 -> retreat despite weapon.
        self.assertEqual(evaluate_retreat_value(state, "u0", 5, {}, posture="defensive"),
                         (True, "critical_despite_weapon"))
        # Aggressive: critical ratio 0.15 -> keep attacking.
        self.assertEqual(evaluate_retreat_value(state, "u0", 5, {}, posture="aggressive")[1],
                         "weapon_engaged")


class AllInTests(unittest.TestCase):
    def test_no_contact_or_no_visible_enemy_never_triggers(self):
        state = scene(n_allies=8, n_enemies=8)
        enter, _ = should_enter_all_in(state, stall=99, contacted=False, unit_states={},
                                       heal_capacity=False)
        self.assertFalse(enter)  # Opening: enemies may simply not be found yet.
        empty = scene(n_allies=8, n_enemies=0)
        enter, _ = should_enter_all_in(empty, stall=99, contacted=True, unit_states={},
                                       heal_capacity=False)
        self.assertFalse(enter)

    def test_low_team_no_heal_and_stall_triggers_with_reason(self):
        state = scene(n_allies=8, n_enemies=8, ally_health=8)  # Team ratio ~0.18.
        enter, reasons = should_enter_all_in(state, stall=6, contacted=True,
                                             unit_states={}, heal_capacity=False)
        self.assertTrue(enter)
        self.assertIn("no_damage_progress", reasons)
        self.assertIn("low_remaining_force(0.18;alive_ratio=0.18)_no_heal", reasons)

    def test_healthy_team_never_all_in_on_stall_alone(self):
        state = scene(n_allies=8, n_enemies=8)  # ratio 1.0.
        enter, _ = should_enter_all_in(state, stall=6, contacted=True,
                                       unit_states={}, heal_capacity=False)
        self.assertFalse(enter)

    def test_all_in_is_sticky_and_blocks_survival_retreat(self):
        state = scene(wounded={6: 6})  # Desperately hurt unit would normally retreat.
        result = allocate_adaptive_focus_fire(state, 8, all_in=True, contacted=True, stall=6)
        self.assertTrue(result["all_in"])
        self.assertEqual(result["phase"], "all_in")
        self.assertEqual(result["retreat_units"], [])  # No survival retreat in All-in.
        self.assertEqual(len(result["attack_units"]), 8)  # Everyone still fights.
        self.assertEqual(result["all_in_reason"], "sticky")

    def test_all_in_still_allows_attacking_units_to_move_legally(self):
        state = scene(n_allies=2, n_enemies=1)
        state["relations"]["u0"]["e0"]["smac_attack_available"] = False  # u0 out of SMAC range.
        result = allocate_adaptive_focus_fire(state, 2, all_in=True, contacted=True, stall=6)
        self.assertIn(result["actions"][0], (1, 2, 3, 4, 5))  # Legal move toward the fight.

    def test_retreat_loop_triggers_all_in(self):
        state = scene(n_allies=8, n_enemies=8, ally_health=40)
        unit_states = {f"u{i}": {"role": "retreat", "retreat_steps": 2, "retreat_cycles": 4}
                       for i in range(8)}
        enter, reasons = should_enter_all_in(state, stall=0, contacted=True,
                                             unit_states=unit_states, heal_capacity=False)
        self.assertTrue(enter)
        self.assertIn("retreat_loop", reasons)


class EightMFailureMechanismTests(unittest.TestCase):
    """Regressions for the two observed 8m failure modes."""

    def test_two_hurt_marines_retreat_then_reengage(self):
        # Mechanism: 8 marines engaged, u6/u7 taking damage -> brief retreat,
        # the other six keep the unified budget; once u6's weapon is ready
        # again with a legal target it returns to the fire plan.
        state = wounded_scene()
        unit_states = {}
        first = allocate_adaptive_focus_fire(state, 8, unit_states=unit_states)
        self.assertEqual(first["retreat_units"], ["u6", "u7"])
        self.assertEqual(len(first["attack_units"]), 6)
        for actor in ("u6", "u7"):
            unit_states[actor] = {"role": "retreat", "retreat_steps": 1, "retreat_cycles": 0}
        # Next step: cooldown expired, weapon ready, still endangered but
        # kill-capable -> re-engage rather than keep running.
        recovered = scene(n_allies=8, n_enemies=8, wounded={6: 12, 7: 10}, enemy_health=5)
        second = allocate_adaptive_focus_fire(recovered, 8, unit_states=unit_states)
        self.assertEqual(second["retreat_units"], [])
        self.assertIn("u6", second["reengaged"])
        self.assertIn("u7", second["reengaged"])

    def test_last_survivor_enters_all_in_instead_of_infinite_retreat(self):
        # Mechanism: one 3 HP marine, multiple enemies, no healers, repeated
        # retreats with no observable damage progress.
        state = scene(n_allies=1, n_enemies=5, wounded={0: 3}, cooldown=(0,))
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 42
        unit_states = {"u0": {"role": "retreat", "retreat_steps": 2, "retreat_cycles": 4}}
        result = allocate_adaptive_focus_fire(state, 1, unit_states=unit_states,
                                              contacted=True, stall=6)
        self.assertTrue(result["all_in"])  # Not another pointless retreat.
        self.assertEqual(result["retreat_units"], [])
        self.assertEqual(result["phase"], "all_in")
        # All-in still attacks legally when possible: no cooldown here means
        # u0 rejoins the fire plan instead of running forever.
        self.assertEqual(result["roles"]["u0"], "all_in_attack")
        self.assertEqual(result["actions"][0], 6)

    def test_valuable_repositioning_not_blindly_forbidden(self):
        # One survivor on cooldown with a safe move: brief retreat is still
        # allowed — low health alone must not force All-in posture moves.
        state = scene(n_allies=1, n_enemies=5, wounded={0: 10}, cooldown=(0,))
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 20
        result = allocate_adaptive_focus_fire(state, 1, contacted=True, stall=0)
        self.assertFalse(result["all_in"])
        self.assertEqual(result["retreat_units"], ["u0"])


class RetreatCapTests(unittest.TestCase):
    def cooldown_state(self, steps):
        state = scene(n_allies=2, n_enemies=1, cooldown=(0,), wounded={0: 15})
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 20
        return state, {"role": "retreat", "retreat_steps": steps, "retreat_cycles": 0}

    def test_max_retreat_steps_parameter_actually_applies(self):
        for cap, steps, expected in ((1, 1, False), (2, 1, True), (3, 2, True),
                                     (1, 0, True), (3, 3, False)):
            state, unit_state = self.cooldown_state(steps)
            value, reason = evaluate_retreat_value(
                state, "u0", 5, unit_state, parameters={"max_retreat_steps": cap})
            self.assertEqual(value, expected, f"cap={cap}, steps={steps}")
            if not expected and steps > 0:
                self.assertEqual(reason, "retreat_cap_reached")

    def test_critical_ready_unit_does_not_bypass_cap(self):
        # P0 regression: the critical-despite-weapon branch must obey the cap.
        state = scene(n_allies=2, n_enemies=1, enemy_health=45, wounded={0: 5})
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 40  # Critical + heavy loss.
        unit_state = {"role": "retreat", "retreat_steps": 2, "retreat_cycles": 0}
        value, reason = evaluate_retreat_value(state, "u0", 5, unit_state)
        self.assertFalse(value)
        self.assertEqual(reason, "retreat_cap_reached")

    def test_retreat_cap_enforced_in_full_allocation(self):
        state = wounded_scene()
        unit_states = {"u6": {"role": "retreat", "retreat_steps": 2, "retreat_cycles": 0}}
        result = allocate_adaptive_focus_fire(state, 8, unit_states=unit_states,
                                              parameters={"max_retreat_steps": 2})
        self.assertNotIn("u6", result["retreat_units"])  # Capped: back to the fire plan.
        self.assertIn("u7", result["retreat_units"])     # Fresh unit still dodges.


class AllInRedesignTests(unittest.TestCase):
    def test_four_productive_retreats_do_not_trigger(self):
        # Four DIFFERENT units each completed one normal retreat with damage
        # progress in between: cycles stay zero (old code summed 4 and tripped).
        state = scene(n_allies=8, n_enemies=8)
        unit_states = {f"u{i}": {"role": "attack", "retreat_steps": 0, "retreat_cycles": 0}
                       for i in range(8)}
        enter, reasons = should_enter_all_in(state, stall=0, contacted=True,
                                             unit_states=unit_states, heal_capacity=False)
        self.assertFalse(enter)

    def test_single_unit_unproductive_loop_triggers(self):
        state = scene(n_allies=8, n_enemies=8)
        unit_states = {f"u{i}": {"role": "attack", "retreat_steps": 0, "retreat_cycles": 0}
                       for i in range(8)}
        unit_states["u3"]["retreat_cycles"] = 3  # One unit, three fruitless retreats.
        enter, reasons = should_enter_all_in(state, stall=0, contacted=True,
                                             unit_states=unit_states, heal_capacity=False)
        self.assertTrue(enter)
        self.assertIn("unproductive_retreat_loop(unit_cycles=3)", reasons)

    def test_remaining_force_vs_alive_ratio_are_distinct(self):
        # 8 marines down to 1 full-HP survivor: alive ratio 1.0 but remaining
        # force 0.125 — the All-in check must use the roster-based metric.
        state = scene(n_allies=8, n_enemies=8)
        for i, actor in enumerate(sorted(state["allies"])):
            unit = state["allies"][actor]
            if i > 0:
                unit["alive"] = False
                unit["health"] = 0
        enter, reasons = should_enter_all_in(state, stall=6, contacted=True,
                                             unit_states={}, heal_capacity=False)
        self.assertTrue(enter)
        self.assertIn("alive_ratio=1.00", reasons)  # Both metrics are reported.

    def test_stall_alone_with_strong_force_still_no_all_in(self):
        state = scene(n_allies=8, n_enemies=8, ally_health=45)
        enter, _ = should_enter_all_in(state, stall=9, contacted=True,
                                       unit_states={}, heal_capacity=False)
        self.assertFalse(enter)

    def test_disable_retreat_ablation_keeps_unified_focus_fire(self):
        state = wounded_scene()
        result = allocate_adaptive_focus_fire(state, 8, parameters={"disable_retreat": True})
        self.assertEqual(result["retreat_units"], [])
        self.assertEqual(len(result["attack_units"]), 8)  # Still one budget, no fallbacks.
        self.assertTrue(all(a >= 6 for a in result["actions"]))

    def test_disable_all_in_ablation_never_enters(self):
        state = scene(n_allies=1, n_enemies=5, wounded={0: 3})
        unit_states = {"u0": {"role": "retreat", "retreat_steps": 2, "retreat_cycles": 9}}
        result = allocate_adaptive_focus_fire(state, 1, unit_states=unit_states,
                                              contacted=True, stall=9,
                                              parameters={"disable_all_in": True})
        self.assertFalse(result["all_in"])  # No back door: even sticky input is ignored.
        result_sticky = allocate_adaptive_focus_fire(state, 1, all_in=True, contacted=True,
                                                     stall=9,
                                                     parameters={"disable_all_in": True})
        self.assertFalse(result_sticky["all_in"])


if __name__ == "__main__":
    unittest.main()
