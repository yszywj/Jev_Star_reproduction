"""Offline unit tests for the FocusFireBudget allocator. No SC2, no API keys."""

import unittest

from jev_micro.skills import allocate_focus_fire, fallback_action
from jev_micro.skills.focus_fire import target_priority

MARINE = 48
PROFILE = {str(MARINE): {"known": True, "armor": 0, "weapons": [
    {"target_type": "ground", "range": 5, "base_damage": 6, "hits_per_attack": 1}]}}


def ally(index, position=(10.0, 16.0), weapon_ready=True, alive=True, type_id=MARINE):
    return {"u%d" % index: {
        "tag": str(100 + index), "type": "Marine", "alive": alive,
        "health": 45 if alive else 0, "max_health": 45, "shield": 0, "max_shield": 0,
        "effective_health": 45 if alive else 0, "unit_type_id": type_id,
        "position": list(position), "radius": 0.375, "is_flying": False,
        "weapon_ready": weapon_ready, "weapon_cooldown_loops": 0 if weapon_ready else 5}}


def enemy(index, health, shield=0.0, is_flying=False, type_id=MARINE, position=(14.0, 16.0)):
    return {"e%d" % index: {
        "tag": str(200 + index), "type": "Marine", "alive": health > 0,
        "health": health, "max_health": 45, "shield": shield, "max_shield": 0,
        "effective_health": health + shield, "unit_type_id": type_id,
        "position": list(position), "radius": 0.375, "is_flying": is_flying}}


def make_state(n_allies, enemies, attackable=None, in_range=None, ready=None,
               positions=None, profiles=None, heal_candidates=None, moves=True):
    """Build a payload-style snapshot. attackable/in_range/ready default to all-True."""
    allies = {}
    for i in range(n_allies):
        allies.update(ally(i, position=(positions[i] if positions else (10.0, 16.0)),
                           weapon_ready=True if ready is None else ready[i]))
    relations = {}
    for actor in allies:
        row = {}
        for enemy_id in enemies:
            allowed = True if attackable is None else attackable.get((actor, enemy_id), True)
            reach = True if in_range is None else in_range.get((actor, enemy_id), True)
            row[enemy_id] = {"distance": 4.0, "in_weapon_range": reach,
                             "smac_attack_available": allowed}
        relations[actor] = row
    movement = {}
    if moves:
        for actor in allies:
            movement[actor] = {"2": {"direction": "north", "destination": [10, 18],
                                     "distance_to_search_location": 9.0,
                                     "distances_to_visible_enemies": {e: 5.0 for e in enemies}}}
    return {"allies": allies, "visible_enemies": enemies, "relations": relations,
            "unit_type_profiles": profiles if profiles is not None else dict(PROFILE),
            "movement_candidates": movement,
            "heal_candidates": heal_candidates or {}}


class BudgetAllocationTests(unittest.TestCase):
    def test_reference_example_two_targets(self):
        state = make_state(5, {**enemy(0, 10), **enemy(1, 15)})
        result = allocate_focus_fire(state, 5)
        self.assertEqual(result["assignments"],
                         {"u0": "e0", "u1": "e0", "u2": "e1", "u3": "e1", "u4": "e1"})
        self.assertEqual(result["actions"], [6, 6, 7, 7, 7])
        self.assertEqual(result["predicted_damage"], {"e0": 12, "e1": 18})
        self.assertEqual(result["estimated_overkill"], {"e0": 2, "e1": 3})

    def test_exact_damage_budget_stops_assigning(self):
        state = make_state(3, {**enemy(0, 12), **enemy(1, 30)})
        result = allocate_focus_fire(state, 3)
        self.assertEqual([result["assignments"][a] for a in ("u0", "u1")], ["e0", "e0"])
        self.assertEqual(result["assignments"]["u2"], "e1")
        self.assertEqual(result["estimated_overkill"]["e0"], 0)

    def test_second_attacker_overflows_slightly(self):
        state = make_state(2, {**enemy(0, 7), **enemy(1, 40)})
        result = allocate_focus_fire(state, 2)
        self.assertEqual(result["assignments"], {"u0": "e0", "u1": "e0"})
        self.assertEqual(result["estimated_overkill"]["e0"], 5)

    def test_insufficient_budget_does_not_early_transfer_all(self):
        state = make_state(5, {**enemy(0, 20), **enemy(1, 6)})
        result = allocate_focus_fire(state, 5)
        # e0 needs ceil(20/6)=4 attackers; only the 5th spills to e1 (overkill path).
        self.assertEqual([v for v in result["assignments"].values() if v == "e0"].__len__(), 4)
        self.assertEqual([v for v in result["assignments"].values() if v == "e1"], ["e1"])

    def test_priority_target_is_attacked_first(self):
        state = make_state(4, {**enemy(0, 30), **enemy(1, 6), **enemy(2, 6)})
        result = allocate_focus_fire(state, 4, priority_target="e0")
        self.assertEqual(result["target_priority"][0], "e0")
        self.assertEqual(set(result["assignments"].values()), {"e0"})  # 30 HP needs all four.
        result_no_priority = allocate_focus_fire(make_state(4, {**enemy(0, 30), **enemy(1, 6), **enemy(2, 6)}), 4)
        self.assertEqual(result_no_priority["target_priority"][0], "e1")
        self.assertEqual(set(result_no_priority["assignments"].values()), {"e0", "e1", "e2"})

    def test_pending_damage_prevents_replanning_committed_volley(self):
        state = make_state(2, {**enemy(0, 12), **enemy(1, 30)})
        result = allocate_focus_fire(state, 2, pending_damage={"e0": 6})
        self.assertEqual(result["assignments"]["u0"], "e0")
        self.assertEqual(result["assignments"]["u1"], "e1")

    def test_previous_assignment_breaks_attacker_ties(self):
        state = make_state(2, {**enemy(0, 6), **enemy(1, 6)})
        result = allocate_focus_fire(state, 2, previous_assignments={"u1": "e0"})
        self.assertEqual(result["assignments"]["u1"], "e0")
        self.assertEqual(result["assignments"]["u0"], "e1")


class VaryingFormationTests(unittest.TestCase):
    def check_formation(self, n_allies, n_enemies, enemy_health=30):
        enemies = {}
        for j in range(n_enemies):
            enemies.update(enemy(j, enemy_health, position=(14.0 + j, 16.0)))
        state = make_state(n_allies, enemies)
        result = allocate_focus_fire(state, n_allies)
        self.assertEqual(len(result["actions"]), n_allies)
        for actor, enemy_id in result["assignments"].items():
            self.assertEqual(result["actions"][int(actor[1:])], 6 + int(enemy_id[1:]))
        total_assignments = sum(1 for a in result["actions"] if a >= 6)
        self.assertEqual(total_assignments, min(n_allies, n_enemies * 99))
        self.assertLessEqual(len(result["participating_units"]), n_allies)
        return result

    def test_three_allies_three_enemies(self):
        self.check_formation(3, 3)

    def test_eight_allies_nine_enemies(self):
        result = self.check_formation(8, 9, enemy_health=12)
        # 8 attackers, each target needs 2 -> 4 targets fully budgeted, rest overflow.
        self.assertTrue(all(result["assignments"]))

    def test_twentyfive_allies(self):
        self.check_formation(25, 7, enemy_health=45)

    def test_twentyfive_vs_twentyfive(self):
        result = self.check_formation(25, 25, enemy_health=45)
        self.assertEqual(len(result["participating_units"]), 25)

    def test_unequal_attacker_damage(self):
        profiles = dict(PROFILE)
        profiles["51"] = {"known": True, "armor": 0, "weapons": [
            {"target_type": "ground", "range": 5, "base_damage": 10, "hits_per_attack": 1}]}
        state = make_state(3, {**enemy(0, 12), **enemy(1, 16)}, profiles=profiles)
        state["allies"]["u2"]["unit_type_id"] = 51  # Marauder-style heavy hitter.
        result = allocate_focus_fire(state, 3)
        # e0 (12 HP, weaker target first) is covered by u2 (10) + u0 (6); u1 then hits e1.
        self.assertEqual(result["assignments"]["u2"], "e0")
        self.assertEqual(result["assignments"]["u0"], "e0")
        self.assertEqual(result["predicted_damage"]["e0"], 16)
        self.assertEqual(result["assignments"]["u1"], "e1")
        self.assertEqual(result["estimated_overkill"]["e0"], 4)

    def test_budget_deficit_recorded_when_insufficient(self):
        state = make_state(2, {**enemy(0, 30), **enemy(1, 40)})
        result = allocate_focus_fire(state, 2)
        self.assertEqual(result["budget_deficit"], {"e0": 18, "e1": 40})

    def test_each_living_ally_gets_exactly_one_legal_action(self):
        enemies = {**enemy(0, 12), **enemy(1, 20)}
        state = make_state(4, enemies,
                           attackable={("u0", "e0"): False, ("u0", "e1"): False,
                                       ("u1", "e1"): False})
        result = allocate_focus_fire(state, 4)
        assigned = set(result["assignments"])
        self.assertEqual(len(assigned) + len(result["fallback_actions"]),
                         sum(1 for u in state["allies"].values() if u["alive"]))
        legal = {0, 2, 6, 7}  # moves always legal in this snapshot; attacks per relations.
        for index, action in enumerate(result["actions"]):
            actor = "u%d" % index
            relation = state["relations"][actor]
            allowed = {0} | {2} | {6 + int(e[1:]) for e, r in relation.items()
                              if r["smac_attack_available"]}
            self.assertIn(action, allowed)


class LegalityAndRealityTests(unittest.TestCase):
    def test_dead_ally_gets_zero_and_dead_enemy_is_absent(self):
        state = make_state(3, enemy(0, 10))
        state["allies"]["u2"]["alive"] = False
        state["allies"]["u2"]["health"] = 0
        result = allocate_focus_fire(state, 3)
        self.assertEqual(result["actions"][2], 0)
        self.assertNotIn("u2", result["assignments"])

    def test_cooldown_attacker_is_assigned_but_not_budgeted(self):
        state = make_state(3, {**enemy(0, 12), **enemy(1, 40)}, ready=[True, False, False])
        result = allocate_focus_fire(state, 3)
        # Only u0 contributes 6; the cooldown attackers fill the budget without counting.
        self.assertEqual(result["assignments"]["u0"], "e0")
        self.assertEqual(result["predicted_damage"]["e0"], 6)
        self.assertEqual([a for a, t in result["assignments"].items() if t == "e0"], ["u0", "u1", "u2"])

    def test_out_of_range_attacker_not_budgeted(self):
        state = make_state(2, {**enemy(0, 12), **enemy(1, 40)},
                           in_range={("u0", "e0"): True, ("u1", "e0"): False})
        result = allocate_focus_fire(state, 2)
        self.assertEqual(result["predicted_damage"]["e0"], 6)
        self.assertEqual(result["assignments"]["u1"], "e0")  # legal SMAC action, just not counted.

    def test_invisible_enemy_is_never_a_candidate(self):
        state = make_state(2, enemy(1, 10))
        result = allocate_focus_fire(state, 2)
        self.assertNotIn("e0", result["assignments"].values())
        self.assertEqual(result["target_priority"], ["e1"])

    def test_armor_and_shield_change_budget(self):
        armored = {str(MARINE): {"known": True, "armor": 2, "weapons": [
            {"target_type": "ground", "range": 5, "base_damage": 6, "hits_per_attack": 1}]}}
        state = make_state(2, {**enemy(0, 6, shield=2)}, profiles=armored)
        result = allocate_focus_fire(state, 2)
        self.assertEqual(result["predicted_damage"]["e0"], 8)  # (6-2) per hit -> needs 2.
        self.assertEqual(result["assignments"]["u1"], "e0")

    def test_unknown_damage_attacker_assigned_without_fictional_numbers(self):
        unknown = {str(MARINE): {"known": True, "armor": 0, "weapons": [
            {"target_type": "any", "range": 6, "base_damage": None, "hits_per_attack": None}]}}
        state = make_state(2, {**enemy(0, 6), **enemy(1, 40)}, profiles=unknown)
        result = allocate_focus_fire(state, 2)
        self.assertEqual(result["predicted_damage"], {"e0": 0, "e1": 0})
        # Unknown damage cannot be budgeted, so no attacker is ever "enough":
        # both stay on the priority target without fabricated numbers.
        self.assertEqual(result["unknown_damage_attackers"], {"u0": "e0", "u1": "e0"})

    def test_incompatible_weapon_type_gets_fallback(self):
        state = make_state(2, {**enemy(0, 10), **enemy(1, 10)},
                           attackable={("u0", "e0"): False, ("u0", "e1"): False})
        result = allocate_focus_fire(state, 2)
        self.assertNotIn("u0", result["assignments"])
        self.assertEqual(result["fallback_actions"]["u0"], 2)  # deterministic move fallback.
        self.assertEqual(result["actions"][0], 2)

    def test_no_legal_attack_targets_terminates(self):
        state = make_state(2, enemy(0, 10),
                           attackable={("u0", "e0"): False, ("u1", "e0"): False})
        result = allocate_focus_fire(state, 2)
        self.assertTrue(result["terminated"])
        self.assertEqual(result["termination_reason"], "no_legal_attacks")

    def test_no_visible_enemies_terminates(self):
        state = make_state(2, {})
        result = allocate_focus_fire(state, 2)
        self.assertTrue(result["terminated"])
        self.assertEqual(result["termination_reason"], "no_visible_targets")
        self.assertEqual(result["actions"], [2, 2])

    def test_medivac_is_never_an_attacker(self):
        state = make_state(3, enemy(0, 10))
        state["allies"]["u2"]["unit_type_id"] = 54
        state["unit_type_profiles"]["54"] = {"known": True, "armor": 0, "weapons": []}
        state["relations"]["u2"]["e0"]["smac_attack_available"] = False
        state["heal_candidates"]["u2"] = {"7": {"target": "u1", "missing_health": 5, "distance": 3.0}}
        result = allocate_focus_fire(state, 3)
        self.assertNotIn("u2", result["assignments"])
        self.assertEqual(result["actions"][2], 7)  # heals instead of attacking.
        self.assertTrue(all(result["actions"][i] != 6 for i in (0, 1, 2) if i == 2))

    def test_fallback_prefers_attack_then_heal_then_move(self):
        state = make_state(1, enemy(0, 10))
        self.assertEqual(fallback_action(state, "u0"), 6)
        state["relations"]["u0"]["e0"]["smac_attack_available"] = False
        state["heal_candidates"]["u0"] = {}
        self.assertEqual(fallback_action(state, "u0"), 2)
        state["movement_candidates"]["u0"] = {}
        self.assertIsNone(fallback_action(state, "u0"))

    def test_target_priority_is_deterministic(self):
        enemies = {**enemy(0, 10), **enemy(1, 10)}
        state = make_state(2, enemies)
        self.assertEqual(target_priority(state), target_priority(state))
        self.assertEqual(target_priority(state), ["e0", "e1"])


if __name__ == "__main__":
    unittest.main()
