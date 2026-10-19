"""Offline unit tests for the CoverRetreat allocator. No SC2, no API keys."""

import math
import unittest
from unittest.mock import patch

from jev_micro.skills import cover_retreat
from jev_micro.skills.cover_retreat import (allocate_cover_retreat,
                                            assign_cover_roles,
                                            detect_retreat_candidates,
                                            select_retreat_action,
                                            threat_priority)

MARINE = 48
PROFILE = {str(MARINE): {"known": True, "armor": 0, "weapons": [
    {"target_type": "ground", "range": 5, "base_damage": 6, "hits_per_attack": 1}]}}


def make_ally(index, position, health=45, max_health=45, shield=0, max_shield=0,
              loss=0, type_id=MARINE, alive=True):
    return {"u%d" % index: {
        "tag": str(100 + index), "type": "Marine", "alive": alive,
        "health": health, "max_health": max_health, "shield": shield,
        "max_shield": max_shield, "effective_health": health + shield,
        "unit_type_id": type_id, "position": list(position), "radius": 0.375,
        "is_flying": False, "weapon_ready": True, "weapon_cooldown_loops": 0,
        "health_lost_since_previous_decision": loss,
        "shield_lost_since_previous_decision": 0}}


def make_enemy(index, position, health=45, shield=0, is_flying=False, type_id=MARINE):
    return {"e%d" % index: {
        "tag": str(200 + index), "type": "Marine", "alive": health > 0,
        "health": health, "max_health": 45, "shield": shield, "max_shield": 0,
        "effective_health": health + shield, "unit_type_id": type_id,
        "position": list(position), "radius": 0.375, "is_flying": is_flying}}


DIRECTIONS = {2: (0, 1), 3: (0, -1), 4: (1, 0), 5: (-1, 0)}  # north/south/east/west.


def build_state(allies, enemies, attackable=None, enemy_in_range=None,
                move_amount=2.0, profiles=None, moves=None):
    relations = {}
    for actor, unit in allies.items():
        row = {}
        for enemy_id, enemy in enemies.items():
            d = math.dist(unit["position"], enemy["position"])
            row[enemy_id] = {
                "distance": round(d, 2),
                "in_weapon_range": True,
                "weapon_range_status": "in_range",
                "in_enemy_weapon_range": (enemy_in_range or {}).get((actor, enemy_id), d <= 5),
                "smac_attack_available": (attackable or {}).get((actor, enemy_id), True)}
        relations[actor] = row
    movement = {}
    for actor, unit in allies.items():
        movement[actor] = {}
        for action_id, (dx, dy) in DIRECTIONS.items():
            if moves is not None and actor not in moves:
                continue
            destination = [round(unit["position"][0] + dx * move_amount, 3),
                           round(unit["position"][1] + dy * move_amount, 3)]
            movement[actor][str(action_id)] = {
                "direction": {2: "north", 3: "south", 4: "east", 5: "west"}[action_id],
                "destination": destination,
                "distance_to_search_location": 9.0,
                "distances_to_visible_enemies": {
                    enemy_id: round(math.dist(destination, enemy["position"]), 2)
                    for enemy_id, enemy in enemies.items()}}
    return {"allies": allies, "visible_enemies": enemies, "relations": relations,
            "unit_type_profiles": profiles if profiles is not None else dict(PROFILE),
            "movement_candidates": movement, "heal_candidates": {}}


def endangered_state(n_allies=3, n_enemies=3):
    """Basic scene: u0 at (10,16) badly hurt, enemy e0 3 units east, friends west."""
    allies = {}
    for i in range(n_allies):
        allies.update(make_ally(i, (8.0 - i, 16.0)))
    allies.update(make_ally(0, (10.0, 16.0), health=10))  # u0 replaces default position.
    enemies = {}
    for j in range(n_enemies):
        enemies.update(make_enemy(j, (13.0, 16.0 + j)))  # e0 is the closest to u0.
    return build_state(allies, enemies)


class DetectionTests(unittest.TestCase):
    def test_low_health_with_threat_is_candidate(self):
        state = endangered_state()
        candidates = detect_retreat_candidates(state)
        self.assertEqual(list(candidates), ["u0"])
        self.assertGreater(candidates["u0"]["risk"], 0)
        self.assertTrue(candidates["u0"]["threats"])

    def test_low_health_without_threat_is_not_candidate(self):
        allies = {**make_ally(0, (10, 16), health=5), **make_ally(1, (9, 16))}
        enemies = {**make_enemy(0, (28, 28))}  # Visible but far outside threat range.
        candidates = detect_retreat_candidates(build_state(allies, enemies))
        self.assertEqual(candidates, {})

    def test_high_health_recent_loss_triggers(self):
        allies = {**make_ally(0, (10, 16), health=40, loss=12), **make_ally(1, (9, 16))}
        enemies = {**make_enemy(0, (13, 16))}
        candidates = detect_retreat_candidates(build_state(allies, enemies))
        self.assertIn("u0", candidates)

    def test_continuing_hysteresis_tightens_threshold_per_unit(self):
        allies = {**make_ally(0, (10, 16), health=14), **make_ally(1, (11, 16), health=8),
                  **make_ally(2, (9, 16))}
        enemies = {**make_enemy(0, (13, 16))}
        state = build_state(allies, enemies)
        # First decision (no history): u0 (ratio 0.31) enters at the 0.35 threshold.
        self.assertIn("u0", detect_retreat_candidates(state))
        # u0 already retreating must fall below 0.28 to continue; u1 (0.18) enters.
        continuing = detect_retreat_candidates(state, retreating={"u0"})
        self.assertNotIn("u0", continuing)
        self.assertIn("u1", continuing)

    def test_entry_threshold_used_on_first_step_of_skill(self):
        # P0 regression: the manager previously passed a squad-wide tightened
        # threshold on the very first step, missing a 0.30-ratio unit.
        allies = {**make_ally(0, (10, 16), health=13.5), **make_ally(1, (9, 16))}
        state = build_state(allies, {**make_enemy(0, (13, 16))})
        result = allocate_cover_retreat(state, 2)  # No retreating_units history.
        self.assertEqual(result["retreat_units"], ["u0"])

    def test_invisible_enemies_are_never_facts(self):
        allies = {**make_ally(0, (10, 16), health=5)}
        state = build_state(allies, {})  # Hidden enemy simply absent from the snapshot.
        self.assertEqual(detect_retreat_candidates(state), {})
        self.assertTrue(state["risks"] if "risks" in state else True)

    def test_risk_report_separates_facts_from_estimate(self):
        state = endangered_state()
        record = detect_retreat_candidates(state)["u0"]
        self.assertIn("facts", record)
        self.assertIn("survival_ratio", record["facts"])
        self.assertIsInstance(record["risk"], float)


class RetreatDirectionTests(unittest.TestCase):
    def test_basic_retreat_moves_away_from_threat(self):
        allies = {**make_ally(0, (10, 16), health=10)}
        enemies = {**make_enemy(0, (13, 16))}
        action, reason, _ = select_retreat_action(build_state(allies, enemies), "u0", set())
        self.assertEqual(action, 5)  # West increases the distance to e0.
        self.assertEqual(reason, "safe_direction")

    def test_direction_away_from_one_enemy_toward_another_is_penalized(self):
        allies = {**make_ally(0, (10, 16), health=10)}
        enemies = {**make_enemy(0, (13, 16)), **make_enemy(1, (4, 16))}
        action, reason, _ = select_retreat_action(build_state(allies, enemies), "u0", set())
        self.assertEqual(action, 2)  # North escapes both instead of closing on e1.

    def test_blocked_directions_are_not_chosen(self):
        allies = {**make_ally(0, (10, 16), health=10)}
        enemies = {**make_enemy(0, (13, 16))}
        state = build_state(allies, enemies)
        state["movement_candidates"]["u0"] = {"3": state["movement_candidates"]["u0"]["3"]}
        action, _, _ = select_retreat_action(state, "u0", set())
        self.assertEqual(action, 3)

    def test_no_legal_moves_reports_reason(self):
        allies = {**make_ally(0, (10, 16), health=10)}
        enemies = {**make_enemy(0, (13, 16))}
        state = build_state(allies, enemies)
        state["movement_candidates"]["u0"] = {}
        action, reason, _ = select_retreat_action(state, "u0", set())
        self.assertIsNone(action)
        self.assertEqual(reason, "no_legal_moves")

    def test_no_safe_improvement_reports_reason(self):
        allies = {**make_ally(0, (10, 16), health=10)}
        enemies = {**make_enemy(0, (12.2, 16)), **make_enemy(1, (7.8, 16)),
                   **make_enemy(2, (10, 18.2)), **make_enemy(3, (10, 13.8))}
        action, reason, _ = select_retreat_action(build_state(allies, enemies), "u0", set())
        self.assertIsNone(action)
        self.assertEqual(reason, "no_safe_improvement")


class RoleTests(unittest.TestCase):
    def test_roles_are_mutually_exclusive_and_variable(self):
        state = endangered_state()
        cover, retreat, other = assign_cover_roles(state, detect_retreat_candidates(state))
        self.assertEqual(retreat, ["u0"])
        self.assertEqual(cover, ["u1", "u2"])
        self.assertEqual(other, [])
        self.assertFalse(set(retreat) & set(cover))

    def test_all_endangered_means_collective_retreat(self):
        allies = {**make_ally(0, (10, 16), health=5), **make_ally(1, (11, 16), health=5),
                  **make_ally(2, (12, 16), health=5)}
        enemies = {**make_enemy(0, (13, 16))}
        state = build_state(allies, enemies)
        cover, retreat, other = assign_cover_roles(state, detect_retreat_candidates(state))
        self.assertEqual(cover, [])  # Nobody is sacrificed as a forced cover unit.
        self.assertEqual(len(retreat), 3)

    def test_medivac_is_other_not_cover_and_can_flee(self):
        allies = {**make_ally(0, (10, 16), health=6, type_id=54),
                  **make_ally(1, (9, 16)), **make_ally(2, (8, 16))}
        enemies = {**make_enemy(0, (13, 16))}
        state = build_state(allies, enemies)
        state["relations"]["u0"]["e0"]["smac_attack_available"] = False  # Healer: no attack slots.
        result = allocate_cover_retreat(state, 3)
        self.assertNotIn("u0", result["cover_units"])
        self.assertIn("u0", result["retreat_units"])
        self.assertIn(result["actions"][0], (2, 3, 4, 5))  # Legal move, never an attack.

    def test_cover_requires_compatible_legal_attack(self):
        allies = {**make_ally(0, (10, 16), health=10), **make_ally(1, (9, 16))}
        enemies = {**make_enemy(0, (13, 16), is_flying=True)}  # Only target is air.
        state = build_state(allies, enemies,
                            attackable={("u1", "e0"): False})  # Ground-only weapon.
        result = allocate_cover_retreat(state, 2)
        self.assertEqual(result["cover_units"], [])
        self.assertEqual(result["other_units"], ["u1"])


class CoverAllocationTests(unittest.TestCase):
    def test_basic_cover_retreat_composition(self):
        state = endangered_state()
        result = allocate_cover_retreat(state, 3)
        self.assertIn(result["actions"][0], (2, 3, 4, 5))  # u0 retreats with a legal move.
        for actor in ("u1", "u2"):
            self.assertEqual(result["actions"][int(actor[1:])], 6)  # Cover attacks e0.
        self.assertEqual(result["threat_priority"][0], "e0")

    def test_retreating_unit_never_receives_an_attack(self):
        state = endangered_state()
        result = allocate_cover_retreat(state, 3)
        retreaters = result["retreat_units"]
        for actor in retreaters:
            self.assertLess(result["actions"][int(actor[1:])], 6)

    def test_role_switch_removes_unit_from_fire_budget(self):
        state = endangered_state()
        result = allocate_cover_retreat(state, 3,
                                        previous_assignments={"u0": "e0", "u1": "e0"})
        self.assertNotIn("u0", result["cover_assignments"])  # u0 is retreating now.

    def test_focus_fire_receives_only_cover_units_and_priority(self):
        state = endangered_state()
        seen = {}
        real = cover_retreat.allocate_focus_fire

        def spy(cover_state, n_agents, priority_target=None, previous_assignments=None, **kw):
            seen["allies"] = set(cover_state["allies"])
            seen["priority"] = priority_target
            return real(cover_state, n_agents, priority_target=priority_target,
                        previous_assignments=previous_assignments)

        with patch.object(cover_retreat, "allocate_focus_fire", spy):
            allocate_cover_retreat(state, 3)
        self.assertEqual(seen["allies"], {"u1", "u2"})
        self.assertEqual(seen["priority"], "e0")

    def test_fallback_cover_when_focus_fire_yields_nothing(self):
        state = endangered_state()
        empty = {"assignments": {}, "actions": [0] * 3, "predicted_damage": {},
                 "estimated_overkill": {}, "termination_reason": "no_legal_attacks",
                 "terminated": True}
        with patch.object(cover_retreat, "allocate_focus_fire", return_value=empty):
            result = allocate_cover_retreat(state, 3)
        for actor in result["cover_units"]:
            self.assertEqual(result["actions"][int(actor[1:])], 6)  # Nearest legal attack.

    def test_threat_priority_prefers_closest_to_retreater(self):
        allies = {**make_ally(0, (10, 16), health=10), **make_ally(1, (6, 16))}
        enemies = {**make_enemy(0, (13, 16)), **make_enemy(1, (11.5, 16))}
        state = build_state(allies, enemies)
        self.assertEqual(threat_priority(state, ["u0"], ["u1"]), ["e1", "e0"])

    def test_8v9_and_25v25_variable_formations(self):
        for n_allies, n_enemies in ((8, 9), (25, 25)):
            allies = {}
            for i in range(n_allies):
                allies.update(make_ally(i, (10.0 - i * 0.5, 16.0),
                                        health=8 if i == 0 else 45))
            enemies = {}
            for j in range(n_enemies):
                enemies.update(make_enemy(j, (14.0 + (j % 5), 16.0 + (j // 5))))
            result = allocate_cover_retreat(build_state(allies, enemies), n_allies)
            self.assertEqual(len(result["actions"]), n_allies)
            self.assertEqual(result["retreat_units"], ["u0"])
            for index, action in enumerate(result["actions"]):
                if index == 0:
                    self.assertIn(action, (2, 3, 4, 5))
                else:
                    self.assertGreaterEqual(action, 6)

    def test_no_retreat_needed_termination(self):
        allies = {**make_ally(0, (10, 16), health=45)}
        enemies = {**make_enemy(0, (28, 28))}
        result = allocate_cover_retreat(build_state(allies, enemies), 1)
        self.assertTrue(result["safe"])
        self.assertEqual(result["termination_reason"], "no_retreat_needed")

    def test_dead_units_get_zero_and_no_role(self):
        allies = {**make_ally(0, (10, 16), health=10), **make_ally(1, (9, 16), alive=False)}
        enemies = {**make_enemy(0, (13, 16))}
        result = allocate_cover_retreat(build_state(allies, enemies), 2)
        self.assertEqual(result["actions"][1], 0)
        self.assertNotIn("u1", result["cover_units"] + result["retreat_units"])

    def test_actions_are_within_snapshot_legal_set(self):
        state = endangered_state()
        result = allocate_cover_retreat(state, 3)
        for actor, unit in state["allies"].items():
            index = int(actor[1:])
            action = result["actions"][index]
            allowed = {0} | {int(a) for a in state["movement_candidates"][actor]} | {
                6 + int(e[1:]) for e, r in state["relations"][actor].items()
                if r["smac_attack_available"]}
            self.assertIn(action, allowed)


if __name__ == "__main__":
    unittest.main()
