"""Offline unit tests for the HealAssign allocator. No SC2, no API keys."""

import math
import unittest

from jev_micro.skills.heal_assign import (assign_healing, is_healer,
                                          score_heal_need)

from test_jev_cover_retreat import build_state, make_ally, make_enemy


def healer(index, position, energy=50, health=150, max_health=150, alive=True):
    record = make_ally(index, position, health=health, max_health=max_health,
                       type_id=54, alive=alive)
    unit = record[f"u{index}"]
    unit["type"] = "Medivac"
    unit["energy"] = energy
    unit["max_energy"] = 200
    return record


def with_heals(state, heal_candidates):
    state = dict(state)
    state["heal_candidates"] = heal_candidates
    return state


def heal_entry(target, distance, missing):
    return {"target": target, "distance": distance, "missing_health": missing}


def standard_state():
    """2 marines wounded, 1 medivac between them, one enemy threatening u0."""
    allies = {**make_ally(0, (10, 16), health=20, loss=15),
              **make_ally(1, (12, 16), health=30, loss=5),
              **healer(2, (11, 16))}
    enemies = {**make_enemy(0, (13.5, 16))}
    state = build_state(allies, enemies)
    return with_heals(state, {"u2": {"6": heal_entry("u0", 2.0, 25),
                                     "7": heal_entry("u1", 2.0, 15)}})


class LegalityTests(unittest.TestCase):
    def test_single_healer_single_wounded_target(self):
        result = assign_healing(standard_state(), 3)
        self.assertTrue(result["available"])
        self.assertEqual(result["assignments"], {"u2": "u0"})  # u0 far more urgent.
        self.assertEqual(result["healer_actions"], {"u2": 6})

    def test_full_health_and_shield_only_targets_are_absent(self):
        # The environment's heal candidates already exclude full-health and
        # shield-only targets; an empty candidate set must mean no demand.
        allies = {**make_ally(0, (10, 16), health=45, shield=0), **healer(1, (11, 16))}
        state = build_state(allies, {**make_enemy(0, (20, 20))})
        result = assign_healing(with_heals(state, {"u1": {}}), 2)
        self.assertTrue(result["available"])
        self.assertEqual(result["termination_reason"], "no_heal_demand")
        self.assertEqual(result["assignments"], {})

    def test_dead_target_not_in_candidates(self):
        state = standard_state()
        result = assign_healing(state, 3)
        for target in result["assignments"].values():
            self.assertTrue(state["allies"][target]["alive"])

    def test_masked_out_heal_action_is_never_used(self):
        state = standard_state()
        state = with_heals(state, {"u2": {"7": heal_entry("u1", 2.0, 15)}})  # Only u1 legal.
        result = assign_healing(state, 3)
        self.assertEqual(result["assignments"], {"u2": "u1"})
        self.assertEqual(result["healer_actions"], {"u2": 7})

    def test_no_healers_at_all_reports_unavailable(self):
        allies = {**make_ally(0, (10, 16), health=10)}
        state = build_state(allies, {**make_enemy(0, (13, 16))})
        result = assign_healing(state, 1)
        self.assertFalse(result["available"])
        self.assertEqual(result["termination_reason"], "no_healers")
        self.assertTrue(result["terminated"])

    def test_dead_healer_excluded(self):
        allies = {**make_ally(0, (10, 16), health=10),
                  **healer(1, (11, 16), alive=False), **healer(2, (12, 16))}
        state = build_state(allies, {**make_enemy(0, (16, 16))})
        state = with_heals(state, {"u1": {}, "u2": {"6": heal_entry("u0", 2.0, 35)}})
        result = assign_healing(state, 3)
        self.assertEqual(result["healers"], ["u2"])
        self.assertEqual(result["assignments"], {"u2": "u0"})


class PriorityTests(unittest.TestCase):
    def test_urgency_prefers_critical_over_nearby_mild(self):
        # u0 is critical but far; u1 is mildly hurt and next to the healer.
        allies = {**make_ally(0, (2, 16), health=10, loss=10),
                  **make_ally(1, (11.8, 16), health=38, loss=3),
                  **healer(2, (12, 16))}
        state = build_state(allies, {**make_enemy(0, (20, 20))})
        state = with_heals(state, {"u2": {"6": heal_entry("u0", 10.0, 35),
                                          "7": heal_entry("u1", 0.5, 7)}})
        result = assign_healing(state, 3)
        self.assertEqual(result["assignments"], {"u2": "u0"})
        self.assertGreater(result["urgency"]["u0"], result["urgency"]["u1"])

    def test_threatened_target_gets_higher_urgency(self):
        allies = {**make_ally(0, (10, 16), health=25), **make_ally(1, (2, 16), health=25),
                  **healer(2, (6, 16))}
        state = build_state(allies, {**make_enemy(0, (12, 16))})  # Only near u0.
        a = score_heal_need(state, "u0")
        b = score_heal_need(state, "u1")
        self.assertGreater(a, b)

    def test_score_is_deterministic(self):
        state = standard_state()
        self.assertEqual(score_heal_need(state, "u0"), score_heal_need(state, "u0"))

    def test_slightly_more_urgent_new_target_does_not_switch(self):
        allies = {**make_ally(0, (10, 16), health=25), **make_ally(1, (12, 16), health=20),
                  **healer(2, (11, 16))}
        state = build_state(allies, {**make_enemy(0, (20, 20))})
        state = with_heals(state, {"u2": {"6": heal_entry("u0", 2.0, 20),
                                          "7": heal_entry("u1", 2.0, 25)}})
        result = assign_healing(state, 3, previous_assignments={"u2": "u0"})
        self.assertEqual(result["assignments"], {"u2": "u0"})  # Switch margin keeps u0.

    def test_clearly_more_urgent_new_target_switches(self):
        allies = {**make_ally(0, (10, 16), health=44), **make_ally(1, (12, 16), health=8, loss=12),
                  **healer(2, (11, 16))}
        state = build_state(allies, {**make_enemy(0, (13.5, 16))})  # Threatening u1.
        state = with_heals(state, {"u2": {"6": heal_entry("u0", 2.0, 1),
                                          "7": heal_entry("u1", 2.0, 37)}})
        result = assign_healing(state, 3, previous_assignments={"u2": "u0"})
        self.assertEqual(result["assignments"], {"u2": "u1"})
        self.assertEqual(result["released"], {"u2": "u0"})

    def test_target_full_again_releases_reservation(self):
        allies = {**make_ally(0, (10, 16), health=45), **healer(2, (11, 16))}
        state = build_state(allies, {**make_enemy(0, (20, 20))})
        state = with_heals(state, {"u2": {}})  # u0 healed to full: candidate gone.
        result = assign_healing(state, 2, previous_assignments={"u2": "u0"})
        self.assertEqual(result["assignments"], {})
        self.assertEqual(result["termination_reason"], "no_heal_demand")


class MatchingTests(unittest.TestCase):
    def two_healer_state(self, h1_targets):
        allies = {**make_ally(0, (10, 16), health=20, loss=10),
                  **make_ally(1, (12, 16), health=25, loss=8),
                  **healer(2, (10.5, 16)), **healer(3, (11.5, 16))}
        state = build_state(allies, {**make_enemy(0, (20, 20))})
        h0 = {"6": heal_entry("u0", 1.0, 25), "7": heal_entry("u1", 2.0, 20)}
        return with_heals(state, {"u2": h0, "u3": h1_targets})

    def test_classic_assignment_conflict_resolved(self):
        # h0 (u2) can heal u0 and u1; h1 (u3) can only heal u0. The exact
        # matching must give u0 to the limited healer and u1 to the flexible one.
        state = self.two_healer_state({"6": heal_entry("u0", 1.5, 25)})
        result = assign_healing(state, 4)
        self.assertEqual(result["assignments"], {"u3": "u0", "u2": "u1"})

    def test_one_wounded_target_gets_exactly_one_healer(self):
        state = self.two_healer_state({"6": heal_entry("u0", 1.5, 25),
                                       "7": heal_entry("u1", 1.5, 20)})
        allies = state["allies"]
        allies["u1"]["health"] = allies["u1"]["max_health"]  # Only u0 wounded now.
        state["heal_candidates"]["u2"].pop("7")
        state["heal_candidates"]["u3"].pop("7")
        result = assign_healing(state, 4)
        self.assertEqual(list(result["assignments"].values()), ["u0"])
        self.assertEqual(len(result["unassigned_healers"]), 1)

    def test_more_healers_than_targets(self):
        allies = {**make_ally(0, (10, 16), health=20),
                  **healer(1, (11, 16)), **healer(2, (11.5, 16)), **healer(3, (12, 16))}
        state = build_state(allies, {**make_enemy(0, (20, 20))})
        entry = {"6": heal_entry("u0", 1.5, 25)}
        state = with_heals(state, {h: dict(entry) for h in ("u1", "u2", "u3")})
        result = assign_healing(state, 4)
        self.assertEqual(list(result["assignments"].values()), ["u0"])  # Exactly one healer.
        self.assertEqual(len(result["unassigned_healers"]), 2)
        self.assertNotIn(list(result["assignments"])[0], result["unassigned_healers"])
        self.assertTrue(all(r == "no_legal_target" for r in result["unassigned_reasons"].values()))

    def test_25_allies_variable_sizes(self):
        allies = {}
        for i in range(24):
            allies.update(make_ally(i, (10.0 + i * 0.3, 16.0), health=45 - i))
        allies.update(healer(24, (17.0, 16.0)))
        state = build_state(allies, {**make_enemy(0, (25, 25))})
        candidates = {str(6 + i): heal_entry(f"u{i}", 1.0 + i * 0.1, i)
                      for i in range(24)}
        state = with_heals(state, {"u24": candidates})
        result = assign_healing(state, 25)
        self.assertEqual(list(result["assignments"]), ["u24"])
        self.assertEqual(result["healer_actions"]["u24"], 6 + 23)  # u23: lowest health.
        self.assertEqual(len(result["urgency"]), 24)


class SafetyTests(unittest.TestCase):
    def test_endangered_healer_self_preserves(self):
        allies = {**make_ally(0, (10, 16), health=10),
                  **healer(2, (11, 16), health=30)}  # Low HP healer next to the enemy.
        state = build_state(allies, {**make_enemy(0, (13, 16))})
        state = with_heals(state, {"u2": {"6": heal_entry("u0", 2.0, 35)}})
        result = assign_healing(state, 3)
        self.assertIn("u2", result["self_preserving"])
        self.assertNotIn("u2", result["assignments"])
        self.assertIn(result["healer_actions"]["u2"], (2, 3, 4, 5))  # Legal move, not heal.

    def test_endangered_healer_releases_reservation(self):
        allies = {**make_ally(0, (10, 16), health=10),
                  **healer(2, (11, 16), health=30)}
        state = build_state(allies, {**make_enemy(0, (13, 16))})
        state = with_heals(state, {"u2": {"6": heal_entry("u0", 2.0, 35)}})
        result = assign_healing(state, 3, previous_assignments={"u2": "u0"})
        self.assertEqual(result["released"], {"u2": "u0"})
        # One unit, one action: the healer moves, it does not also heal.
        self.assertNotEqual(result["healer_actions"]["u2"], 6)

    def test_safe_healer_keeps_healing_far_from_threats(self):
        allies = {**make_ally(0, (10, 16), health=10),
                  **healer(2, (11, 16), health=30)}
        state = build_state(allies, {**make_enemy(0, (13, 16))})
        state["relations"]["u2"]["e0"]["distance"] = 20.0  # Healer far from the threat.
        state["relations"]["u2"]["e0"]["in_enemy_weapon_range"] = False
        state = with_heals(state, {"u2": {"6": heal_entry("u0", 2.0, 35)}})
        result = assign_healing(state, 3)
        self.assertEqual(result["assignments"], {"u2": "u0"})

    def test_existing_heal_order_is_detected_and_kept(self):
        allies = {**make_ally(0, (10, 16), health=20), **healer(2, (11, 16))}
        state = build_state(allies, {**make_enemy(0, (20, 20))})
        state["allies"]["u2"]["orders"] = [{"ability_id": 386, "kind": "heal",
                                            "target_unit": "u0", "target_tag": "100",
                                            "target_position": [10, 16]}]
        state = with_heals(state, {"u2": {"6": heal_entry("u0", 2.0, 25)}})
        result = assign_healing(state, 3, previous_assignments={"u2": "u0"})
        self.assertEqual(result["assignments"], {"u2": "u0"})
        self.assertTrue(result["retained_orders"].get("u2"))
        # The same action ID is re-issued; the existing executor logic turns it
        # into existing_heal_order_retained instead of a duplicate command.
        self.assertEqual(result["healer_actions"]["u2"], 6)

    def test_non_healers_are_never_assigned(self):
        state = standard_state()
        result = assign_healing(state, 3)
        for actor in result["healer_actions"]:
            self.assertTrue(is_healer(state["allies"][actor]))
        self.assertNotIn("u0", result["healer_actions"])
        self.assertNotIn("u1", result["healer_actions"])

    def test_action_ids_are_within_legal_heal_candidates(self):
        state = standard_state()
        result = assign_healing(state, 3)
        legal = {int(a) for a in state["heal_candidates"]["u2"]}
        self.assertIn(result["healer_actions"]["u2"], legal)


class MatchingScaleTests(unittest.TestCase):
    """P0 regression battery for the SciPy-based joint matching."""

    def build(self, n_healers, n_targets, legal=None):
        allies = {}
        for i in range(n_targets):
            allies.update(make_ally(i, (10.0 + i, 16.0), health=20 + (i % 20)))
        for h in range(n_healers):
            allies.update(healer(100 + h, (9.0, 16.0 + h * 0.1)))
        state = build_state(allies, {})
        candidates = {}
        for h in range(n_healers):
            options = {}
            for i in range(n_targets):
                if legal is not None and (h, i) not in legal:
                    continue
                options[str(6 + i)] = heal_entry(f"u{i}", 1.0 + (h % 3) * 0.1 + i * 0.05, 25 - (i % 20))
            candidates[f"u{100 + h}"] = options
        return with_heals(state, candidates)

    def check(self, n_healers, n_targets, legal=None):
        state = self.build(n_healers, n_targets, legal)
        result = assign_healing(state, n_healers + n_targets)
        targets_assigned = list(result["assignments"].values())
        self.assertEqual(len(targets_assigned), len(set(targets_assigned)))  # One per target.
        for healer, target in result["assignments"].items():
            entry = state["heal_candidates"][healer].get(str(6 + int(target[1:])))
            self.assertIsNotNone(entry)  # Only legal edges become assignments.
        self.assertTrue(all(len(h) > 0 for h in [result["assignments"]]))
        return result

    def test_9_healers_12_targets(self):
        self.check(9, 12)

    def test_12_healers_5_targets(self):
        result = self.check(12, 5)
        self.assertEqual(len(result["assignments"]), 5)
        self.assertEqual(len(result["unassigned_healers"]), 7)

    def test_20_healers_20_targets(self):
        import time
        start = time.monotonic()
        result = self.check(20, 20)
        self.assertLess(time.monotonic() - start, 1.0)  # No exponential blow-up.
        self.assertEqual(len(result["assignments"]), 20)

    def test_all_edges_illegal(self):
        state = self.build(3, 4, legal=set())  # No legal pairs at all.
        result = assign_healing(state, 7)
        self.assertEqual(result["assignments"], {})
        self.assertEqual(len(result["unassigned_healers"]), 3)

    def test_partial_energy_shortage(self):
        # Healers with no candidates (e.g. out of energy) stay idle legally.
        state = self.build(4, 3, legal={(0, 0), (0, 1), (1, 0), (2, 2)})
        result = assign_healing(state, 7)
        for healer in result["assignments"]:
            self.assertNotEqual(healer, "u103")  # u103 has no legal target.
        self.assertIn("u103", result["unassigned_healers"])

    def test_identical_scores_are_deterministic(self):
        state1 = self.build(10, 10)
        state2 = self.build(10, 10)
        self.assertEqual(assign_healing(state1, 20)["assignments"],
                         assign_healing(state2, 20)["assignments"])

    def test_matches_exhaustive_optimum_on_small_inputs(self):
        from jev_micro.skills.heal_assign import _best_matching, _exact_matching
        healers = ["h%d" % i for i in range(6)]
        edges = {h: [(1.0 + ((int(h[1]) + t) % 3) * 0.5, "u%d" % t) for t in range(6)]
                 for h in healers}
        self.assertEqual(_best_matching(healers, edges), _exact_matching(healers, edges))

    def test_healer_actions_are_integers(self):
        from jev_micro.skills.heal_assign import healer_idle_action
        state = self.build(3, 2)
        pool = {entry["target"]: {} for entry in state["heal_candidates"]["u100"].values()}
        action = healer_idle_action(state, "u100", pool, {"u0": 0.5, "u1": 0.4})
        self.assertIsInstance(action, int)  # P0 regression: was a string key.

    def test_endangered_healer_safe_move_never_approaches_danger(self):
        from jev_micro.skills.heal_assign import healer_safe_move
        # Healer east of a wounded ally, enemy just beyond the ally: moving
        # east toward the wounded target would close on the enemy.
        allies = {**make_ally(0, (12, 16), health=10),
                  **healer(2, (10, 16), health=30)}
        state = build_state(allies, {**make_enemy(0, (14, 16))})
        action = healer_safe_move(state, "u2")
        self.assertIsInstance(action, int)
        destination = state["movement_candidates"]["u2"][str(action)]["destination"]
        distance_after = math.dist(destination, (14.0, 16.0))
        self.assertGreaterEqual(distance_after, 4.0 - 1e-9)  # Never closer than now.
        # The safest move here is west (action 5), away from both.
        self.assertEqual(action, 5)


    def test_out_of_range_wounded_ally_is_approached_not_ignored(self):
        # P0 field regression: legal heal edges were empty because the
        # medivac never entered heal range, so it idled forever. It must now
        # approach the wounded ally along a legal move (never counted as
        # provided healing) instead of terminating with no_heal_demand.
        allies = {**make_ally(0, (2.0, 16.0), health=20), **healer(1, (10.0, 16.0))}
        state = build_state(allies, {})  # No enemies: nothing endangers anyone.
        state = with_heals(state, {"u1": {}})  # Out of range: no legal edges.
        result = assign_healing(state, 2)
        self.assertFalse(result["terminated"])
        self.assertEqual(result["approach_targets"], ["u0"])
        self.assertEqual(result["healer_actions"]["u1"], 5)  # West, toward u0.
        self.assertEqual(result["assignments"], {})  # Approach != assignment.


if __name__ == "__main__":
    unittest.main()
