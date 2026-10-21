"""Shared movement scoring: reduce danger while preserving fire opportunity.

Inspired by Sharpy's generic_micro.py / pathing_manager.py idea that moving
away from enemies is not intrinsically better — the best retreat position is
one that is safer AND lets the unit keep (or quickly regain) its next attack.
Re-designed for SMAC's four legal moves; no new coordinates, no pathing, no
hidden enemy data. Scores are heuristic estimates of safety/attack value and
are never claims about damage avoided or dealt.

Two internal movement intents (NOT top-level skills):

- ``tactical`` (TacticalReposition): reposition during weapon cooldown while
  staying at fighting distance from the focus target.
- ``emergency`` (EmergencyRetreat): clear mortality risk; leaving attack
  range is acceptable, re-engagement must stay possible.
"""

import math

from .combat_timing import can_fire_within_window, window_loops
from .focus_fire import _compatible_weapon

PARAMS = {
    "threat_weight": 1.0,          # per world unit of distance to closest visible enemy
    "danger_radius": 5.5,          # enemy closer than this after the move is dangerous
    "danger_penalty": 2.0,         # per dangerous enemy after the move
    "fire_opportunity_weight": 0.8,  # value of keeping/holding the next shot
    "support_weight": 0.15,        # preference for moving toward allied support
    "reentry_cost": 1.5,           # cost of leaving own weapon range of the target
    "direction_change_cost": 0.25, # penalty for flipping direction (anti ping-pong)
    "range_buffer": 0.75,          # stay this far inside own weapon range when tactical
    "melee_range": 1.5,            # own native range below this counts as melee
    "approach_weight": 1.2,        # melee/shorter-range: closing distance is valuable
    "movement_improvement_min": 0.0,  # ties keep the attack; smaller scores are better
}


def predicted_move(state, actor, action_id):
    """Estimate travel in one control window, not the full command distance.

    SC2 catalog movement_speed is in simulation units per 16 game loops.
    Acceleration, collision and moving enemies remain unmodelled.
    """
    nominal = state["movement_candidates"][actor][str(action_id)]
    origin = state["allies"][actor]["position"]
    unit_id = str(state["allies"][actor]["unit_type_id"])
    speed = state.get("unit_type_profiles", {}).get(unit_id, {}).get("movement_speed_native")
    length = math.dist(origin, nominal["destination"])
    fraction = min(1.0, speed * state["step_mul"] / 16.0 / length) \
        if speed is not None and "step_mul" in state and length > 0 else 1.0
    destination = [a + fraction * (b - a) for a, b in zip(origin, nominal["destination"])]
    return {**nominal, "destination": destination,
            "distances_to_visible_enemies": {
                e: math.dist(destination, u["position"]) for e, u in state["visible_enemies"].items()}}


def engagement_reach(state, actor, target_id):
    """Intersection of estimated native reach and the command candidate limit."""
    relation = state.get("relations", {}).get(actor, {}).get(target_id, {})
    native_range = relation.get("native_weapon_range")
    target = state["visible_enemies"].get(target_id)
    if native_range is None or target is None:
        return None
    unit = state["allies"][actor]
    reach = native_range + unit["radius"] + target["radius"]
    mask_range = relation.get("smac_attack_distance_limit")
    return min(reach, mask_range) if mask_range is not None else reach


def threat_count(state, actor, position, parameters=None):
    """Visible enemy weapon envelopes, with a heuristic fallback if unknown."""
    params = {**PARAMS, **(parameters or {})}
    unit = state["allies"][actor]
    count = 0
    for enemy in state["visible_enemies"].values():
        profile = state.get("unit_type_profiles", {}).get(str(enemy["unit_type_id"]), {})
        weapon = _compatible_weapon(profile, unit)
        reach = (weapon["range"] + unit["radius"] + enemy["radius"]
                 if weapon and weapon.get("range") is not None else params["danger_radius"])
        count += math.dist(position, enemy["position"]) <= reach
    return count


def score_movement_candidate(state, actor, action_id, target_id, movement_mode,
                             parameters=None, previous_action=None):
    """Score one legal movement candidate (smaller is better).

    Only legal snapshot moves are scored (the caller passes candidates from
    ``movement_candidates``); the nominal destination is a prediction — SC2
    may not fully realize it, so no hard guarantees are derived from it.
    """
    move = predicted_move(state, actor, action_id)
    return _score_position(state, actor, move, target_id, movement_mode,
                           parameters, action_id, previous_action)


def _score_position(state, actor, move, target_id, movement_mode,
                    parameters=None, action_id=None, previous_action=None):
    """Same objective for moving and staying; staying is not a Stop command."""
    params = {**PARAMS, **(parameters or {})}
    relations = state.get("relations", {}).get(actor, {})
    destination = move["destination"]
    distances = list(move["distances_to_visible_enemies"].values())
    min_enemy = min(distances) if distances else params["danger_radius"] + 1
    danger_count = threat_count(state, actor, destination, parameters)

    # Tactical mode optimizes the fighting position, not raw distance: the
    # threat term is scaled down so a comfortable in-range spot can beat a
    # farther but range-breaking one. Emergency mode keeps full weight.
    effective_threat = params["threat_weight"] * (0.4 if movement_mode == "tactical" else 1.0)
    score = -min_enemy * effective_threat + danger_count * params["danger_penalty"]

    # Fire opportunity: staying inside own native range of the focus target
    # matters most when the weapon can fire within this window. Three zones:
    # comfortably inside (bonus), at the edge without buffer (small cost),
    # out of range (re-entry cost). Melee units in tactical mode weight
    # threat ~0: stepping away from a melee target saves nothing, closing in
    # restores the next attack.
    fires_soon = can_fire_within_window(state, actor, target_id,
                                        parameters=parameters) \
        if target_id in relations else False
    relation = relations.get(target_id)
    native_range = (relation or {}).get("native_weapon_range")
    melee = native_range is not None and native_range <= params["melee_range"]
    if melee and movement_mode == "tactical":
        score += min_enemy * effective_threat  # Cancel the raw distance term.
    if native_range is not None:
        target_distance = ((destination[0] - state["visible_enemies"][target_id]["position"][0]) ** 2
                           + (destination[1] - state["visible_enemies"][target_id]["position"][1]) ** 2) ** 0.5
        reach = engagement_reach(state, actor, target_id)
        if movement_mode == "tactical":
            if target_distance <= reach - params["range_buffer"]:
                score -= params["fire_opportunity_weight"]  # Keeps the next shot.
            elif target_distance <= reach:
                score += 0.5 * params["reentry_cost"]  # At the edge: fragile position.
            else:
                score += params["reentry_cost"]  # Must close back in later.
        else:  # emergency: leaving range is accepted, but re-entry gets no bonus
            if fires_soon and target_distance > reach:
                score += params["reentry_cost"] * 0.5
    elif fires_soon:
        score -= params["fire_opportunity_weight"] * 0.5  # Unknown range: mild bonus.

    # Allied support: closer to living non-retreating allies is better.
    supporters = [u["position"] for a, u in state["allies"].items()
                  if a != actor and u.get("alive") and not u.get("__retreating__")]
    if supporters:
        support = min((destination[0] - p[0]) ** 2 + (destination[1] - p[1]) ** 2
                      for p in supporters) ** 0.5
        score += support * params["support_weight"]

    if action_id is not None and previous_action is not None and str(previous_action) != str(action_id):
        score += params["direction_change_cost"]
    return round(score, 3)


def _fire_opportunity(state, actor, destination, target_id, parameters=None):
    relation = state.get("relations", {}).get(actor, {}).get(target_id, {})
    target = state["visible_enemies"].get(target_id)
    reach = engagement_reach(state, actor, target_id)
    if target is None or reach is None:
        return None
    unit = state["allies"][actor]
    distance = sum((a - b) ** 2 for a, b in zip(destination, target["position"])) ** 0.5
    return (distance <= reach
            and unit.get("weapon_cooldown_loops", 0) <= window_loops(state, parameters))


def select_movement(state, actor, target_id, movement_mode, previous_action=None,
                    parameters=None):
    """Best legal move under the given intent. Returns (action, reason, info).

    Falls back to None (no move beats standing fire) only for tactical mode
    when every candidate is worse than keeping the attack; emergency mode
    always returns the best available move because staying put is the risk.
    """
    moves = state.get("movement_candidates", {}).get(actor, {})
    position = state["allies"][actor]["position"]
    standing = {"destination": position, "distances_to_visible_enemies": {
        enemy_id: sum((a - b) ** 2 for a, b in zip(position, enemy["position"])) ** 0.5
        for enemy_id, enemy in state["visible_enemies"].items()}}
    standing_score = _score_position(state, actor, standing, target_id, movement_mode, parameters)
    info = {"movement_mode": movement_mode, "standing_score": standing_score,
            "movement_candidate_scores": {}, "selected_action": None,
            "predicted_destinations": {}, "rejected_candidates": {},
            "standing_threat_count": threat_count(state, actor, position, parameters),
            "standing_fire_opportunity": _fire_opportunity(state, actor, position, target_id, parameters),
            "meaning": "Heuristic position scores and geometric fire estimates; not observed damage. "
                       "No move means return to the attack allocator, not issue Stop."}
    if not moves:
        return None, "no_legal_moves", info
    scored = []
    reach = engagement_reach(state, actor, target_id)
    target = state["visible_enemies"].get(target_id)
    currently_engaged = reach is not None and math.dist(position, target["position"]) <= reach
    for action_id in moves:
        prediction = predicted_move(state, actor, action_id)
        info["predicted_destinations"][str(action_id)] = prediction["destination"]
        # Do not give up the ability to issue attacks for a tactical move that
        # still leaves the unit in as many enemy weapon envelopes.
        if (movement_mode == "tactical" and currently_engaged
                and math.dist(prediction["destination"], target["position"]) > reach
                and threat_count(state, actor, prediction["destination"], parameters)
                >= info["standing_threat_count"]):
            info["rejected_candidates"][str(action_id)] = "loses_attack_without_reducing_exposure"
            continue
        score = score_movement_candidate(state, actor, action_id, target_id,
                                         movement_mode, parameters, previous_action)
        scored.append((score, int(action_id)))
    scored.sort()
    if not scored:
        return None, "keep_attack_position", info
    best_score, best_action = scored[0]
    info.update(movement_candidate_scores={str(a): s for s, a in scored},
                best_move_action=best_action, best_move_score=best_score,
                improvement=round(standing_score - best_score, 3),
                keeps_fire_opportunity=_fire_opportunity(
                    state, actor, predicted_move(state, actor, best_action)["destination"],
                    target_id, parameters),
                selected_threat_count=threat_count(
                    state, actor, predicted_move(state, actor, best_action)["destination"], parameters))
    margin = {**PARAMS, **(parameters or {})}["movement_improvement_min"]
    if movement_mode == "tactical" and best_score >= standing_score - margin:
        return None, "keep_attack_position", info
    info["selected_action"] = best_action
    return best_action, "scored_move", info
