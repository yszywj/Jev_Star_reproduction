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

from .combat_timing import can_fire_within_window, window_loops

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
}


def score_movement_candidate(state, actor, action_id, target_id, movement_mode,
                             parameters=None, previous_action=None):
    """Score one legal movement candidate (smaller is better).

    Only legal snapshot moves are scored (the caller passes candidates from
    ``movement_candidates``); the nominal destination is a prediction — SC2
    may not fully realize it, so no hard guarantees are derived from it.
    """
    params = {**PARAMS, **(parameters or {})}
    move = state["movement_candidates"][actor][str(action_id)]
    relations = state.get("relations", {}).get(actor, {})
    destination = move["destination"]
    distances = list(move["distances_to_visible_enemies"].values())
    min_enemy = min(distances) if distances else params["danger_radius"] + 1
    danger_count = sum(1 for d in distances if d <= params["danger_radius"])

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
        radii = state["allies"][actor]["radius"] + state["visible_enemies"][target_id]["radius"]
        target_distance = ((destination[0] - state["visible_enemies"][target_id]["position"][0]) ** 2
                           + (destination[1] - state["visible_enemies"][target_id]["position"][1]) ** 2) ** 0.5
        reach = native_range + radii
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

    if previous_action is not None and str(previous_action) != str(action_id):
        score += params["direction_change_cost"]
    return round(score, 3)


def select_movement(state, actor, target_id, movement_mode, previous_action=None,
                    parameters=None):
    """Best legal move under the given intent. Returns (action, reason, info).

    Falls back to None (no move beats standing fire) only for tactical mode
    when every candidate is worse than keeping the attack; emergency mode
    always returns the best available move because staying put is the risk.
    """
    moves = state.get("movement_candidates", {}).get(actor, {})
    if not moves:
        return None, "no_legal_moves", {}
    scored = []
    for action_id in moves:
        score = score_movement_candidate(state, actor, action_id, target_id,
                                         movement_mode, parameters, previous_action)
        scored.append((score, int(action_id)))
    scored.sort()
    best_score, best_action = scored[0]
    info = {"movement_mode": movement_mode,
            "movement_candidate_scores": {str(a): s for s, a in scored},
            "keeps_fire_opportunity": score_movement_candidate(
                state, actor, best_action, target_id, movement_mode, parameters,
                previous_action) <= min(s for s, _ in scored)}
    return best_action, "scored_move", info
