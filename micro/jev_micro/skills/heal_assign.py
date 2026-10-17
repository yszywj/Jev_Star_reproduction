"""HealAssign: joint, one-to-one healer/target assignment with sustained healing.

Model assumptions and scope (first version):

- Works on one ``build_request`` snapshot (allies / visible_enemies /
  relations / heal_candidates / movement_candidates). Healer legality comes
  exclusively from the snapshot's ``heal_candidates``, which the environment
  already gates on aliveness, energy, target type, flight limits, missing
  health and the SMAC legality mask — this module never bypasses
  ``get_avail_agent_actions()``.
- Heal demand is health-only: ``missing_health = max_health - health``.
  Shield loss never makes a target healable by a Medivac (the environment's
  heal candidates already enforce this; we keep the same semantics here).
- The urgency score is an explainable heuristic over observed facts only
  (health ratio, recent observed health loss, visible threats, distance).
  Future damage, heal rates and arrival times are not predicted; urgency is
  an estimate, never a claim about future health.
- Assignment is a deterministic maximum-weight one-to-one matching between
  healers and legal targets (exact for up to 8 healers, greedy+swap beyond).
  Several healers may heal the same target in the game engine; one-to-one is
  our scheduling policy to avoid pointless duplication, not an engine rule.
- Planned healing is never counted as restored health. Observed health
  changes are net of enemy damage; this module records assignments only.
- An endangered healer self-preserves: it is removed from the matching, its
  reservation is released, and it moves via CoverRetreat's legal retreat
  scoring (reused, not reimplemented). One unit, one final action per step.
- Non-healer units are NOT controlled here; HealAssign returns actions only
  for healer slots and the caller merges a declared base policy for the rest.
"""

from .cover_retreat import PARAMS as COVER_PARAMS, select_retreat_action

import math

PARAMS = {
    "switch_margin": 0.15,      # score bonus for keeping the previous target (hysteresis)
    "threat_distance": COVER_PARAMS["threat_distance"],
    "healer_danger_ratio": 0.35,  # healer survival ratio below which it may self-preserve
    "distance_cost": 0.02,       # per world unit of healer-target distance
    "exact_matching_healers": 8,  # exact search bound; greedy+swap beyond this
}

HEAL_ORDER_ABILITY = 386  # Medivac heal, as used by the existing executor.


def is_healer(unit):
    """Medivac-style unit identified by its public type name in the snapshot."""
    return str(unit.get("type", "")).lower().startswith("medivac")


def health_ratio(unit):
    return unit["health"] / unit["max_health"] if unit["max_health"] else 1.0


def _threatened(state, actor):
    return any(r["distance"] <= PARAMS["threat_distance"]
               or r.get("in_enemy_weapon_range") is True
               for r in state.get("relations", {}).get(actor, {}).values())


def score_heal_need(state, target_id, parameters=None):
    """Explainable urgency of healing one target (healer-independent part).

    Combines missing health share, low health ratio, recent observed health
    loss and visible threats. Deterministic; ties are broken by target ID at
    the caller.
    """
    params = {**PARAMS, **(parameters or {})}
    target = state["allies"][target_id]
    maximum = target["max_health"] or 1
    missing = max(0.0, maximum - target["health"]) / maximum
    ratio = health_ratio(target)
    recent_loss = min(target.get("health_lost_since_previous_decision", 0) / maximum, 0.5)
    threatened = 1.0 if _threatened(state, target_id) else 0.0
    return round(0.6 * (1 - ratio) + 0.2 * missing + 0.25 * recent_loss + 0.15 * threatened, 3)


def healer_endangered(state, actor, parameters=None):
    params = {**PARAMS, **(parameters or {})}
    unit = state["allies"][actor]
    return health_ratio(unit) < params["healer_danger_ratio"] and _threatened(state, actor)


def _edge_score(state, healer, target_id, entry, urgency, previous_assignments, parameters):
    params = {**PARAMS, **(parameters or {})}
    score = urgency + 0.2 * entry["missing_health_factor"] \
        - params["distance_cost"] * entry["distance"]
    if (previous_assignments or {}).get(healer) == target_id:
        score += params["switch_margin"]  # keep a still-valid relationship stable
    return score


def _best_matching(healers, edges):
    """Deterministic maximum-weight one-to-one matching (healers -> targets).

    ``edges[h]`` is a list of (score, target) sorted by (-score, target id).
    Exact branch-and-bound for small healer counts; greedy plus one swap
    round beyond the bound. Healers may stay unassigned.
    """
    if len(healers) <= PARAMS["exact_matching_healers"]:
        best = {"score": 0.0, "assign": {}}

        def recurse(index, used, total, assign):
            if index == len(healers):
                if total > best["score"] or (total == best["score"] and not best["assign"]):
                    best["score"], best["assign"] = total, dict(assign)
                return
            healer = healers[index]
            recurse(index + 1, used, total, assign)  # leaving this healer idle
            for score, target in edges[healer]:
                if target in used or score <= 0:
                    continue
                assign[healer] = target
                recurse(index + 1, used | {target}, total + score, assign)
                del assign[healer]

        recurse(0, frozenset(), 0.0, {})
        return best["assign"]

    assign, used = {}, set()
    for healer in healers:  # greedy pass in deterministic edge order
        for score, target in edges[healer]:
            if score > 0 and target not in used:
                assign[healer], _ = target, used.add(target)
                break
    for healer in healers:  # one swap round: an idle healer steals a target
        if healer in assign:
            continue
        for score, target in edges[healer]:
            if score <= 0:
                continue
            holder = next((h for h, t in assign.items() if t == target), None)
            if holder is None:
                assign[healer] = target
                break
            if edges[healer] and score > dict(edges[holder]).get(target, 0) + 1e-9:
                del assign[holder]
                assign[healer] = target
                break
    return assign


def healer_idle_action(state, healer, target_pool, urgency, parameters=None):
    """Legal idle behavior for a healer with no assignment: approach the most
    urgent wounded target along a legal move, else take a safe positioning
    move. Returns an action ID or None (base policy keeps control).

    Medivacs never get attack fallbacks: their SMAC attack slots are heal
    commands and attacking enemies is not a legal action for them.
    """
    params = {**PARAMS, **(parameters or {})}
    moves = state.get("movement_candidates", {}).get(healer, {})
    if target_pool:
        best_target = max(sorted(target_pool), key=lambda t: (urgency.get(t, 0), -int(t[1:])))
        goal = state["allies"][best_target]["position"]
        rows = [(math.dist(entry["destination"], goal), int(action_id), action_id)
                for action_id, entry in moves.items()]
        if rows:
            return min(rows)[2]  # Approach: never counted as provided healing.
    action, _, _ = select_retreat_action(state, healer, {healer}, params)
    return action


def assign_healing(state, n_agents, previous_assignments=None, parameters=None):
    """One HealAssign decision step on a fresh snapshot.

    Returns a partial action map (healer slots only), full-length filler is
    the caller's responsibility, plus diagnostics. Non-healer allies are
    never assigned here.
    """
    params = {**PARAMS, **(parameters or {})}
    allies = state["allies"]
    healers = sorted((a for a, u in allies.items() if u["alive"] and is_healer(u)),
                     key=lambda a: int(a[1:]))

    if not any(is_healer(u) for u in allies.values()):
        return {"available": False, "termination_reason": "no_healers",
                "terminated": True, "assignments": {}, "healer_actions": {},
                "healers": [], "self_preserving": {}, "urgency": {},
                "unassigned_healers": [], "unassigned_reasons": {},
                "retained_orders": {}}

    endangered = {h: healer_endangered(state, h, params) for h in healers}
    dead_healers = [a for a, u in allies.items() if is_healer(u) and not u["alive"]]

    # Candidate targets and urgency come from each healer's legal heal set.
    target_pool, urgency = {}, {}
    for healer in healers:
        if endangered[healer]:
            continue
        for action_id, entry in state.get("heal_candidates", {}).get(healer, {}).items():
            target_id = entry["target"]
            if target_id not in urgency:
                urgency[target_id] = score_heal_need(state, target_id, params)
            target_pool.setdefault(target_id, {})[healer] = {
                "action_id": int(action_id),
                "distance": entry["distance"],
                "missing_health": entry["missing_health"],
                "missing_health_factor": round(
                    entry["missing_health"] / (allies[target_id]["max_health"] or 1), 3)}

    edges = {}
    for healer in healers:
        if endangered[healer]:
            edges[healer] = []
            continue
        rows = []
        for target_id, info in target_pool.items():
            if healer not in info:
                continue  # Not a legal healer-target pair in this snapshot.
            score = _edge_score(state, healer, target_id, info[healer],
                                urgency[target_id], previous_assignments, params)
            rows.append((round(score, 4), target_id))
        edges[healer] = sorted(rows, key=lambda r: (-r[0], r[1]))

    assignments = _best_matching([h for h in healers if not endangered[h]], edges) \
        if any(not endangered[h] for h in healers) else {}

    healer_actions, retained_orders, released = {}, {}, {}
    for healer in healers:
        if endangered[healer]:
            # Self-preservation: retreat move via CoverRetreat's scorer; the
            # heal reservation is released this step, never both actions.
            action, reason, _ = select_retreat_action(state, healer, {healer}, params)
            healer_actions[healer] = action if action is not None \
                else healer_idle_action(state, healer, target_pool, urgency, params)
            if (previous_assignments or {}).get(healer):
                released[healer] = previous_assignments[healer]
            continue
        target_id = assignments.get(healer)
        if target_id is None:
            # Legal idle behavior: approach a wounded ally or reposition;
            # None leaves the slot to the declared base policy.
            healer_actions[healer] = healer_idle_action(state, healer, target_pool,
                                                        urgency, params)
            continue
        info = target_pool[target_id][healer]
        healer_actions[healer] = info["action_id"]
        retained_orders[healer] = any(
            order.get("ability_id") == HEAL_ORDER_ABILITY
            and order.get("target_unit") == target_id
            for order in allies[healer].get("orders", []))
        if (previous_assignments or {}).get(healer) and previous_assignments[healer] != target_id:
            released[healer] = previous_assignments[healer]

    reason = None
    live_healers = [h for h in healers]
    if not live_healers:
        reason = "all_healers_dead"
    elif not target_pool:
        reason = "no_heal_demand"
    elif not assignments and all(endangered.values()):
        reason = "all_healers_endangered"

    return {
        "available": True,
        "healers": healers,
        "dead_healers": dead_healers,
        "wounded_targets": sorted(target_pool),
        "assignments": assignments,
        "healer_actions": healer_actions,
        "urgency": urgency,
        "missing_health": {t: target_pool[t][next(iter(target_pool[t]))]["missing_health"]
                           for t in target_pool},
        "self_preserving": {h: True for h, flag in endangered.items() if flag},
        "retained_orders": retained_orders,
        "released": released,
        "unassigned_healers": [h for h in healers if h not in assignments and not endangered[h]],
        "unassigned_reasons": {h: "no_legal_target" for h in healers
                               if h not in assignments and not endangered[h]},
        "terminated": reason is not None,
        "termination_reason": reason,
    }
