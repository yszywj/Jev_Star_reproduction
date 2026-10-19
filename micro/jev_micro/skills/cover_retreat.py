"""CoverRetreat: identify endangered allies, retreat them along legal moves,
and let remaining combatants cover with fire borrowed from FocusFireBudget.

Model assumptions and scope (first version):

- Works on one ``build_request`` snapshot (allies / visible_enemies /
  relations / movement_candidates / unit_type_profiles). Only team-visible
  allied observation data is used; hidden enemies are unknown, not facts.
- "Threat" is an observation-derived estimate: a visible enemy within
  ``threat_distance`` or reported inside this unit's weapon range via
  ``relations[actor][enemy]["in_enemy_weapon_range"]``. Enemy intent, hidden
  positions and weapon cooldowns are never assumed known.
- Danger classification is explainable and threshold-based (see ``PARAMS``);
  no learned components. Thresholds live in one place and are configurable.
- Retreat planning scores the existing legal ``movement_candidates`` only:
  single-step moves (north/south/east/west), no new coordinate commands and
  no duplicate pathing system. Map-edge/corner reasoning is limited to what
  the candidate distances reveal; blocked moves simply never appear.
- Cover fire is delegated to FocusFireBudget restricted to cover-role units,
  so retreating units never contribute to the damage budget of the same
  step. If FocusFireBudget yields nothing, cover units fall back to the
  shared deterministic legal fallback (attack nearest legal, then move).
- Cover attacks are not assumed to change enemy targeting; they only remove
  threats through legal damage and time.
"""

from .focus_fire import allocate_focus_fire, fallback_action

# Central, configurable thresholds. "观测事实" (threats, distances, health)
# are separated from "风险估计" (the risk score) in every returned record.
PARAMS = {
    "risk_threshold": 0.35,      # effective survival ratio below which a threatened unit retreats
    "recent_loss_ratio": 0.20,   # or lost this share of max effective health in the last window
    "continue_factor": 0.8,      # while already retreating, require ratio < threshold * this (hysteresis)
    "threat_distance": 8.0,      # visible enemy within this center distance counts as a threat
    "danger_radius": 5.5,        # post-move distance under which an enemy is considered dangerous
    "support_distance": 6.0,     # allies within this range provide support
    "improvement_min": 0.1,      # retreat move must improve min threat distance by at least this
}


def _alive(unit):
    return unit.get("alive", unit.get("health", 0) > 0)


def survival_ratio(unit):
    maximum = (unit["max_health"] or 0) + (unit["max_shield"] or 0)
    if maximum <= 0:
        return 1.0
    return (unit["health"] + unit["shield"]) / maximum


def _threats_to(state, actor):
    """Observation facts: visible enemies close enough or in weapon range of actor."""
    row = state.get("relations", {}).get(actor, {})
    return sorted(
        enemy_id for enemy_id, relation in row.items()
        if relation["distance"] <= PARAMS["threat_distance"]
        or relation.get("in_enemy_weapon_range") is True)


def detect_retreat_candidates(state, parameters=None, retreating=frozenset()):
    """Return {actor: {...}} for endangered allies at their own threshold.

    A unit retreats only when a visible threat exists AND its effective
    survival ratio is below threshold OR it just lost a large share of
    health. Low health without threats, or threats without damage pressure,
    does not trigger retreat. Hysteresis is per unit, based on its own
    history: units already retreating (``retreating``) must fall below the
    tightened ``continue_factor`` threshold to keep retreating, so borderline
    units drop out — while a unit entering for the first time always uses the
    full entry threshold.
    """
    params = {**PARAMS, **(parameters or {})}
    candidates = {}
    for actor, unit in state["allies"].items():
        if not _alive(unit):
            continue
        threats = _threats_to(state, actor)
        if not threats:
            continue  # No visible threat: never retreat on low health alone.
        threshold = params["risk_threshold"] * (params["continue_factor"]
                                                if actor in retreating else 1.0)
        ratio = survival_ratio(unit)
        maximum = (unit["max_health"] or 0) + (unit["max_shield"] or 0)
        recent_loss = (unit.get("health_lost_since_previous_decision", 0)
                       + unit.get("shield_lost_since_previous_decision", 0)) / (maximum or 1)
        reasons = []
        if ratio < threshold:
            reasons.append(f"survival_ratio={ratio:.2f}<{threshold:.2f}")
        if recent_loss >= params["recent_loss_ratio"]:
            reasons.append(f"recent_loss={recent_loss:.2f}>={params['recent_loss_ratio']:.2f}")
        if not reasons:
            continue
        in_range = any(state["relations"][actor][e].get("in_enemy_weapon_range") is True for e in threats)
        risk = min(1.0, (1 - ratio) * 0.6 + min(recent_loss, 0.5) * 0.8
                   + min(len(threats), 3) * 0.05 + (0.2 if in_range else 0.0))
        candidates[actor] = {
            "risk": round(risk, 3), "reasons": reasons, "threats": threats,
            "facts": {"survival_ratio": round(ratio, 3), "recent_loss_ratio": round(recent_loss, 3),
                      "visible_threat_count": len(threats),
                      "in_enemy_weapon_range": in_range},
        }
    return candidates


def select_retreat_action(state, actor, retreat_units, parameters=None, previous_action=None):
    """Pick the safest legal move for one retreating unit.

    Scores every legal movement candidate: maximize distance to the closest
    visible enemy, avoid entering other enemies' danger radius, prefer moves
    toward allied support, and keep the previous direction to avoid
    ping-pong. Returns (action_id or None, reason, telemetry).
    """
    params = {**PARAMS, **(parameters or {})}
    moves = state.get("movement_candidates", {}).get(actor, {})
    if not moves:
        return None, "no_legal_moves", {}
    relations = state.get("relations", {}).get(actor, {})
    current_min = min((r["distance"] for r in relations.values()), default=None)

    supporters = [u["position"] for a, u in state["allies"].items()
                  if a != actor and _alive(u) and a not in (retreat_units or set())]

    def support_distance(destination):
        if not supporters:
            return None
        return min((destination[0] - p[0]) ** 2 + (destination[1] - p[1]) ** 2 for p in supporters) ** 0.5

    scored = []
    for action_id, entry in moves.items():
        distances = list(entry["distances_to_visible_enemies"].values())
        min_enemy = min(distances) if distances else params["threat_distance"] + 1
        danger_count = sum(1 for d in distances if d <= params["danger_radius"])
        support = support_distance(entry["destination"])
        continuity_penalty = 0.0 if str(previous_action) == str(action_id) else 0.25
        score = (-min_enemy + danger_count * 2.0 + (support if support is not None else 0.0) * 0.15
                 + continuity_penalty)
        scored.append((score, int(action_id), action_id, min_enemy, danger_count, support))
    scored.sort()
    best = scored[0]
    _, action_id, action, min_enemy, danger_count, support = best
    current_danger = sum(1 for r in relations.values() if r["distance"] <= params["danger_radius"])
    if current_min is not None and min_enemy <= current_min + params["improvement_min"] \
            and danger_count >= current_danger:
        # No direction meaningfully improves safety: report instead of
        # fabricating a pointless or riskier move.
        return None, "no_safe_improvement", {"best_action": int(action), "scored_candidates": len(scored)}
    return int(action), "safe_direction", {
        "min_enemy_distance_after": round(min_enemy, 2),
        "dangerous_enemies_after": danger_count, "support_distance": round(support, 2) if support else None}


def assign_cover_roles(state, retreat_candidates):
    """Split living allies into retreat / cover / other roles (variable size).

    Cover units are alive, not retreating, and have at least one legal
    attack action (healers never qualify — their SMAC attack slots are heal
    commands). If everyone is retreating, the team retreats collectively and
    the cover set is empty rather than sacrificing a unit.
    """
    retreat_units = sorted(retreat_candidates, key=lambda a: int(a[1:]))
    cover_units, other_units = [], []
    for actor, unit in state["allies"].items():
        if not _alive(unit) or actor in retreat_candidates:
            continue
        relations = state.get("relations", {}).get(actor, {})
        if any(r["smac_attack_available"] for r in relations.values()):
            cover_units.append(actor)
        else:
            other_units.append(actor)
    return (sorted(cover_units, key=lambda a: int(a[1:])),
            retreat_units,
            sorted(other_units, key=lambda a: int(a[1:])))


def threat_priority(state, retreat_units, cover_units):
    """Rank visible enemies: threaten retreater + closest to one first."""
    rows = []
    for enemy_id in state["visible_enemies"]:
        threat_distance = min(
            (state["relations"][a][enemy_id]["distance"] for a in retreat_units
             if enemy_id in state.get("relations", {}).get(a, {})), default=None)
        attackable = any(state.get("relations", {}).get(a, {}).get(enemy_id, {})
                         .get("smac_attack_available") for a in cover_units)
        rows.append((
            0 if threat_distance is not None else 1,          # threatens a retreater first
            threat_distance if threat_distance is not None else float("inf"),
            0 if attackable else 1,
            state["visible_enemies"][enemy_id]["effective_health"],
            int(enemy_id[1:]), enemy_id))
    return [row[5] for row in sorted(rows)]


def allocate_cover_retreat(state, n_agents, parameters=None, previous_assignments=None,
                           retreating_units=None):
    """One CoverRetreat decision step on a fresh snapshot.

    ``retreating_units`` carries the previous step's retreating set for
    per-unit hysteresis (the very first step passes None/empty, so every unit
    is judged at the full entry threshold). Roles are mutually exclusive: a
    retreating unit never receives an attack from this allocation, and its
    damage is excluded from the cover budget.
    """
    params = {**PARAMS, **(parameters or {})}
    entry_candidates = detect_retreat_candidates(state, params)  # Entry threshold.
    retreat_candidates = detect_retreat_candidates(
        state, params, retreating=frozenset(retreating_units or ()))
    cover_units, retreat_units, other_units = assign_cover_roles(state, retreat_candidates)

    retreat_actions, retreat_telemetry = {}, {}
    for actor in retreat_units:
        action, reason, detail = select_retreat_action(
            state, actor, set(retreat_units), params, previous_action=None)
        retreat_actions[actor] = action  # None -> degraded fallback below.
        retreat_telemetry[actor] = {"reason": reason, **detail}

    priorities = threat_priority(state, retreat_units, cover_units) if retreat_units else []
    cover_assignments, cover_result = {}, None
    if cover_units and priorities:
        # FocusFireBudget sees only cover-role units: retreating units are
        # removed from the ally set so their damage is never budgeted.
        cover_state = dict(state)
        cover_state["allies"] = {a: state["allies"][a] for a in cover_units}
        cover_state["relations"] = {a: state["relations"][a] for a in cover_units}
        cover_result = allocate_focus_fire(
            cover_state, n_agents, priority_target=priorities[0],
            previous_assignments={a: t for a, t in (previous_assignments or {}).items()
                                  if a in cover_units})
        cover_assignments = cover_result["assignments"]

    actions = [0] * n_agents
    stop_fallbacks = []
    for actor, unit in state["allies"].items():
        index = int(actor[1:])
        if index >= n_agents or not _alive(unit):
            continue  # Dead units keep SMAC no-op 0.
        if actor in retreat_units:
            action = retreat_actions.get(actor)
            if action is None:
                action = fallback_action(state, actor)
        elif actor in cover_assignments:
            action = 6 + int(cover_assignments[actor][1:])
        else:
            action = fallback_action(state, actor)
        if action is None:
            # Alive unit with no attack/heal/move candidate at all: Stop is
            # the only explicit legal action left (it cancels the current
            # order — never a free no-op). Recorded, never silent.
            action = 1
            stop_fallbacks.append(actor)
        actions[index] = action

    reason = None
    if not retreat_units:
        reason = "no_retreat_needed"
    elif all(retreat_actions.get(a) is None for a in retreat_units) and not cover_assignments:
        reason = "no_executable_retreat_or_cover"

    return {
        "actions": actions,
        "retreat_units": retreat_units,
        "cover_units": cover_units,
        "other_units": other_units,
        "risks": {a: {"risk": v["risk"], "reasons": v["reasons"], "facts": v["facts"]}
                  for a, v in retreat_candidates.items()},
        "entry_danger_units": sorted(entry_candidates),
        "retreat_actions": retreat_telemetry,
        "threat_priority": priorities,
        "cover_assignments": cover_assignments,
        "cover_budget": (cover_result or {}).get("predicted_damage"),
        "cover_overkill": (cover_result or {}).get("estimated_overkill"),
        "cover_terminated": (cover_result or {}).get("termination_reason"),
        "stop_fallbacks": stop_fallbacks,
        # Safe means nobody would even ENTER retreat at the full entry
        # threshold; a thinner "below-continue-threshold" reading is not
        # treated as safety.
        "safe": not entry_candidates,
        "terminated": reason is not None,
        "termination_reason": reason,
    }
