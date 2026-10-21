"""AdaptiveFocusFire: focus fire by default, brief local retreats when they
pay off, and a sticky All-in phase when retreating no longer helps.

Composition (no duplicated algorithms):

- Danger detection and retreat move selection reuse CoverRetreat's
  ``detect_retreat_candidates`` / ``select_retreat_action``.
- All attacking units — even when NOBODY retreats — are assigned by
  FocusFireBudget's ``allocate_focus_fire`` restricted to the attacker set.
  The controller never degrades to per-unit nearest fallbacks.
- Healers (Medivacs) are Support: legal heal candidates or safe movement,
  never attack orders.

Model assumptions and scope (first version):

- Retreat value is an explainable heuristic over observed facts only (health
  ratio, recent observed loss, weapon readiness, legal targets, kill
  opportunity, existence of a safe legal move). It is NOT a survival-rate or
  future-damage prediction.
- "Combat progress" is measured only through observable visible-enemy health
  changes; hidden enemies are never assumed dead or damaged. Contact must
  have happened before the stall counter can trigger All-in, so a fresh
  scenario with no visible enemies never mis-triggers.
- All-in is sticky once entered (per episode, in manager state) and always
  recorded with its trigger reasons.
- Planned damage from FocusFireBudget is prediction, never actual damage.
"""

from .combat_timing import can_fire_within_window
from .cover_retreat import PARAMS as COVER_PARAMS, detect_retreat_candidates, select_retreat_action
from .focus_fire import allocate_focus_fire, expected_damage, fallback_action
from .movement_scoring import predicted_move, threat_count, score_movement_candidate
from .combat_timing import window_loops

PARAMS = {
    "max_retreat_steps": 2,       # per unit, per retreat episode
    "stall_steps": 6,             # contact made + no observable enemy damage this long -> reassess
    "all_in_team_ratio": 0.35,    # remaining-force ratio seen as severe disadvantage
    "max_retreat_cycles": 3,      # per unit: unproductive retreat episodes before calling it a loop
    "disable_retreat": False,     # ablation: no local retreats at all
    "disable_all_in": False,      # ablation: never enter All-in
    "max_recover_steps": 2,       # brief cooldown recovery after leaving attack candidates
    "search_memory_steps": 4,     # remembered positions are search hints, never attack targets
}

# Postures only tune the explainable thresholds; one template, three configs.
POSTURE_PARAMS = {
    "aggressive": {"critical_ratio": 0.15, "recent_loss_needed": 0.25},
    "balanced": {"critical_ratio": 0.25, "recent_loss_needed": 0.20},
    "defensive": {"critical_ratio": 0.35, "recent_loss_needed": 0.15},
}

DEFAULT_UNIT_STATE = {"role": "attack", "retreat_steps": 0, "retreat_cycles": 0}


def _is_healer(unit):
    return str(unit.get("type", "")).lower().startswith("medivac")


def _legal_attack_targets(state, actor):
    return [enemy_id for enemy_id, relation in state.get("relations", {}).get(actor, {}).items()
            if relation["smac_attack_available"]]


def _has_kill_opportunity(state, actor, parameters=None):
    """True when this unit can very likely kill a visible target this step.

    Uses the same window fire-timing as FocusFireBudget (shared helper), so
    the two skills can never disagree about whether a cooling-down unit can
    still shoot this control window.
    """
    unit = state["allies"][actor]
    fires = (lambda target: can_fire_within_window(state, actor, target,
                                                    parameters=parameters))         if (parameters or {}).get("window_fire") else (lambda target: unit["weapon_ready"])
    if (not unit["weapon_ready"] and not (parameters or {}).get("window_fire")):
        return False
    for enemy_id in _legal_attack_targets(state, actor):
        relation = state["relations"][actor][enemy_id]
        if relation["in_weapon_range"] is not True:
            continue
        if not fires(enemy_id):
            continue
        damage = expected_damage(state, actor, enemy_id)
        enemy = state["visible_enemies"][enemy_id]
        if damage is not None and damage >= enemy["health"] + enemy["shield"]:
            return True
    return False


def evaluate_retreat_value(state, actor, retreat_move, unit_state, posture="balanced",
                           parameters=None):
    """Should this endangered unit retreat this step? (bool, reason).

    Heuristic rules, in priority order:
    - A unit that can immediately kill a target keeps attacking.
    - A ready, in-range weapon with a legal target keeps attacking unless the
      unit is critical AND just lost a large chunk of health.
    - A unit on cooldown / out of range retreats only when a safe legal move
      exists, the retreat cap is not exhausted, and re-engagement is possible
      (it still has some legal attack relation to come back to).

    The per-unit retreat cap applies to EVERY retreat-returning branch —
    including the critical-unit branch. There is deliberately no emergency
    exception: a capped unit stays in the fire plan (or repositions via the
    normal attack path) instead of retreating forever.
    """
    params = {**PARAMS, **(parameters or {})}
    posture_params = POSTURE_PARAMS[posture]
    unit = state["allies"][actor]
    if unit_state.get("retreat_steps", 0) >= params["max_retreat_steps"]:
        if _has_kill_opportunity(state, actor, parameters):
            return False, "kill_opportunity"
        return False, "retreat_cap_reached"
    if _has_kill_opportunity(state, actor, parameters):
        return False, "kill_opportunity"
    if (parameters or {}).get("window_fire"):
        can_attack_now = any(can_fire_within_window(state, actor, e, parameters=parameters)
                             for e in _legal_attack_targets(state, actor))
    else:
        can_attack_now = (unit["weapon_ready"]
                          and any(state["relations"][actor][e]["in_weapon_range"] is not False
                                  for e in _legal_attack_targets(state, actor)))
    maximum = (unit["max_health"] or 0) + (unit["max_shield"] or 0) or 1
    ratio = (unit["health"] + unit["shield"]) / maximum
    recent_loss = (unit.get("health_lost_since_previous_decision", 0)
                   + unit.get("shield_lost_since_previous_decision", 0)) / maximum
    if can_attack_now:
        if ratio < posture_params["critical_ratio"] \
                and recent_loss >= posture_params["recent_loss_needed"]:
            return (retreat_move is not None), "critical_despite_weapon"
        return False, "weapon_engaged"
    if retreat_move is None:
        return False, "no_safe_move"
    if not _legal_attack_targets(state, actor):
        if params.get("range_preserving_move"):
            # Hand off to bounded recovery/re-entry instead of starting a
            # fresh retreat or immediately reversing while still on cooldown.
            return False, "cooldown_recovery" if (
                unit_state.get("role") in ("retreat", "recover")
                and unit["weapon_cooldown_loops"] > window_loops(state, parameters)
            ) else "reentry_required"
        return False, "cannot_reengage"
    return True, "cooldown_or_out_of_range"


def recovery_action(state, actor, history, previous_move, parameters):
    """Brief non-reversing recovery outside the attack mask, then re-enter.

    A deliberate Stop can hold position while the weapon recovers. It is not
    a no-op and is only used when no attack is currently legal.
    """
    if (history.get("role") not in ("retreat", "recover")
            or history.get("recover_steps", 0) >= parameters["max_recover_steps"]
            or _legal_attack_targets(state, actor)
            or not state["visible_enemies"]
            or state["allies"][actor]["weapon_cooldown_loops"] <= window_loops(state, parameters)):
        return None
    opposite = {2: 3, 3: 2, 4: 5, 5: 4}.get(previous_move)
    origin = state["allies"][actor]["position"]
    exposure = threat_count(state, actor, origin, parameters)
    target = min(state["relations"].get(actor, {}),
                 key=lambda e: state["relations"][actor][e]["distance"], default=None)
    options = []
    for action in state.get("movement_candidates", {}).get(actor, {}):
        # Lateral moves avoid both immediate reversal and extending the same
        # retreat past its cap. Holding is preferable to a worse exposure.
        if int(action) in (previous_move, opposite):
            continue
        move = predicted_move(state, actor, action)
        if threat_count(state, actor, move["destination"], parameters) <= exposure:
            options.append((score_movement_candidate(state, actor, action, target,
                                                      "tactical", parameters, previous_move),
                            int(action)))
    return min(options)[1] if options else 1


def should_enter_all_in(state, stall, contacted, unit_states, heal_capacity,
                        parameters=None):
    """Sticky All-in trigger. Returns (enter, reasons).

    Normal short retreats (threat -> brief move -> re-engage -> damage) are
    NOT penalized: ``retreat_cycles`` counts only a single unit's
    *unproductive* retreat episodes (retreating again without any observable
    enemy damage in between), maintained by the SkillManager. Two separate
    survival metrics are reported and never conflated:

    - ``alive_ratio``: average health state of currently living units.
    - ``remaining_force_ratio``: surviving effective health relative to the
      full roster's maximum (8 full-HP marines down to 1 -> 0.125 even
      though that survivor is at 100%).

    A stall alone never triggers (a healthy team may legitimately kite); it
    must combine with a real disadvantage, or one unit must be stuck in a
    genuinely unproductive retreat loop.
    """
    params = {**PARAMS, **(parameters or {})}
    if not contacted or not state["visible_enemies"]:
        return False, None  # Never mis-trigger before contact / with no visible enemy.
    alive = [u for u in state["allies"].values() if u["alive"]]
    if not alive:
        return False, None
    alive_max = sum((u["max_health"] or 0) + (u["max_shield"] or 0) for u in alive) or 1
    alive_ratio = sum(u["health"] + u["shield"] for u in alive) / alive_max
    roster_max = sum((u["max_health"] or 0) + (u["max_shield"] or 0)
                     for u in state["allies"].values()) or 1
    remaining_force_ratio = sum(u["health"] + u["shield"] for u in alive) / roster_max
    worst_cycles = max((u.get("retreat_cycles", 0) for u in unit_states.values()),
                       default=0)
    reasons = []
    if stall >= params["stall_steps"]:
        reasons.append("no_damage_progress")
    if remaining_force_ratio < params["all_in_team_ratio"] and not heal_capacity:
        reasons.append(f"low_remaining_force({remaining_force_ratio:.2f};"
                       f"alive_ratio={alive_ratio:.2f})_no_heal")
    if worst_cycles >= params["max_retreat_cycles"]:
        reasons.append(f"unproductive_retreat_loop(unit_cycles={worst_cycles})")
    if any(r.startswith("unproductive_retreat_loop") for r in reasons):
        return True, ";".join(reasons)
    if reasons and "no_damage_progress" in reasons and len(reasons) > 1:
        return True, ";".join(reasons)
    return False, None


def allocate_adaptive_focus_fire(state, n_agents, unit_states=None, all_in=False,
                                 stall=0, contacted=False, posture="balanced",
                                 parameters=None, previous_attack_assignments=None, move_history=None):
    """One AdaptiveFocusFire decision step on a fresh snapshot.

    ``unit_states`` carries per-unit role history (retreat step counts and
    cycles) owned by the SkillManager and persisted across 4-step skill
    windows within the episode, so re-activation cannot reset retreat caps.
    """
    params = {**PARAMS, **(parameters or {})}
    unit_states = unit_states or {}
    allies = state["allies"]

    heal_capacity = any(candidates for candidates in state.get("heal_candidates", {}).values())
    if all_in and not params["disable_all_in"]:
        all_in_now, all_in_reason = True, "sticky"
    elif params["disable_all_in"]:
        all_in_now, all_in_reason = False, None  # Ablation: never All-in, no back door.
    else:
        all_in_now, all_in_reason = should_enter_all_in(state, stall, contacted,
                                                        unit_states, heal_capacity, params)

    retreat_units, retreat_actions, retreat_reasons, movement_diagnostics = {}, {}, {}, {}
    if not all_in_now and not params["disable_retreat"]:
        # Danger thresholds follow the same risk_threshold CLI knob as
        # CoverRetreat when the caller supplies one.
        cover_parameters = {**COVER_PARAMS, **params}
        endangered = detect_retreat_candidates(state, cover_parameters)
        for actor in sorted(endangered, key=lambda a: int(a[1:])):
            unit = state["allies"][actor]
            maximum = (unit["max_health"] or 0) + (unit["max_shield"] or 0) or 1
            recent_loss = (unit.get("health_lost_since_previous_decision", 0)
                           + unit.get("shield_lost_since_previous_decision", 0)) / maximum
            movement_mode = ("emergency"
                             if (unit["health"] + unit["shield"]) / maximum < params.get("emergency_ratio", 0.15)
                             or recent_loss >= params.get("emergency_recent_loss", 0.5)
                             else "tactical")
            move, move_reason, detail = select_retreat_action(
                state, actor, {actor}, cover_parameters, movement_mode=movement_mode,
                previous_action=(move_history or {}).get(actor))
            value, reason = evaluate_retreat_value(
                state, actor, move, unit_states.get(actor, DEFAULT_UNIT_STATE),
                posture, params)
            movement_diagnostics[actor] = {
                **detail, "movement_mode": movement_mode, "reason": move_reason,
                "proposed_action": move, "retreat_accepted": value, "retreat_value_reason": reason}
            if value:
                retreat_units[actor] = True
                retreat_actions[actor] = move
                retreat_reasons[actor] = f"{reason};move={move_reason}"

    retreat_set = set(retreat_units)
    healers = sorted((a for a, u in allies.items() if u["alive"] and _is_healer(u)),
                     key=lambda a: int(a[1:]))
    attackers = sorted((a for a, u in allies.items() if u["alive"] and a not in retreat_set
                        and a not in healers and _legal_attack_targets(state, a)),
                        key=lambda a: int(a[1:]))

    # Everyone able to fight goes through FocusFireBudget together —
    # including the no-retreater case. Retreating units are excluded from
    # the budget's ally set, so their damage is never counted while fleeing.
    attack_result = None
    if attackers:
        restricted = dict(state)
        restricted["allies"] = {a: allies[a] for a in attackers}
        restricted["relations"] = {a: state["relations"][a] for a in attackers}
        attack_result = allocate_focus_fire(
            restricted, n_agents,
            previous_assignments={a: t for a, t in (previous_attack_assignments or {}).items()
                                  if a in attackers},
            parameters={k: v for k, v in (parameters or {}).items()
                        if k in ("window_fire", "target_scoring", "window_tolerance")})
        attack_assignments = attack_result["assignments"]
    else:
        attack_assignments = {}

    actions, roles, reentry_decisions = [0] * n_agents, {}, {}
    for actor, unit in allies.items():
        index = int(actor[1:])
        if index >= n_agents or not unit["alive"]:
            continue  # Dead units: SMAC no-op 0 only.
        if actor in retreat_set:
            actions[index] = retreat_actions[actor]
            roles[actor] = "retreat"
        elif actor in attack_assignments or actor in attackers:
            actions[index] = attack_result["actions"][index]
            roles[actor] = "all_in_attack" if all_in_now else "attack"
        elif actor in healers:
            actions[index] = fallback_action(state, actor) or 1  # Support: heal or move.
            roles[actor] = "support"
        else:
            recover = recovery_action(
                state, actor, unit_states.get(actor, {}), (move_history or {}).get(actor), params
            ) if params.get("range_preserving_move") and not all_in_now else None
            if recover is not None:
                actions[index], roles[actor] = recover, "recover"
                reentry_decisions[actor] = {"reason": "bounded_cooldown_recovery", "action": recover}
            else:
                actions[index] = fallback_action(state, actor) or 1
                roles[actor] = "reposition"

    reengaged = sorted((a for a in roles
                        if roles[a] in ("attack", "all_in_attack")
                        and unit_states.get(a, DEFAULT_UNIT_STATE).get("role") in ("retreat", "recover")),
                       key=lambda a: int(a[1:]))
    reason = None
    if not any(u["alive"] for u in allies.values()):
        reason = "all_units_dead"

    return {
        "actions": actions,
        "phase": "all_in" if all_in_now else ("mixed" if retreat_set else "engage"),
        "all_in": all_in_now,
        "all_in_reason": all_in_reason,
        "attack_units": sorted((a for a in roles if roles[a] in ("attack", "all_in_attack")),
                               key=lambda a: int(a[1:])),
        "retreat_units": sorted(retreat_set, key=lambda a: int(a[1:])),
        "support_units": sorted((a for a in roles if roles[a] == "support"),
                                key=lambda a: int(a[1:])),
        "attack_assignments": attack_assignments,
        "retreat_actions": retreat_actions,
        "retreat_reasons": retreat_reasons,
        "roles": roles,
        "reengaged": reengaged,
        "predicted_damage": (attack_result or {}).get("predicted_damage"),
        "estimated_overkill": (attack_result or {}).get("estimated_overkill"),
        "immediate_damage": (attack_result or {}).get("immediate_damage", {}),
        "window_only_damage": (attack_result or {}).get("window_only_damage", {}),
        "window_fire_enabled": bool(params.get("window_fire")),
        "target_scoring_enabled": bool(params.get("target_scoring")),
        "range_preserving_move_enabled": bool(params.get("range_preserving_move")),
        "movement_diagnostics": movement_diagnostics,
        "reentry_decisions": reentry_decisions,
        "focus_fire_terminated": (attack_result or {}).get("termination_reason"),
        "terminated": reason is not None,
        "termination_reason": reason,
    }
