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

from .cover_retreat import PARAMS as COVER_PARAMS, detect_retreat_candidates, select_retreat_action
from .focus_fire import allocate_focus_fire, expected_damage, fallback_action

PARAMS = {
    "max_retreat_steps": 2,       # per unit, per retreat episode
    "stall_steps": 6,             # contact made + no observable enemy damage this long -> reassess
    "all_in_team_ratio": 0.35,    # remaining-force ratio seen as severe disadvantage
    "max_retreat_cycles": 3,      # per unit: unproductive retreat episodes before calling it a loop
    "disable_retreat": False,     # ablation: no local retreats at all
    "disable_all_in": False,      # ablation: never enter All-in
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


def _has_kill_opportunity(state, actor):
    """True when this unit can very likely kill a visible target this step."""
    unit = state["allies"][actor]
    if not unit["weapon_ready"]:
        return False
    for enemy_id in _legal_attack_targets(state, actor):
        relation = state["relations"][actor][enemy_id]
        if relation["in_weapon_range"] is not True:
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
        if _has_kill_opportunity(state, actor):
            return False, "kill_opportunity"
        return False, "retreat_cap_reached"
    if _has_kill_opportunity(state, actor):
        return False, "kill_opportunity"
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
        return False, "cannot_reengage"
    return True, "cooldown_or_out_of_range"


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
                                 parameters=None, previous_attack_assignments=None):
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

    retreat_units, retreat_actions, retreat_reasons = {}, {}, {}
    if not all_in_now and not params["disable_retreat"]:
        # Danger thresholds follow the same risk_threshold CLI knob as
        # CoverRetreat when the caller supplies one.
        cover_parameters = ({k: params[k]} if "risk_threshold" in params else None)
        endangered = detect_retreat_candidates(state, cover_parameters)
        for actor in sorted(endangered, key=lambda a: int(a[1:])):
            move, move_reason, _ = select_retreat_action(
                state, actor, {actor}, cover_parameters)
            value, reason = evaluate_retreat_value(
                state, actor, move, unit_states.get(actor, DEFAULT_UNIT_STATE),
                posture, params)
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
                                  if a in attackers})
        attack_assignments = attack_result["assignments"]
    else:
        attack_assignments = {}

    actions, roles = [0] * n_agents, {}
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
            actions[index] = fallback_action(state, actor) or 1  # Reposition: explicit legal move.
            roles[actor] = "reposition"

    reengaged = sorted((a for a in roles
                        if roles[a] in ("attack", "all_in_attack")
                        and unit_states.get(a, DEFAULT_UNIT_STATE).get("role") == "retreat"),
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
        "focus_fire_terminated": (attack_result or {}).get("termination_reason"),
        "terminated": reason is not None,
        "termination_reason": reason,
    }
