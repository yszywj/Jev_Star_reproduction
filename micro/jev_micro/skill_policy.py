"""JEV team-level skill selection: one Choice question, not per-unit decoding."""

from .client import validate_response

SKILL_SCHEMA_VERSION = 1

DIRECT_CONTROL = "direct_control"
FOCUS_FIRE_BUDGET = "focus_fire_budget"
COVER_RETREAT = "cover_retreat"
HEAL_ASSIGN = "heal_assign"

SKILL_INSTRUCTIONS = (
    "Choose the squad control mode for this StarCraft II micro battle step window. "
    "With direct_control you keep choosing each surviving ally's individual action "
    "every step. With focus_fire_budget a local executor concentrates fire: it ranks "
    "visible targets, assigns just enough attackers to kill each target based on "
    "predicted damage, and moves remaining attackers to the next target; explicit "
    "focus_fire_budget candidates also name the priority target. With cover_retreat "
    "endangered allies retreat along legal moves while remaining combatants cover "
    "them with budgeted fire. With heal_assign Medivacs keep a joint, one-to-one "
    "healer/target assignment while other units keep their current control policy. "
    "Pick the mode and priority that fit the state and plan."
)

MAX_EXPLICIT_TARGETS = 4


def skill_choices(visible_enemies, healers_present=True, max_explicit_targets=MAX_EXPLICIT_TARGETS):
    """Dynamic candidate table: 0=direct, 1=auto focus, 2=cover_retreat,
    3=heal_assign (only when the snapshot has healers), then explicit focus
    targets bound to the currently visible enemies.

    Candidates are bound to the current snapshot only; no fixed number of
    targets or healers is assumed. Unavailable skills never get a candidate.
    """
    criteria = {
        "0": "direct_control: JEV chooses every surviving ally's action each step.",
        "1": ("focus_fire_budget: local damage-budget allocator picks its own "
              "deterministic target priority."),
        "2": ("cover_retreat: endangered allies retreat along scored legal moves "
              "while able allies cover them with focus fire."),
    }
    priorities = {"0": None, "1": None, "2": None}
    modes = {"0": DIRECT_CONTROL, "1": FOCUS_FIRE_BUDGET, "2": COVER_RETREAT}
    next_id = 3
    if healers_present:
        criteria["3"] = ("heal_assign: Medivacs get a joint one-to-one healer/target "
                         "matching; non-healers keep the declared control policy.")
        priorities["3"] = None
        modes["3"] = HEAL_ASSIGN
        next_id = 4
    for offset, enemy_id in enumerate(sorted(visible_enemies, key=lambda e: int(e[1:]))[:max_explicit_targets]):
        choice = str(next_id + offset)
        criteria[choice] = (f"focus_fire_budget with priority target {enemy_id}: "
                            "attackers concentrate on it first, then spill to other targets.")
        priorities[choice] = enemy_id
        modes[choice] = FOCUS_FIRE_BUDGET
    return criteria, priorities, modes


def skill_request(payload, model, skill_max_steps, max_explicit_targets=MAX_EXPLICIT_TARGETS,
                  active_skill=None, last_termination=None):
    """Build a compact team-level Choice request from an existing snapshot.

    Uses only shared team-visible state already present in the payload
    (never evaluation globals or hidden enemy state) plus the Astra plan
    digest when one is attached.
    """
    state = payload["state"]
    allies, relations = state["allies"], state.get("relations", {})
    coverage = {}
    for enemy_id, enemy in state["visible_enemies"].items():
        legal = ready = 0
        for actor, actor_relations in relations.items():
            relation = actor_relations.get(enemy_id)
            if relation and relation["smac_attack_available"]:
                legal += 1
                ready += bool(allies[actor]["alive"] and allies[actor]["weapon_ready"])
        coverage[enemy_id] = {"effective_health": enemy["effective_health"],
                              "type": enemy["type"], "legal_attackers": legal,
                              "ready_legal_attackers": ready}
    # Compact team risk/heal summary for skill choice, from the same snapshot
    # only (entry-threshold danger detection reused from CoverRetreat).
    from .skills.cover_retreat import detect_retreat_candidates
    endangered = detect_retreat_candidates(state)
    wounded = sorted((actor for actor, u in allies.items()
                      if u["alive"] and u["health"] < u["max_health"]
                      and (u["max_health"] - u["health"]) >= max(1.0, u["max_health"] * 0.1)),
                     key=lambda a: int(a[1:]))
    recently_damaged = {actor: round(u.get("health_lost_since_previous_decision", 0)
                                     + u.get("shield_lost_since_previous_decision", 0), 2)
                        for actor, u in allies.items() if u["alive"]
                        and (u.get("health_lost_since_previous_decision", 0)
                             + u.get("shield_lost_since_previous_decision", 0)) > 0}
    heal_demand = len({entry["target"]
                       for candidates in state.get("heal_candidates", {}).values()
                       for entry in candidates.values()})
    skill_state = {
        "objective": state["objective"],
        "skill_schema_version": SKILL_SCHEMA_VERSION,
        "map": state["map"], "episode": state["episode"], "game_loop": state["game_loop"],
        "steps_remaining": state["steps_remaining"],
        "mode_window_steps": skill_max_steps,
        "live_allies": sum(u["alive"] for u in allies.values()),
        "healers_present": (any(u["alive"] and str(u.get("type", "")).lower().startswith("medivac")
                                for u in allies.values()) and heal_demand > 0),
        "team_risk_summary": {
            "endangered_units": sorted(endangered, key=lambda a: int(a[1:])),
            "low_health_units": [actor for actor in wounded
                                 if allies[actor]["health"] / (allies[actor]["max_health"] or 1) < 0.35],
            "recently_damaged": recently_damaged,
            "threatened_unit_count": sum(1 for actor, u in allies.items() if u["alive"] and any(
                r["distance"] <= 8.0 or r.get("in_enemy_weapon_range") is True
                for r in relations.get(actor, {}).values())),
            "legal_healers": sum(1 for candidates in state.get("heal_candidates", {}).values()
                                 if candidates),
            "units_needing_heal": heal_demand,
        },
        "active_skill": active_skill,
        "last_skill_termination": last_termination,
        "allies_core": {actor: {"type": u["type"], "effective_health": u["effective_health"],
                                "weapon_ready": u["weapon_ready"]}
                        for actor, u in allies.items() if u["alive"]},
        "visible_enemy_coverage": coverage,
        "scenario_brief": state.get("scenario_brief"),
    }
    plan = state.get("strategic_plan")
    if plan is not None:
        skill_state["strategic_plan"] = {
            "plan_sha256": plan.get("plan_sha256"),
            "objective": plan.get("objective"),
            "target_selection": plan.get("target_selection"),
        }
    criteria, _, _ = skill_choices(state["visible_enemies"],
                                   healers_present=skill_state["healers_present"],
                                   max_explicit_targets=max_explicit_targets)
    return {"model": model, "state": skill_state,
            "questions": {"team_skill": {
                "type": "choice", "instructions": SKILL_INSTRUCTIONS,
                "criteria": criteria}}}


def decode_skill(payload, answers):
    """Validate and decode the single team_skill answer.

    Returns (mode, priority_target): priority_target is an enemy ID string for
    explicit focus candidates, else None. Never routed through decode_actions().
    """
    try:
        answer = answers["team_skill"]
    except (KeyError, TypeError):
        raise ValueError("Missing team_skill answer.") from None
    choice = answer.get("choice") if isinstance(answer, dict) else None
    visible = payload["state"].get("visible_enemy_coverage", payload["state"].get("visible_enemies", {}))
    healers_present = payload["state"].get("healers_present", True)
    criteria, priorities, modes = skill_choices(visible, healers_present=healers_present)
    if choice not in criteria:
        raise ValueError("Skill choice is outside the candidate set.")
    return modes[choice], priorities[choice]


def validate_skill_response(data, payload):
    """Same strict Choice validation as unit answers, kept separate by key set."""
    return validate_response(data, payload)
