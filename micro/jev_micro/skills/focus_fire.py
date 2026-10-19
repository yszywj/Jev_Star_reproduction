"""FocusFireBudget: a formation-size-independent focus-fire allocator.

Model assumptions and scope (first version, deliberately conservative):

- The allocator works on one snapshot of a ``build_request`` payload state
  (allies / visible_enemies / relations / unit_type_profiles /
  movement_candidates / heal_candidates). It never reads hidden enemy state:
  only enemies present in ``state["visible_enemies"]`` can ever be attacked.
- Predicted damage covers exactly one pending attack per attacker: a unit
  whose weapon is ready, whose SMAC attack action is legal and whose weapon
  range is not known to be exceeded contributes its expected single-attack
  damage to the budget. Attackers on cooldown or out of range may still be
  assigned (attack orders persist through cooldown) but contribute nothing to
  the budget this step, so the budget under-counts rather than over-counts.
- Expected damage per attack uses the public combat profile weapons
  (base_damage * hits_per_attack - target armor, floor 0.5 per SC2 rules for
  known damage). If any required datum is unknown (effect-based weapons,
  missing API weapon lists, unknown range), that attacker's damage is treated
  as unknown: it is assigned but never accumulated into the budget.
- The budget is needs-based, recomputed from the current observation every
  step: a target's need is its observed effective health. Earlier versions
  subtracted a cross-step ``pending_damage`` ledger, but the observation
  cannot reliably tell a landed attack from an in-flight one, so that ledger
  could double-subtract damage already reflected in observed health. The
  ledger is therefore disabled (option A): each step re-derives the budget
  from fresh observations only. Consequence, documented: attacks ordered but
  not yet resolved are not credited, so the budget can overestimate the
  firepower still needed and briefly over-assign attackers (visible as
  estimated overkill). That is deliberately preferred over double-counting.
- SC2 ResponseAction success is never interpreted as damage dealt.
"""

import math

NO_ATTACK_FALLBACK = None  # A living unit with no legal action at all.


def effective_health(enemy):
    return enemy["health"] + enemy["shield"]


def _compatible_weapon(profile, enemy):
    """Return the profile weapon usable against this target, or None."""
    target_types = {2, 3} if enemy["is_flying"] else {1, 3}
    if enemy["unit_type_id"] == 4:  # Colossus: also hittable by anti-air.
        target_types.add(2)
    for weapon in profile.get("weapons", []):
        kind = {"ground": 1, "air": 2, "any": 3}.get(weapon.get("target_type"), 3)
        if kind in target_types:
            return weapon
    return None


def expected_damage(state, actor, enemy_id):
    """Damage one attack from actor is expected to deal, or None if unknown.

    Armor is applied per hit (SC2 semantics): two hits of 5 damage against 2
    armor deal (5-2)*2 = 6, not 10-2 = 8. Unknown weapon data OR unknown
    target armor yields None — never a fabricated number, and never a
    zero-armor assumption that would overestimate the budget.
    """
    ally = state["allies"][actor]
    enemy = state["visible_enemies"][enemy_id]
    profile = state.get("unit_type_profiles", {}).get(str(ally["unit_type_id"]), {})
    enemy_profile = state.get("unit_type_profiles", {}).get(str(enemy["unit_type_id"]), {})
    weapon = _compatible_weapon(profile, enemy)
    if weapon is None or not profile.get("known", False):
        return None
    base = weapon.get("base_damage")
    hits = weapon.get("hits_per_attack") or 1
    if base is None:  # Effect-based weapons omitted from the API.
        return None
    if not enemy_profile.get("known", False):
        return None  # Target armor unknown: refuse to guess zero armor.
    armor = enemy_profile.get("armor")
    if armor is None:
        return None
    per_hit = max(base - armor, 0.5)  # SC2 floors each hit at 0.5.
    return round(per_hit * hits, 3)


def _countable(state, actor, relation):
    """True when this attacker's next attack is plausibly imminent and known."""
    return (state["allies"][actor]["weapon_ready"]
            and relation["smac_attack_available"]
            and relation["in_weapon_range"] is not False)


def _attacker_order(state, actor, enemy_id, relation, damage, previous_assignments):
    """Deterministic attacker preference for one target (smaller is better)."""
    return (
        0 if (previous_assignments or {}).get(actor) == enemy_id else 1,  # hysteresis
        0 if relation["in_weapon_range"] is True else 1,
        0 if state["allies"][actor]["weapon_ready"] else 1,
        -(damage if damage is not None else 0.0),
        int(actor[1:]),
    )


def _target_order(state, enemy_id, priority_target, legal_attackers, damage_sum):
    """Deterministic target priority (smaller is better).

    Priority target first, then most attackers able to hit it, then cheapest
    to kill, then lowest effective health, then enemy ID as final tie-break.
    """
    enemy = state["visible_enemies"][enemy_id]
    need = effective_health(enemy)
    attacks_to_kill = math.ceil(need / damage_sum) if damage_sum > 0 else float("inf")
    return (
        0 if enemy_id == priority_target else 1,
        -legal_attackers,
        attacks_to_kill,
        need,
        int(enemy_id[1:]),
    )


def target_priority(state, priority_target=None, previous_assignments=None):
    """Ordered list of visible enemy IDs with deterministic tie-breaking."""
    relations = state.get("relations", {})
    priority_target = priority_target if priority_target in state["visible_enemies"] else None
    rows = []
    for enemy_id in state["visible_enemies"]:
        legal = ready_damage = 0
        for actor, actor_relations in relations.items():
            relation = actor_relations.get(enemy_id)
            if relation is None:
                continue
            if relation["smac_attack_available"]:
                legal += 1
                if _countable(state, actor, relation):
                    damage = expected_damage(state, actor, enemy_id)
                    if damage is not None:
                        ready_damage += damage
        rows.append((_target_order(state, enemy_id, priority_target, legal, ready_damage), enemy_id))
    return [enemy_id for _, enemy_id in sorted(rows)]


def allocate_focus_fire(state, n_agents, priority_target=None,
                        pending_damage=None, previous_assignments=None):
    """Assign attackers to visible targets under a per-target damage budget.

    ``pending_damage`` is accepted for interface compatibility but ignored
    (see module docstring: cross-step pending ledgers can double-count damage
    already reflected in observed health). The budget is recomputed from the
    current observation each step.

    Returns a plain dict with the variable-length assignment map, the
    environment-length action list, budget telemetry and a termination hint.
    Every emitted attack action is legal in the snapshot; units that cannot
    attack anything receive an explicit deterministic fallback action.
    """
    previous_assignments = previous_assignments or {}
    allies, relations = state["allies"], state.get("relations", {})
    targets = target_priority(state, priority_target, previous_assignments)
    attackers = sorted((actor for actor, unit in allies.items() if unit["alive"]), key=lambda a: int(a[1:]))

    damage = {(actor, enemy_id): expected_damage(state, actor, enemy_id)
              for actor in attackers for enemy_id in targets}
    need = {enemy_id: max(0.0, effective_health(state["visible_enemies"][enemy_id])) for enemy_id in targets}
    predicted = {enemy_id: 0.0 for enemy_id in targets}
    assignments, unknown_damage = {}, {}

    free = list(attackers)
    for enemy_id in targets:
        ordered = sorted(
            (actor for actor in free
             if relations.get(actor, {}).get(enemy_id, {}).get("smac_attack_available")),
            key=lambda actor: _attacker_order(state, actor, enemy_id,
                                              relations[actor][enemy_id], damage[(actor, enemy_id)],
                                              previous_assignments))
        for actor in ordered:
            if predicted[enemy_id] >= need[enemy_id]:
                break  # Budget met; keep remaining attackers for later targets.
            assignments[actor] = enemy_id
            free.remove(actor)
            pair = damage[(actor, enemy_id)]
            if _countable(state, actor, relations[actor][enemy_id]):
                if pair is None:
                    unknown_damage[actor] = enemy_id  # Assigned but not budgeted.
                else:
                    predicted[enemy_id] += pair
    # Attackers with no budget slot left still fire somewhere useful: their
    # best legal target in priority order. Their damage becomes overkill.
    overflow = {}
    for actor in free:
        options = [enemy_id for enemy_id in targets
                   if relations.get(actor, {}).get(enemy_id, {}).get("smac_attack_available")]
        if options:
            assignments[actor] = options[0]
            overflow[actor] = options[0]
            if _countable(state, actor, relations[actor][options[0]]) and damage[(actor, options[0])] is not None:
                predicted[options[0]] += damage[(actor, options[0])]

    overkill = {enemy_id: round(max(0.0, predicted[enemy_id]
                                    - effective_health(state["visible_enemies"][enemy_id])), 2)
                for enemy_id in targets}
    # Predicted shortfall against each target's survival: 0 once the budget
    # covers it. Positive after all attackers are spent means the budget is
    # insufficient this step (prediction only, never a kill guarantee).
    budget_deficit = {enemy_id: round(max(0.0, need[enemy_id] - predicted[enemy_id]), 2)
                      for enemy_id in targets}

    actions, fallbacks, unexecutable = [0] * n_agents, {}, []
    for actor in attackers:
        index = int(actor[1:])
        if index >= n_agents:
            continue
        enemy_id = assignments.get(actor)
        if enemy_id is not None:
            actions[index] = 6 + int(enemy_id[1:])
            continue
        action = fallback_action(state, actor)
        if action is NO_ATTACK_FALLBACK:
            unexecutable.append(actor)
        else:
            actions[index] = action
            fallbacks[actor] = action

    reason = None
    if not state["visible_enemies"]:
        reason = "no_visible_targets"
    elif not assignments:
        reason = "no_legal_attacks"
    elif unexecutable:
        reason = "units_without_legal_actions"

    return {
        "assignments": assignments,
        "actions": actions,
        "target_priority": targets,
        "predicted_damage": {enemy_id: round(value, 2) for enemy_id, value in predicted.items()},
        "pending_damage_applied": {},  # Cross-step ledger disabled; see docstring.
        "estimated_overkill": overkill,
        "budget_deficit": budget_deficit,
        "unknown_damage_attackers": unknown_damage,
        "overflow_attackers": overflow,
        "unassigned_units": sorted(free, key=lambda a: int(a[1:])),
        "fallback_actions": fallbacks,
        "participating_units": sorted(assignments, key=lambda a: int(a[1:])),
        "unexecutable_units": unexecutable,
        "terminated": reason is not None,
        "termination_reason": reason,
    }


def fallback_action(state, actor):
    """Explicit deterministic legal fallback: attack, heal, then move.

    Returns an action ID valid for this snapshot, or None when the unit has
    no legal non-stop action at all. Stop is never used as a no-op.
    """
    relations = state.get("relations", {}).get(actor, {})
    attacks = sorted((enemy_id for enemy_id, relation in relations.items()
                      if relation["smac_attack_available"]),
                     key=lambda enemy_id: (relations[enemy_id]["distance"], int(enemy_id[1:])))
    if attacks:
        return 6 + int(attacks[0][1:])
    heals = state.get("heal_candidates", {}).get(actor, {})
    if heals:
        return int(min(heals, key=lambda a: (heals[a]["distance"], int(a))))
    moves = state.get("movement_candidates", {}).get(actor, {})
    if moves:
        def move_key(action_id):
            entry = moves[action_id]
            distances = list(entry["distances_to_visible_enemies"].values())
            return (min(distances) if distances else entry["distance_to_search_location"],
                    int(action_id))
        return int(min(moves, key=move_key))
    return NO_ATTACK_FALLBACK
