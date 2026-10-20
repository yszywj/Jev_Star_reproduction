"""Shared fire-timing helpers for the combat skills.

Inspired by Sharpy's ``ready_to_shoot()`` (default_micro_methods.py): in
fixed-step mode a weapon whose cooldown expires *within the current control
window* can still fire this window if the unit keeps attacking, so retreat
decisions must not treat "cooldown > 0 right now" as "cannot contribute".

This is a re-design for SMAC-Hard's fixed step_mul, not a copy: the window
length comes from the snapshot's ``step_mul`` and the tolerance is an
explicit parameter instead of Sharpy's real-time constant.

Three strictly separate concepts:

- ``weapon_ready``: cooldown is zero right now (unchanged observation field).
- ``can_fire_within_window``: if the unit keeps attacking, it plausibly gets
  one shot off during this control window. A prediction, never a hit.
- actual damage: only what the game observation later shows.
"""

PARAMS = {
    "window_tolerance": 1.0,  # extra loops beyond step_mul still counted as in-window
    "default_step_mul": 8,
}


def window_loops(state, parameters=None):
    """Length of the current control window in game loops."""
    params = {**PARAMS, **(parameters or {})}
    step_mul = state.get("step_mul") or params["default_step_mul"]
    return step_mul + params["window_tolerance"]


def can_fire_within_window(state, actor, target_id, window=None, parameters=None):
    """True when this attacker plausibly fires at target_id this window.

    Conservative checks: the SMAC attack action must be legal now, the enemy
    must be visible, range must not be known to be exceeded, and the weapon
    must be compatible (legality already encodes that). Unknown range counts
    as possible (matching the observation's own "unknown" semantics) but is
    never turned into damage by the callers without expected_damage support.
    """
    if window is None:
        window = window_loops(state, parameters)
    ally = state["allies"].get(actor)
    if ally is None or not ally.get("alive"):
        return False
    relation = state.get("relations", {}).get(actor, {}).get(target_id)
    if relation is None or not relation.get("smac_attack_available"):
        return False  # Illegal, invisible or incompatible target.
    if relation.get("in_weapon_range") is False:
        return False  # Known to be out of native weapon range.
    cooldown = ally.get("weapon_cooldown_loops", 0) or 0
    return cooldown <= window


def window_fire_profile(state, actor, parameters=None):
    """Per-attacker timing summary for diagnostics (never actual damage)."""
    ally = state["allies"].get(actor, {})
    window = window_loops(state, parameters)
    cooldown = ally.get("weapon_cooldown_loops", 0) or 0
    candidates = {}
    for target_id, relation in state.get("relations", {}).get(actor, {}).items():
        candidates[target_id] = {
            "legal": bool(relation.get("smac_attack_available")),
            "in_range": relation.get("in_weapon_range"),
            "can_fire_within_window": can_fire_within_window(state, actor, target_id,
                                                             window, parameters),
        }
    return {
        "weapon_cooldown_loops": cooldown,
        "weapon_ready": bool(ally.get("weapon_ready")),
        "window_loops": window,
        "targets": candidates,
    }
