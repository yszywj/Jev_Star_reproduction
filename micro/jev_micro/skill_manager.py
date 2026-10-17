"""Skill lifecycle: activation, multi-step execution, termination, telemetry."""

from .skill_policy import DIRECT_CONTROL, FOCUS_FIRE_BUDGET, COVER_RETREAT, HEAL_ASSIGN
from .skills.focus_fire import allocate_focus_fire
from .skills.cover_retreat import allocate_cover_retreat, PARAMS as COVER_PARAMS
from .skills.heal_assign import assign_healing, PARAMS as HEAL_PARAMS


class SkillManager:
    """Tracks the currently selected control mode and one active skill run.

    One manager instance spans the whole experiment; ``reset`` must be called
    at every episode start so no state leaks across episodes.
    """

    def __init__(self, max_steps=4, risk_threshold=None, safe_streak_required=2,
                 heal_switch_margin=None):
        self.max_steps = max_steps
        self.safe_streak_required = max(1, safe_streak_required)
        self.cover_params = dict(COVER_PARAMS)
        if risk_threshold is not None:
            self.cover_params["risk_threshold"] = risk_threshold
        self.heal_params = dict(HEAL_PARAMS)
        if heal_switch_margin is not None:
            self.heal_params["switch_margin"] = heal_switch_margin
        self.stats = {"activations": 0, "steps": 0, "terminations": {},
                      "reassignments": 0, "estimated_overkill": 0.0,
                      "predicted_damage": 0.0, "selections": {},
                      "fallback_steps": 0, "role_reassignments": 0,
                      "safe_streak_exits": 0, "retreat_unit_steps": 0,
                      "heal_assignments_created": 0, "heal_target_switches": 0,
                      "heal_self_preserves": 0, "heal_no_target_steps": 0}
        self.reset()

    def reset(self):
        """Clear per-episode execution state; cumulative stats survive for the run."""
        self.current_skill = None
        self.skill_start_step = None
        self.elapsed_steps = 0
        self.mode_remaining = 0
        self.assigned_units = []
        self.target_assignments = {}
        self.priority_target = None
        self.termination_reason = None
        self.last_observation_step = None
        self.retreat_units = []
        self.cover_units = []
        self.retreat_direction = None
        self.safe_streak = 0
        self._pending_damage = {}

    # -- selection -------------------------------------------------------

    def select(self, skill, priority_target=None):
        """JEV (or --skill-mode always) chose a mode; hold it for one window."""
        if skill not in (DIRECT_CONTROL, FOCUS_FIRE_BUDGET, COVER_RETREAT, HEAL_ASSIGN):
            raise ValueError(f"Unknown skill: {skill}")
        self.current_skill = skill
        self.priority_target = priority_target
        self.mode_remaining = self.max_steps
        self.stats["selections"][skill] = self.stats["selections"].get(skill, 0) + 1
        if skill != DIRECT_CONTROL:
            self._start_skill()

    def _start_skill(self):
        self.skill_start_step = None
        self.elapsed_steps = 0
        self.assigned_units = []
        self.target_assignments = {}
        self.termination_reason = None
        self.retreat_units = []
        self.cover_units = []
        self.retreat_direction = None
        self.safe_streak = 0
        self._pending_damage = {}
        self.stats["activations"] += 1

    @property
    def active(self):
        return (self.current_skill in (FOCUS_FIRE_BUDGET, COVER_RETREAT, HEAL_ASSIGN)
                and self.termination_reason is None)

    # -- execution -------------------------------------------------------

    def step(self, payload, n_agents):
        """Run one local decision step of the active skill on a fresh snapshot.

        Returns (actions, telemetry). ``actions`` is always a full
        n_agents-length list of legal snapshot actions.
        """
        if self.current_skill == COVER_RETREAT:
            return self._step_cover_retreat(payload, n_agents)
        if self.current_skill == HEAL_ASSIGN:
            return self._step_heal_assign(payload, n_agents)
        return self._step_focus_fire(payload, n_agents)

    def _step_focus_fire(self, payload, n_agents):
        state = payload["state"]
        result = allocate_focus_fire(state, n_agents, priority_target=self.priority_target,
                                     pending_damage=self._pending_damage,
                                     previous_assignments=self.target_assignments or None)
        reassigned = bool(self.target_assignments) and result["assignments"] != self.target_assignments
        if reassigned:
            self.stats["reassignments"] += 1
        self.last_observation_step = state["game_loop"]
        self.skill_start_step = self.skill_start_step if self.skill_start_step is not None \
            else state["game_loop"]
        self.elapsed_steps += 1
        self.assigned_units = result["participating_units"]
        self.target_assignments = result["assignments"]
        self.stats["steps"] += 1
        self.stats["estimated_overkill"] += sum(result["estimated_overkill"].values())
        self.stats["predicted_damage"] += sum(result["predicted_damage"].values())
        # Predicted-but-not-yet-observed damage for the next step's budget.
        self._pending_damage = dict(result["predicted_damage"])
        telemetry = {"skill": "focus_fire_budget", "skill_step": self.elapsed_steps,
                     "skill_start_step": self.skill_start_step,
                     "participating_units": result["participating_units"],
                     "visible_targets": sorted(state["visible_enemies"]),
                     "target_priority": result["target_priority"],
                     "target_assignments": result["assignments"],
                     "predicted_damage": result["predicted_damage"],
                     "pending_damage_applied": result["pending_damage_applied"],
                     "estimated_overkill": result["estimated_overkill"],
                     "budget_deficit": result["budget_deficit"],
                     "reassigned": reassigned,
                     "unknown_damage_attackers": result["unknown_damage_attackers"],
                     "unassigned_units": result["unassigned_units"],
                     "fallback_actions": result["fallback_actions"]}
        if result["terminated"]:
            self._terminate(result["termination_reason"])
        elif self.elapsed_steps >= self.max_steps:
            self._terminate("max_steps")
        if self.termination_reason is not None:
            telemetry["termination_reason"] = self.termination_reason
        return result["actions"], telemetry

    def _step_cover_retreat(self, payload, n_agents):
        state = payload["state"]
        result = allocate_cover_retreat(state, n_agents, parameters=self.cover_params,
                                        previous_assignments=self.target_assignments or None,
                                        continuing=True)
        self.last_observation_step = state["game_loop"]
        self.skill_start_step = self.skill_start_step if self.skill_start_step is not None \
            else state["game_loop"]
        self.elapsed_steps += 1
        self.stats["steps"] += 1
        self.stats["retreat_unit_steps"] += len(result["retreat_units"])
        if (self.retreat_units or self.cover_units) and \
                (result["retreat_units"] != self.retreat_units
                 or result["cover_units"] != self.cover_units):
            self.stats["role_reassignments"] += 1
        self.retreat_units = result["retreat_units"]
        self.cover_units = result["cover_units"]
        self.target_assignments = result["cover_assignments"]
        if result["retreat_units"]:
            first = result["retreat_units"][0]
            action = result["actions"][int(first[1:])]
            self.retreat_direction = action if action in (2, 3, 4, 5) else None
        # Exit hysteresis: only leave after N consecutive threat-free steps,
        # so low-health-but-safe units do not oscillate retreat/attack.
        if result["safe"]:
            self.safe_streak += 1
        else:
            self.safe_streak = 0
        telemetry = {"skill": "cover_retreat", "skill_step": self.elapsed_steps,
                     "skill_start_step": self.skill_start_step,
                     "risks": result["risks"],
                     "retreat_units": result["retreat_units"],
                     "cover_units": result["cover_units"],
                     "other_units": result["other_units"],
                     "retreat_actions": result["retreat_actions"],
                     "retreat_direction": self.retreat_direction,
                     "threat_priority": result["threat_priority"],
                     "cover_assignments": result["cover_assignments"],
                     "cover_predicted_damage": result["cover_budget"],
                     "cover_estimated_overkill": result["cover_overkill"],
                     "safe_streak": self.safe_streak,
                     "safe_streak_required": self.safe_streak_required}
        if result["terminated"] and result["termination_reason"] != "no_retreat_needed":
            self._terminate(result["termination_reason"])
        elif self.safe_streak >= self.safe_streak_required:
            self.stats["safe_streak_exits"] += 1
            self._terminate("safe_streak")
        elif self.elapsed_steps >= self.max_steps:
            self._terminate("max_steps")
        if self.termination_reason is not None:
            telemetry["termination_reason"] = self.termination_reason
        return result["actions"], telemetry

    def _step_heal_assign(self, payload, n_agents):
        """One HealAssign step. Returns (partial_actions, telemetry).

        ``partial_actions`` maps actor index -> action for healer slots only;
        the caller merges a declared base policy for every other unit, so no
        unit ever receives two action sources in one step.
        """
        state = payload["state"]
        result = assign_healing(state, n_agents,
                                previous_assignments=self.target_assignments or None,
                                parameters=self.heal_params)
        self.last_observation_step = state["game_loop"]
        self.skill_start_step = self.skill_start_step if self.skill_start_step is not None \
            else state["game_loop"]
        self.elapsed_steps += 1
        self.stats["steps"] += 1
        if not result["available"]:
            self.stats["heal_no_target_steps"] += 1
            telemetry = {"skill": "heal_assign", "skill_step": self.elapsed_steps,
                         "skill_start_step": self.skill_start_step,
                         "available": False}
            self._terminate(result["termination_reason"])
            telemetry["termination_reason"] = self.termination_reason
            return {}, telemetry
        switches = sum(1 for healer, previous in (self.target_assignments or {}).items()
                       if healer in result["assignments"] and result["assignments"][healer] != previous)
        self.stats["heal_assignments_created"] += len(result["assignments"])
        self.stats["heal_target_switches"] += switches
        self.stats["heal_self_preserves"] += len(result["self_preserving"])
        if not result["assignments"] and not result["self_preserving"]:
            self.stats["heal_no_target_steps"] += 1
        partial = {int(actor[1:]): action for actor, action in result["healer_actions"].items()
                   if action is not None}
        self.assigned_units = sorted(result["healer_actions"], key=lambda a: int(a[1:]))
        self.target_assignments = result["assignments"]
        telemetry = {"skill": "heal_assign", "skill_step": self.elapsed_steps,
                     "skill_start_step": self.skill_start_step,
                     "available": True, "healers": result["healers"],
                     "wounded_targets": result["wounded_targets"],
                     "assignments": result["assignments"],
                     "partial_actions": partial,
                     "urgency": result["urgency"],
                     "missing_health": result["missing_health"],
                     "healer_energy": {h: state["allies"][h].get("energy") for h in result["healers"]},
                     "retained_orders": result["retained_orders"],
                     "released": result["released"],
                     "self_preserving": result["self_preserving"],
                     "unassigned_healers": result["unassigned_healers"],
                     "target_switches": switches}
        if result["terminated"]:
            self._terminate(result["termination_reason"])
        elif self.elapsed_steps >= self.max_steps:
            self._terminate("max_steps")
        if self.termination_reason is not None:
            telemetry["termination_reason"] = self.termination_reason
        return partial, telemetry

    def _terminate(self, reason):
        self.termination_reason = reason
        self.stats["terminations"][reason] = self.stats["terminations"].get(reason, 0) + 1

    def spend_mode_step(self):
        """Consume one decision step of the current mode window."""
        if self.mode_remaining > 0:
            self.mode_remaining -= 1

    def record_fallback(self):
        self.stats["fallback_steps"] += 1
