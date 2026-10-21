"""Skill lifecycle: activation, multi-step execution, termination, telemetry."""

from .skill_policy import (ADAPTIVE_FOCUS_FIRE, COVER_RETREAT, DIRECT_CONTROL,
                           FOCUS_FIRE_BUDGET, HEAL_ASSIGN)
from .skills.focus_fire import allocate_focus_fire
from .skills.cover_retreat import allocate_cover_retreat, PARAMS as COVER_PARAMS
from .skills.heal_assign import assign_healing, PARAMS as HEAL_PARAMS
from .skills.adaptive_focus_fire import (PARAMS as ADAPTIVE_PARAMS,
                                         allocate_adaptive_focus_fire)


class SkillManager:
    """Tracks the currently selected control mode and one active skill run.

    One manager instance spans the whole experiment; ``reset`` must be called
    at every episode start so no state leaks across episodes.
    """

    def __init__(self, max_steps=4, risk_threshold=None, safe_streak_required=2,
                 heal_switch_margin=None, adaptive_params=None, combat_params=None):
        self.max_steps = max_steps
        self.safe_streak_required = max(1, safe_streak_required)
        self.cover_params = dict(COVER_PARAMS)
        if risk_threshold is not None:
            self.cover_params["risk_threshold"] = risk_threshold
        self.heal_params = dict(HEAL_PARAMS)
        if heal_switch_margin is not None:
            self.heal_params["switch_margin"] = heal_switch_margin
        self.adaptive_params = dict(ADAPTIVE_PARAMS)
        if adaptive_params:
            self.adaptive_params.update(adaptive_params)
        # Shared Sharpy-inspired combat mechanisms (window fire, move scoring,
        # target scoring): one source of truth for every skill.
        self.combat_params = dict(combat_params or {})
        self.stats = {"activations": 0, "steps": 0, "terminations": {},
                      "reassignments": 0, "estimated_overkill": 0.0,
                      "predicted_damage": 0.0, "selections": {},
                      "fallback_steps": 0, "role_reassignments": 0,
                      "safe_streak_exits": 0, "retreat_unit_steps": 0,
                      "heal_assignments_created": 0, "heal_target_switches": 0,
                      "heal_self_preserves": 0, "heal_no_target_steps": 0,
                      "adaptive_engage_steps": 0, "adaptive_retreat_steps": 0,
                      "adaptive_all_in_steps": 0, "adaptive_retreat_unit_steps": 0,
                      "adaptive_reengagements": 0, "adaptive_all_in_triggers": 0}
        self.stats.update(immediate_damage=0.0, window_only_damage=0.0,
                          movement_evaluations=0, kept_attack_positions=0,
                          adaptive_recovery_unit_steps=0)
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
        # AdaptiveFocusFire episode-scoped history: survives 4-step skill
        # windows so re-activation cannot reset retreat caps or All-in.
        self.posture = "balanced"
        self.unit_states = {}
        self.all_in = False
        self.all_in_reason = None
        self.stall_steps = 0
        self.contacted = False
        self._last_enemy_health = None
        self._last_move = {}  # per-unit previous movement direction (anti ping-pong)
        self._last_seen_enemies = {}

    # -- selection -------------------------------------------------------

    def select(self, skill, priority_target=None, posture=None):
        """JEV (or --skill-mode always) chose a mode; hold it for one window."""
        if skill not in (DIRECT_CONTROL, FOCUS_FIRE_BUDGET, COVER_RETREAT,
                         HEAL_ASSIGN, ADAPTIVE_FOCUS_FIRE):
            raise ValueError(f"Unknown skill: {skill}")
        self.current_skill = skill
        self.priority_target = priority_target
        if posture:
            self.posture = posture
        self.mode_remaining = self.max_steps
        self.stats["selections"][skill] = self.stats["selections"].get(skill, 0) + 1
        if skill != DIRECT_CONTROL:
            self._start_skill()

    def _start_skill(self):
        self.skill_start_step = None
        self.elapsed_steps = 0
        self.assigned_units = []
        if self.current_skill != ADAPTIVE_FOCUS_FIRE:
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
        return (self.current_skill in (FOCUS_FIRE_BUDGET, COVER_RETREAT,
                                       HEAL_ASSIGN, ADAPTIVE_FOCUS_FIRE)
                and self.termination_reason is None)

    # -- execution -------------------------------------------------------

    def _record_diagnostics(self, result, telemetry):
        """Preserve allocator diagnostics in the actual skill_step event."""
        for key in ("immediate_damage", "window_only_damage", "movement_diagnostics"):
            telemetry[key] = result.get(key, {})
        for key, parameter in (("window_fire_enabled", "window_fire"),
                               ("target_scoring_enabled", "target_scoring"),
                               ("range_preserving_move_enabled", "range_preserving_move")):
            telemetry[key] = result.get(key, bool(self.combat_params.get(parameter)))
        for key in ("immediate_damage", "window_only_damage"):
            self.stats[key] += sum(telemetry[key].values())
        self.stats["movement_evaluations"] += len(telemetry["movement_diagnostics"])
        self.stats["kept_attack_positions"] += sum(
            row.get("reason") == "keep_attack_position"
            for row in telemetry["movement_diagnostics"].values())

    def step(self, payload, n_agents):
        """Run one local decision step of the active skill on a fresh snapshot.

        Returns (actions, telemetry). ``actions`` is always a full
        n_agents-length list of legal snapshot actions.
        """
        if self.current_skill == COVER_RETREAT:
            return self._step_cover_retreat(payload, n_agents)
        if self.current_skill == HEAL_ASSIGN:
            return self._step_heal_assign(payload, n_agents)
        if self.current_skill == ADAPTIVE_FOCUS_FIRE:
            return self._step_adaptive(payload, n_agents)
        return self._step_focus_fire(payload, n_agents)

    def _step_focus_fire(self, payload, n_agents):
        state = payload["state"]
        result = allocate_focus_fire(state, n_agents, priority_target=self.priority_target,
                                     previous_assignments=self.target_assignments or None,
                                     parameters=self.combat_params)
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
        # Cross-step pending-damage ledgers are disabled: observations cannot
        # reliably separate landed from in-flight attacks, and subtracting a
        # stale ledger double-counted damage already visible in health.
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
        self._record_diagnostics(result, telemetry)
        if result["terminated"]:
            self._terminate(result["termination_reason"])
        elif self.elapsed_steps >= self.max_steps:
            self._terminate("max_steps")
        if self.termination_reason is not None:
            telemetry["termination_reason"] = self.termination_reason
        return result["actions"], telemetry

    def _step_cover_retreat(self, payload, n_agents):
        state = payload["state"]
        # Per-unit hysteresis: the previous step's retreating set decides who
        # is judged at the tightened continue threshold; first-step and newly
        # endangered units always use the full entry threshold.
        result = allocate_cover_retreat(state, n_agents,
                                        parameters={**self.cover_params, **self.combat_params},
                                        previous_assignments=self.target_assignments or None,
                                        retreating_units=self.retreat_units,
                                        move_history=self._last_move)
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
        for actor in self.retreat_units:
            action = result["actions"][int(actor[1:])]
            if action in (2, 3, 4, 5):
                self._last_move[actor] = action
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
        self._record_diagnostics(result, telemetry)
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
        if not result["assignments"] and not result["self_preserving"] \
                and not result.get("approach_targets"):
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

    def _step_adaptive(self, payload, n_agents):
        """One AdaptiveFocusFire step with persistent per-unit role history."""
        state = payload["state"]
        # Observable combat progress, fog-safe: compare per-enemy health only
        # over enemies visible in BOTH steps, so an enemy entering or leaving
        # the fog is never misread as damage dealt or undone.
        current_health = {enemy_id: e["health"] + e["shield"]
                          for enemy_id, e in state["visible_enemies"].items()}
        overlap = set(current_health) & set(self._last_enemy_health or {})
        damage_observed = any(current_health[e] < self._last_enemy_health[e] - 1e-6
                              for e in overlap)
        if damage_observed:
            self.stall_steps = 0
            self.contacted = True  # Observable damage proves contact.
            # Damage describes the interval that just finished, before this
            # step's role selection. Delayed hits during retreat/recovery count.
            # This is team-observed progress, not per-unit damage attribution.
            for history in self.unit_states.values():
                history["progress_since_retreat"] = True
                history["retreat_cycles"] = 0
        elif self.contacted and state["visible_enemies"] and overlap:
            self.stall_steps += 1
        if not self.contacted and state["visible_enemies"] and any(
                relation.get("smac_attack_available")
                for row in state.get("relations", {}).values()
                for relation in row.values()):
            self.contacted = True  # Visible enemy plus a legal attack edge = engagement.
        self._last_enemy_health = current_health
        loop = state["game_loop"]
        for enemy, unit in state["visible_enemies"].items():
            self._last_seen_enemies[enemy] = {"position": list(unit["position"]), "game_loop": loop}
        ttl = self.adaptive_params["search_memory_steps"] * state.get("step_mul", 8)
        self._last_seen_enemies = {
            e: record for e, record in self._last_seen_enemies.items()
            if 0 <= loop - record["game_loop"] <= ttl}
        search_hint = None
        if not state["visible_enemies"] and self._last_seen_enemies:
            enemy, record = max(self._last_seen_enemies.items(),
                                key=lambda item: (item[1]["game_loop"], -int(item[0][1:])))
            search_hint = {**record, "enemy": enemy, "age_loops": loop - record["game_loop"]}
            state = {**state, "search_hint": search_hint}

        result = allocate_adaptive_focus_fire(
            state, n_agents, unit_states=self.unit_states, all_in=self.all_in,
            stall=self.stall_steps, contacted=self.contacted, posture=self.posture,
            parameters={**self.adaptive_params, **self.combat_params},
            previous_attack_assignments=self.target_assignments or None,
            move_history=self._last_move)
        for actor in result["roles"]:
            action = result["actions"][int(actor[1:])]
            if action in (2, 3, 4, 5):
                self._last_move[actor] = action
        all_in_started_now = result["all_in"] and not self.all_in
        previous_roles = {a: v.get("role") for a, v in self.unit_states.items()}
        retreat_ended = sorted((a for a, role in result["roles"].items()
                                if role != "retreat" and previous_roles.get(a) == "retreat"),
                               key=lambda a: int(a[1:]))
        self.last_observation_step = state["game_loop"]
        self.skill_start_step = self.skill_start_step if self.skill_start_step is not None \
            else state["game_loop"]
        self.elapsed_steps += 1
        self.stats["steps"] += 1
        if result["all_in"] and not self.all_in:
            self.stats["adaptive_all_in_triggers"] += 1
        self.all_in, self.all_in_reason = result["all_in"], result["all_in_reason"]
        self.stats["adaptive_all_in_steps" if self.all_in else
                   ("adaptive_retreat_steps" if result["retreat_units"]
                    else "adaptive_engage_steps")] += 1
        self.stats["adaptive_retreat_unit_steps"] += len(result["retreat_units"])
        self.stats["adaptive_recovery_unit_steps"] += len(result["reentry_decisions"])
        self.stats["adaptive_reengagements"] += len(result["reengaged"])
        if isinstance(result.get("predicted_damage"), dict):
            self.stats["predicted_damage"] += sum(result["predicted_damage"].values())
        if isinstance(result.get("estimated_overkill"), dict):
            self.stats["estimated_overkill"] += sum(result["estimated_overkill"].values())

        # Per-unit role history. A retreat episode only counts toward the
        # unproductive-cycle tally when the unit's intervening attack stint
        # produced no observable enemy damage: normal dodge -> re-engage ->
        # damage cycles are free, repeated fruitless retreats are not.
        for actor, role in result["roles"].items():
            unit_state = self.unit_states.setdefault(
                actor, {"role": "attack", "retreat_steps": 0, "retreat_cycles": 0})
            if role == "retreat":
                if unit_state.get("role") != "retreat":
                    if not unit_state.pop("progress_since_retreat", False):
                        unit_state["retreat_cycles"] += 1
                    unit_state["retreat_steps"] = 1
                else:
                    unit_state["retreat_steps"] += 1
            elif role == "recover":
                unit_state["recover_steps"] = unit_state.get("recover_steps", 0) + 1
            else:
                if unit_state.get("role") == "retreat":
                    if not self.combat_params.get("range_preserving_move"):
                        unit_state["retreat_steps"] = 0
                if role in ("attack", "all_in_attack"):
                    unit_state["retreat_steps"] = 0
                    unit_state["recover_steps"] = 0
            unit_state["role"] = role
        self.unit_states = {a: v for a, v in self.unit_states.items()
                            if a in state["allies"] and state["allies"][a]["alive"]}
        self.target_assignments = result["attack_assignments"]

        telemetry = {"skill": "adaptive_focus_fire", "skill_step": self.elapsed_steps,
                     "skill_start_step": self.skill_start_step,
                     "posture": self.posture, "phase": result["phase"],
                     "all_in": self.all_in, "all_in_reason": self.all_in_reason,
                     "attack_units": result["attack_units"],
                     "retreat_units": result["retreat_units"],
                     "support_units": result["support_units"],
                     "roles": result["roles"],
                     "retreat_reasons": result["retreat_reasons"],
                     "retreat_actions": result["retreat_actions"],
                     "reengaged": result["reengaged"],
                     "attack_assignments": result["attack_assignments"],
                     "predicted_damage": result["predicted_damage"],
                     "estimated_overkill": result["estimated_overkill"],
                     "stall_steps": self.stall_steps,
                     "all_in_started_now": result["all_in_reason"] if all_in_started_now else None,
                     "retreat_ended": retreat_ended,
                     "retreat_state": {a: dict(self.unit_states.get(a, {}))
                                       for a in result["retreat_units"]},
                     "reentry_decisions": result["reentry_decisions"],
                     "search_hint": search_hint,
                     "observed_team_damage_progress": damage_observed,
                     "progress_basis": "visible enemy health decrease in the preceding interval"}
        self._record_diagnostics(result, telemetry)
        if result["terminated"]:
            self._terminate(result["termination_reason"])
        elif self.elapsed_steps >= self.max_steps:
            self._terminate("max_steps")
        if self.termination_reason is not None:
            telemetry["termination_reason"] = self.termination_reason
        return result["actions"], telemetry

    def _terminate(self, reason):
        self.termination_reason = reason
        # A genuinely terminated skill must hand control back at the next
        # decision boundary instead of keeping a stale mode window.
        if self.current_skill != DIRECT_CONTROL:
            self.mode_remaining = 0
        self.stats["terminations"][reason] = self.stats["terminations"].get(reason, 0) + 1

    def spend_mode_step(self):
        """Consume one decision step of the current mode window."""
        if self.mode_remaining > 0:
            self.mode_remaining -= 1

    def record_fallback(self):
        self.stats["fallback_steps"] += 1
