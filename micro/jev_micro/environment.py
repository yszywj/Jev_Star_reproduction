"""Raw micro adapter fixes, isolated from SMAC's unused RL feature tensors."""

from s2clientprotocol import data_pb2
from s2clientprotocol import sc2api_pb2 as sc_pb

from smac_hard.env import StarCraft2Env
from smac_hard.env.scripts import SCRIPT_DICT
from smac_hard.env.scripts.base_script import DecisionTreeScript as BaseScript
from smac_hard.env.scripts.attack_nearest import DecisionTreeScript as NearestScript
from smac_hard.env.scripts.attack_weakest import DecisionTreeScript as WeakestScript
from smac_hard.env.scripts.utils.actions_api import attack


def is_medivac(unit, data):
    return (
        unit.unit_type == 54
        or data.units.get(unit.unit_type, "").lower().startswith("medivac")
    )


def is_biological(unit, data):
    stat = data.unit_stats.get(unit.unit_type)
    return stat is not None and data_pb2.Biological in stat.attributes


class ExtendedBaseScript(BaseScript):
    """Explicit attack-move counterpart for maps absent from the V2 script pool."""

    def __init__(self, map_name, destination):
        super().__init__(map_name)
        self.destination = destination

    def script(
        self,
        agents,
        enemies,
        agent_ability,
        visible_matrix,
        iteration,
    ):
        if iteration < 5:
            return []

        return [
            attack(unit, self.destination, visible_matrix)
            for unit in agents.values()
            if unit.health > 0
        ]


def opponent_for(env, name):
    if name == "mixed":
        if env.map_name not in SCRIPT_DICT:
            raise ValueError("This map has no upstream mixed opponent pool.")
        return None

    if name in {"nearest", "weakest"}:
        return {
            "nearest": NearestScript,
            "weakest": WeakestScript,
        }[name](env.map_name)

    if env.map_name in SCRIPT_DICT:
        return BaseScript(env.map_name)

    # Explicit attack-move extensions; two destinations come from the v1 base script.
    destinations = {
        "2m_vs_1z": (9, 16),
        "2s_vs_1sc": (9, 16),
        "so_many_baneling": (4.5, 4.5),
        "bane_vs_bane": (16, 8),
        "unit_test": (6, 16),
        "pvt_large": (9, 31),
    }

    return ExtendedBaseScript(
        env.map_name,
        destinations[env.map_name],
    )


class JevMicroEnv(StarCraft2Env):

    def __init__(self, **kwargs):
        self.micro_data = None
        super().__init__(**kwargs)

    def _launch(self):
        self.micro_data = None
        self.last_submissions = []
        return super()._launch()

    # ------------------------------------------------------------------
    # Recording / visualization camera
    # ------------------------------------------------------------------

    def _center_recording_camera(self):
        """
        Visualization only.

        Keep the visible StarCraft II camera centered on the current battle.

        This does NOT change:
        - Jev observations
        - Jev actions
        - Astra plans
        - unit commands
        - rewards
        - fixed-step timing

        It only changes the visible camera position in the SC2 windows.
        """

        if not self._controllers:
            return

        # All currently alive units on both teams.
        alive_units = [
            unit
            for unit in (
                list(self.agents.values())
                + list(self.enemies.values())
            )
            if unit.health > 0
        ]

        if not alive_units:
            return

        # Geometric center of all surviving units.
        center_x = sum(
            unit.pos.x for unit in alive_units
        ) / len(alive_units)

        center_y = sum(
            unit.pos.y for unit in alive_units
        ) / len(alive_units)

        # Build a raw SC2 camera movement action.
        action = sc_pb.Action()

        camera = action.action_raw.camera_move.center_world_space
        camera.x = center_x
        camera.y = center_y
        camera.z = 0

        request = sc_pb.RequestAction(
            actions=[action]
        )

        # Micro launches two participant processes.
        # Move both cameras so either visible window stays on the fight.
        for controller in self._controllers:
            try:
                controller.actions(request)
            except Exception:
                # Camera movement is visualization-only.
                # It must never affect the actual experiment.
                pass

    def reset(self, *args, **kwargs):
        """
        Reset the episode, then immediately center the camera
        on the initial battlefield.
        """

        result = super().reset(
            *args,
            **kwargs,
        )

        self._center_recording_camera()

        return result

    def step(self, actions):
        """
        Execute one fixed-step micro decision, then move the visible
        camera to the current center of the battle.
        """

        self.last_submissions = []
        self._collecting_actions = True

        try:
            result = super().step(actions)
        finally:
            self._collecting_actions = False

        # Visualization only.
        self._center_recording_camera()

        return result

    # ------------------------------------------------------------------
    # Original JEV-Star Micro behavior
    # ------------------------------------------------------------------

    def _init_ally_unit_types(self, min_unit_type):
        super()._init_ally_unit_types(min_unit_type)

        if self.micro_data is None:
            self.micro_data = self._controllers[0].data()

        roles = (
            "marine",
            "marauder",
            "medivac",
            "stalker",
            "zealot",
            "colossus",
            "hydralisk",
            "zergling",
            "baneling",
            "queen",
        )

        # V2 uses native, nonconsecutive IDs
        # (Marine=48, Marauder=51, Medivac=54).
        for unit in self.agents.values():
            name = self.micro_data.units.get(
                unit.unit_type,
                "",
            ).lower()

            for role in roles:
                if name.startswith(role):
                    setattr(
                        self,
                        role + "_id",
                        unit.unit_type,
                    )
                    break

    def can_move(self, unit, direction):
        if unit.is_flying:
            from smac_hard.env.starcraft2.starcraft2 import Direction

            offset = {
                Direction.NORTH: (0, 1),
                Direction.SOUTH: (0, -1),
                Direction.EAST: (1, 0),
                Direction.WEST: (-1, 0),
            }[direction]

            return self.check_bounds(
                unit.pos.x + offset[0] * self._move_amount,
                unit.pos.y + offset[1] * self._move_amount,
            )

        return super().can_move(
            unit,
            direction,
        )

    def get_avail_agent_actions(self, agent_id):
        mask = super().get_avail_agent_actions(
            agent_id
        )

        unit = self.agents[agent_id]

        if unit.health <= 0 or self.micro_data is None:
            return mask

        if is_medivac(
            unit,
            self.micro_data,
        ):
            for index in range(self.padding):
                target = self.agents.get(index)

                mask[6 + index] = int(
                    bool(
                        mask[6 + index]
                        and target
                        and target.tag != unit.tag
                        and target.health > 0
                        and target.health < target.health_max
                        and not target.is_flying
                        and is_biological(
                            target,
                            self.micro_data,
                        )
                        and unit.energy > 0
                    )
                )

        else:
            stat = self.micro_data.unit_stats.get(
                unit.unit_type
            )

            for index, target in self.enemies.items():

                # Battlecruiser's attacks are implemented as effects
                # and omitted from SC2's weapon list
                # (also special-cased by python-sc2).
                if (
                    not mask[6 + index]
                    or stat is None
                    or unit.unit_type == 57
                ):
                    continue

                if not stat.weapons:
                    name = self.micro_data.units.get(
                        unit.unit_type,
                        "",
                    ).lower()

                    # The API also omits Void Ray beams and
                    # Baneling suicide attacks.
                    # An empty list does not imply these units
                    # are unarmed.
                    if (
                        unit.unit_type == 80
                        or name.startswith("voidray")
                    ):
                        continue

                    if (
                        name.startswith("baneling")
                        and not target.is_flying
                    ):
                        continue

                target_types = (
                    {2, 3}
                    if target.is_flying
                    else {1, 3}
                )

                # Colossi can also be hit by weapons targeting air.
                if target.unit_type == 4:
                    target_types.add(2)

                if not any(
                    weapon.type in target_types
                    for weapon in stat.weapons
                ):
                    mask[6 + index] = 0

        return mask

    def only_medivac_left(self, ally):
        if (
            self.map_type != "MMM"
            or self.micro_data is None
        ):
            return False

        units = (
            self.agents
            if ally
            else self.enemies
        )

        alive = [
            unit
            for unit in units.values()
            if unit.health > 0
        ]

        return (
            bool(alive)
            and all(
                is_medivac(
                    unit,
                    self.micro_data,
                )
                for unit in alive
            )
        )

    def get_agent_action(self, a_id, action):
        command, submission = self._micro_action(
            a_id,
            action,
        )

        if getattr(
            self,
            "_collecting_actions",
            False,
        ):
            self.last_submissions.append(
                {
                    "actor": f"u{a_id}",
                    "action_id": int(action),
                    "command": command,
                    "submission": submission,
                }
            )

        return command

    def _micro_action(self, a_id, action):
        unit = self.agents[a_id]

        if (
            6 <= action < 6 + self.padding
            and is_medivac(
                unit,
                self.micro_data,
            )
        ):
            if not self.get_avail_agent_actions(a_id)[action]:
                raise ValueError(
                    "Unavailable heal action"
                )

            target = self.agents[
                action - 6
            ]

            if any(
                order.ability_id == 386
                and order.target_unit_tag == target.tag
                for order in unit.orders
            ):
                # Continuing the selected heal needs no duplicate command;
                # SC2 otherwise returns AlreadyTargeted (206).
                return (
                    None,
                    "existing_heal_order_retained",
                )

        command = super().get_agent_action(
            a_id,
            action,
        )

        return (
            command,
            "submitted"
            if command is not None
            else "dead_unit_noop",
        )