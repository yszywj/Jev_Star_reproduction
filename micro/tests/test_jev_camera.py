"""Camera-follow regression tests: visualization must never break mock or real envs."""

import unittest
from types import SimpleNamespace

import test_jev_micro as fixtures
from jev_micro.environment import JevMicroEnv, StarCraft2Env


class CameraSafetyTests(unittest.TestCase):
    def test_step_survives_without_controllers_attribute(self):
        # CI regression: mock environments lack _controllers entirely.
        env = fixtures.ExtendedMapTests().environment()
        self.assertFalse(hasattr(env, "_controllers"))

        def step(_env, choices):
            _env.last_action_results = []
            return 0, False, {}

        with unittest.mock.patch.object(StarCraft2Env, "step", step):
            result = env.step([6, 0, 0])  # Must not raise AttributeError.
        self.assertEqual(result[1], False)

    def test_camera_moves_with_fake_controllers(self):
        from s2clientprotocol import sc2api_pb2 as sc_pb
        env = fixtures.ExtendedMapTests().environment()
        requests = []

        class FakeController:
            def actions(self, request):
                requests.append(request)

        env._controllers = [FakeController(), FakeController()]
        env._center_recording_camera()
        self.assertEqual(len(requests), 2)  # Both participant cameras move.
        for request in requests:
            action = request.actions[0].action_raw.camera_move.center_world_space
            self.assertEqual(action.z, 0)
            # Camera centers on the geometric middle of the living units.
            self.assertTrue(9 <= action.x <= 15)

    def test_camera_failure_never_breaks_control(self):
        env = fixtures.ExtendedMapTests().environment()

        class BrokenController:
            def actions(self, request):
                raise RuntimeError("controller gone")

        env._controllers = [BrokenController()]
        env._center_recording_camera()  # Swallowed: visualization only.


import unittest.mock  # noqa: E402  (kept close to usage for readability)

if __name__ == "__main__":
    unittest.main()
