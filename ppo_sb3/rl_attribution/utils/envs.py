import numpy as np
from gymnasium import ObservationWrapper, spaces
from gymnasium import ActionWrapper

import gymnasium as gym
import minigrid
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.wrappers import ResizeObservation
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation

from minigrid.wrappers import ImgObsWrapper, FlatObsWrapper
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecTransposeImage, VecNormalize

from gymnasium.wrappers import FrameStack, TransformObservation

class RestrictedActionSpaceWrapper(gym.Wrapper):
    def __init__(self, env, n_actions, action_mapping=None):
        super().__init__(env)

        self.action_space = gym.spaces.Discrete(n_actions)

        if action_mapping is not None:
            self.action_mapping = action_mapping
        else:
            self.action_mapping = {i:i for i in range(n_actions)}

    def step(self, action):

        mapped_action = self.action_mapping[action]

        obs, reward, terminated, truncated, info = self.env.step(mapped_action)
        return obs, reward, terminated, truncated, info

class SB3CompatWrapper(gym.Wrapper):

    def get_wrapper_attr(self, name: str):
        env = self
        while isinstance(env, gym.Wrapper):
            if hasattr(env, name):
                return getattr(env, name)
            env = env.env
        if hasattr(env, name):
            return getattr(env, name)
        raise AttributeError(f"{name} not found in wrapper stack")

class DropLastChannel(gym.ObservationWrapper):
    def __init__(self, env, n_stack):
        super().__init__(env)
        shp = self.observation_space.shape
        assert len(shp) == 4 and shp[0] == n_stack and shp[-1] == 1, \
            f"Expected ({n_stack}, H, W, 1), got {shp}"

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=shp[:-1],
            dtype=self.observation_space.dtype,
        )

    def observation(self, obs):
        return obs[..., 0]

class ContinuousActionDTypeWrapper(ActionWrapper):
    """Ensure continuous actions match the environment action-space dtype."""

    def __init__(self, env):
        super().__init__(env)
        self._is_box = isinstance(self.action_space, spaces.Box)
        self._action_dtype = self.action_space.dtype if self._is_box else None

    def action(self, action):
        if not self._is_box:
            return action
        arr = np.asarray(action, dtype=self._action_dtype).reshape(self.action_space.shape)

        return arr.tolist()

class TerminalObservationArrayWrapper(gym.Wrapper):
    """Ensure terminal_observation is an ndarray for SB3 obs_to_tensor."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        terminal_obs = info.get("terminal_observation")
        if terminal_obs is not None:
            info["terminal_observation"] = np.asarray(terminal_obs)
        return obs, reward, terminated, truncated, info

class NumpyObservationWrapper(ObservationWrapper):
    """Convert observations (e.g., LazyFrames) to NumPy arrays."""

    def observation(self, observation):
        return np.asarray(observation)

def make_env(env_name, env_cfg, render_mode=None):
    def _init():
        cfg = {} if env_cfg is None else dict(env_cfg)

        base_env_name, act_n = env_name, None
        if "-act" in env_name:
            base_env_name, act_n = env_name.rsplit("-act", 1)
            act_n = int(act_n)

        if base_env_name == "FrozenLake-v1":
            env = gym.make(base_env_name, render_mode=render_mode, map_name="4x4", is_slippery=False)

        elif base_env_name == "highway-v0":
            import highway_env
            env = gym.make(
                base_env_name,
                render_mode=render_mode,
                config={
                    "observation": {"type": "Kinematics"},
                    "action": {"type": "DiscreteMetaAction"},

                    "lanes_count": 3,
                    "vehicles_count": 30,
                },
            )

        elif "merge" in base_env_name:
            import highway_env
            env = gym.make(base_env_name, render_mode=render_mode)

        elif base_env_name.startswith("MiniGrid"):
            env = gym.make(base_env_name, render_mode=render_mode)
            if base_env_name.startswith("MiniGrid-Empty"):
                env = ImgObsWrapper(env)
            else:
                env = FlatObsWrapper(env)

            if act_n is not None:
                env = RestrictedActionSpaceWrapper(env, act_n)

        elif "Pong" in base_env_name:
            env = gym.make(base_env_name, render_mode=render_mode)
            env = AtariWrapper(env)
            env = FrameStack(env, 4)
            env = DropLastChannel(env, 4)

            if act_n is not None:
                if act_n == 3:
                    action_mapping = {0: 0, 1: 2, 2: 3}
                elif act_n == 2:
                    action_mapping = {0: 2, 1: 3}
                else:
                    action_mapping = None

                env = RestrictedActionSpaceWrapper(
                    env,
                    n_actions=act_n,
                    action_mapping=action_mapping,
                )

        elif base_env_name.startswith("CarRacing"):
            env = gym.make(base_env_name, render_mode=render_mode, continuous=True)
            env = ContinuousActionDTypeWrapper(env)
            env = ResizeObservation(env, (32, 32))
            env = GrayScaleObservation(env)
            env = FrameStack(env, 2)
            env = NumpyObservationWrapper(env)

        else:
            env = gym.make(base_env_name, render_mode=render_mode)

        env = RecordEpisodeStatistics(env)
        env = TerminalObservationArrayWrapper(env)

        env = SB3CompatWrapper(env)
        return env

    return _init

def make_vec_env(env_name: str, n_envs: int, render_mode: str = None):
    """
    Build a vectorized env using your make_env factory.
    Uses SubprocVecEnv for heavier envs; DummyVecEnv otherwise.
    """

    n_envs = max(1, int(n_envs))
    env_fns = [lambda: make_env(env_name, None, render_mode=render_mode)() for _ in range(n_envs)]

    vec_env = DummyVecEnv(env_fns) if n_envs == 1 else SubprocVecEnv(env_fns)

    vec_env = VecMonitor(vec_env)
    return vec_env

class CloneableFrozenLake(gym.Wrapper):
    """
    Adds get_state()/set_state() to FrozenLake-v1.

    What we snapshot:
      - env.unwrapped.s          : current discrete cell index (int)
      - env.unwrapped.np_random  : RNG state (so slipping etc. is reproducible)

    Notes:
      - We do NOT regenerate the transition table; we assume the map (desc) and
        is_slippery flag are fixed for the lifetime of the env. That matches
        standard usage where desc/is_slippery are chosen at make(...).
    """
    def get_state(self):
        u = self.unwrapped
        rng_state = None

        if hasattr(u, "np_random") and hasattr(u.np_random, "bit_generator"):
            rng_state = u.np_random.bit_generator.state
        return {
            "s": int(getattr(u, "s", 0)),
            "rng": rng_state,
        }

    def set_state(self, snap):
        u = self.unwrapped
        u.s = int(snap["s"])
        if snap.get("rng") is not None:
            u.np_random.bit_generator.state = snap["rng"]

class CloneableAcrobot(gym.Wrapper):
    """
    Acrobot-v1 stores full internal dynamics in `unwrapped.state`
    as [cos(theta1), sin(theta1), cos(theta2), sin(theta2), dtheta1, dtheta2].
    We also persist/reset the RNG state for determinism across repeats.
    """
    def get_state(self):
        u = self.unwrapped
        state_arr = np.array(u.state, dtype=np.float64).copy()
        rng_state = None
        if hasattr(u, "np_random") and hasattr(u.np_random, "bit_generator"):
            rng_state = u.np_random.bit_generator.state
        return {"state": state_arr, "rng": rng_state}

    def set_state(self, snap):
        u = self.unwrapped
        u.state = np.array(snap["state"], dtype=np.float64).copy()
        if snap["rng"] is not None:
            u.np_random.bit_generator.state = snap["rng"]

class CloneableLunarLander(gym.Wrapper):
    """
    Snapshot/restore Box2D world: all bodies' (pos, vel, angle, angvel).
    Also stores RNG state so particles/thrusters reproduce deterministically.
    Works for LunarLander-v2 and LunarLanderContinuous-v2.
    """
    def get_state(self):
        u = self.unwrapped
        world = u.world
        bodies = []
        for b in world.bodies:
            bodies.append({
                "position": np.array(b.position, dtype=np.float64).copy(),
                "linearVelocity": np.array(b.linearVelocity, dtype=np.float64).copy(),
                "angle": float(b.angle),
                "angularVelocity": float(b.angularVelocity),
            })
        rng_state = None
        if hasattr(u, "np_random") and hasattr(u.np_random, "bit_generator"):
            rng_state = u.np_random.bit_generator.state

        leg_contacts = []
        if hasattr(u, "legs"):
            for leg in u.legs:
                leg_contacts.append(bool(getattr(leg, "ground_contact", False)))
        return {"bodies": bodies, "rng": rng_state, "leg_contacts": leg_contacts}

    def set_state(self, snap):
        u = self.unwrapped
        world = u.world

        for b, sb in zip(world.bodies, snap["bodies"]):
            b.position = sb["position"]
            b.linearVelocity = sb["linearVelocity"]
            b.angle = sb["angle"]
            b.angularVelocity = sb["angularVelocity"]

        if snap["rng"] is not None:
            u.np_random.bit_generator.state = snap["rng"]

        world.Step(0.0, 6, 2) if hasattr(world, "Step") else world.step(0.0)

        if hasattr(u, "legs") and snap.get("leg_contacts") is not None:
            for leg, flag in zip(u.legs, snap["leg_contacts"]):
                setattr(leg, "ground_contact", bool(flag))
