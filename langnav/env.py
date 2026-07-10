import gymnasium
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path

Assets_dir = Path(__file__).parent.parent / "assets"


class HuskyTerrainEnv(gymnasium.Env):
    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode
        print(" i love rahma apiiiiiii")

        self.observation_space = spaces.Box(low = -np.inf,
                                            high = np.inf,
                                            shape=(10,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low = -1.0,
                                      high = 1.0,
                                      shape = (4,),
                                       dtype=np.float32)

        self.client = None
        self.husky_id = None
        self.terrain_id = None
        self.step_count = 0

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)

        if self.client is None:
            mode = None
            if self.render_mode == "human":
                mode = p.GUI
            else:
                mode = p.DIRECT
            self.client = p.connect(mode)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.resetSimulation(physicsClientId=self.client)
            p.setGravity(0, 0, -9.8,
                         physicsClientId=self.client)

            self.terrain_id = p.loadURDF("plane.urdf",
                                         physicsClientId=self.client)

            self.husky_id = p.loadURDF("r2d2.urdf",
                                       basePosition=[0, 0, 1.0],
                                       physicsClientId=self.client)

            self.step_count = 0
            obs = np.zeros(self.observation_space.shape, dtype = np.float32)
            info ={}
            return obs, info

    def step(self, action):
        p.stepSimulation(physicsClientId=self.client)
        self.step_count += 1
        obs = np.zeros(self.observation_space.shape, dtype = np.float32)
        reward = 0.0
        terminate = False
        truncate = self.step_count >= 1000
        info = {}
        return obs, reward, terminate, truncate, info
    def render(self):
        if self.render_mode == "human":
            return None

        return None

    def close(self):
        if self.client is not None:
            p.disconnect(physicsClientId=self.client)
            self.client = None
