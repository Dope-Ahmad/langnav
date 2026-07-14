import gymnasium
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path
import time

Assets_dir = Path(__file__).parent.parent / "assets"


class HuskyTerrainEnv(gymnasium.Env):
    MAX_WHEEL_VELOCITY = 10.0  # rad/s — tune this after seeing how fast it drives
    MAX_WHEEL_FORCE = 30.0  # N·m torque limit per wheel
    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode
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

    def _find_wheel_joints(self):
        wheel_names = ["front_left_wheel", "front_right_wheel",
                   "rear_left_wheel", "rear_right_wheel"]
        name_to_idx = {}
        for x in range(p.getNumJoints(self.husky_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.husky_id, x, physicsClientId=self.client)
            name_to_idx[info[1].decode("utf-8")] = info[0]
        return [name_to_idx[x] for x in wheel_names]
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
        p.setGravity(0, 0, -3.72,
                        physicsClientId=self.client)



        #Setting Terrain

        collision_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                              fileName=str(Assets_dir/"gale_crater_patch1.stl"),
                                              physicsClientId=self.client,
                                              meshScale=[1, 1, 1],
                                              flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        visual_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                        fileName=str(Assets_dir/"gale_crater_patch1.stl"),
                                        physicsClientId=self.client,
                                        meshScale=[1, 1, 1],)


        self.terrain_id = p.createMultiBody(baseMass=0,
                                            baseCollisionShapeIndex=collision_id,
                                            baseVisualShapeIndex=visual_id,
                                            basePosition=[0, 0, 0],
                                            physicsClientId=self.client)
        # Recolor the terrain to your chosen hex blue (#052cfa)
        p.changeVisualShape(
            objectUniqueId=self.terrain_id,
            linkIndex=-1,  # -1 always targets the main base/root of the body
            rgbaColor=[0.91, 0.49, 0.07, 1.0],  # [R, G, B, A] scaled between 0.0 and 1.0
            physicsClientId=self.client
        )


        #Setting Robot

            #Getting Z Coordinate -- Start

        ray_info = p.rayTest(rayFromPosition=[0, 0, 10],
                             rayToPosition=[0, 0, -10],
                             physicsClientId=self.client)[0]
        z_coordinate = ray_info[3][2]
            #Getting Z Coordiante -- End
        self.husky_id = p.loadURDF("husky/husky.urdf",
                                   basePosition=[0, 0, z_coordinate + 0.3],
                                   baseOrientation=p.getQuaternionFromEuler([0,0,0]),
                                   physicsClientId=self.client)

        # Configure a wide-angle overview camera shot
        p.resetDebugVisualizerCamera(
            cameraDistance=10.0,  # Distance in meters from the target point (higher = more zoomed out)
            cameraYaw=220.0,  # Horizontal rotation angle in degrees around the target (0 to 360)
            cameraPitch=-90.0,  # Vertical angle in degrees (negative looks downward)
            cameraTargetPosition=[0, 0, z_coordinate],  # The X, Y, Z point the camera centers its gaze on
            physicsClientId=self.client
        )

        p.changeDynamics(self.terrain_id, -1, lateralFriction=1.0, physicsClientId=self.client)

        self.step_count = 0
        obs = np.zeros(self.observation_space.shape, dtype = np.float32)
        info ={}
        self.wheel_joints = self._find_wheel_joints()
        return obs, info

    def step(self, action):

        action = np.clip(action, -1.0, 1.0)
        target_velocities = action * self.MAX_WHEEL_VELOCITY
        p.setJointMotorControlArray(bodyUniqueId=self.husky_id,
                                    jointIndices=self.wheel_joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=target_velocities.tolist(),
                                    forces=[self.MAX_WHEEL_FORCE]*4,
                                    physicsClientId=self.client)

        p.stepSimulation(physicsClientId=self.client)
        if self.render_mode == "human":
            time.sleep(1.0 / 240.0)
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
