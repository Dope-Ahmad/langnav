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
                                            shape=(37,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low = -1.0,
                                      high = 1.0,
                                      shape = (4,),
                                       dtype=np.float32)

        self.client = None
        self.husky_id = None
        self.terrain_id = None
        self.step_count = 0
        self.goal_position = None
        self.prev_action = np.zeros(4,dtype=np.float32)

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
        self.goal_position = self.np_random.uniform(low = -8.0, high = 8.0, size = 2)
        self.prev_action = np.zeros(4, dtype=np.float32)

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
        self.wheel_joints = self._find_wheel_joints()
        obs = self._get_obs()
        info ={}
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

        terminated, truncated, tipped, reached = self._check_termination()
        reward, reward_info = self._compute_reward(action, tipped, reached)


        self.prev_action = action
        self.step_count += 1
        obs = self._get_obs()
        info = dict(reward_info)
        info["tipped"] = tipped
        info["reached"] = reached
        return obs, reward, terminated, truncated, info
    def render(self):
        if self.render_mode == "human":
            return None

        return None

    def close(self):
        if self.client is not None:
            p.disconnect(physicsClientId=self.client)
            self.client = None

    def _get_wheel_contacts(self):
        contacts = []
        for joint in self.wheel_joints:
            pts = p.getContactPoints(bodyA=self.husky_id,
                                     linkIndexA=joint,
                                     physicsClientId=self.client
            )
            if len(pts) > 0:
                contacts.append(1.0)
            else:
                contacts.append(0.0)

        return np.array(contacts, dtype=np.float32)

    def _get_goal_vector(self, position, orientation):
        dx = self.goal_position[0] - position[0]
        dy = self.goal_position[1] - position[1]
        yaw = p.getEulerFromQuaternion(orientation)[2]
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        x_local = cos_y * dx + sin_y * dy
        y_local = -sin_y * dx + cos_y * dy
        goal_distance = float(np.sqrt(dx**2 + dy**2))
        return np.array([x_local, y_local],dtype=np.float32), goal_distance

    def _get_height_patch(self, position, orientation,  grid_size = 4, spacing = 0.5):

        yaw = p.getEulerFromQuaternion(orientation)[2]
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        offsets = np.linspace(-(grid_size - 1)/2 * spacing,
                              (grid_size - 1)/2 * spacing,
                              grid_size)
        ray_from, ray_to = [], []
        for x in offsets:
            for y in offsets:
                wx = position[0] + (cos_y * x - sin_y * y)
                wy = position[1] + (sin_y * x + cos_y * y)

                ray_from.append([wx,wy,position[2]+5.0])
                ray_to.append([wx,wy,position[2]-5.0])
        results = p.rayTestBatch(ray_from,ray_to,physicsClientId=self.client)

        heights = []
        for r in results:
            if r[0] != -1:
                heights.append(r[3][2] - position[2])
            else:
                heights.append(0.0)

        return np.array(heights, dtype=np.float32)

    def _get_obs(self):
        linear_velocity, angular_velocity = p.getBaseVelocity(self.husky_id,
                                                              physicsClientId=self.client)
        position, orientation = p.getBasePositionAndOrientation(self.husky_id,
                                                                physicsClientId=self.client)
        contacts = self._get_wheel_contacts()
        local_goal, goal_distance = self._get_goal_vector(position, orientation)
        height_patch = self._get_height_patch(position, orientation)

        obs = np.concatenate([np.array(linear_velocity, dtype=np.float32),
                              np.array(angular_velocity, dtype=np.float32),
                              np.array(orientation, dtype=np.float32),
                              contacts,
                              local_goal,
                              np.array([goal_distance], dtype=np.float32),
                              height_patch,
                              self.prev_action
                              ]).astype(np.float32)
        return obs

    def _check_termination(self):
        position, orientation = p.getBasePositionAndOrientation(self.husky_id, physicsClientId=self.client)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
        tipped: bool
        if (abs(roll) > np.radians(25.0)) or (abs(pitch) > np.radians(35.0)):
            tipped = True
        else:
            tipped = False

        dx = self.goal_position[0] - position[0]
        dy = self.goal_position[1] - position[1]
        distance = np.sqrt(dx**2 + dy**2)
        reached = bool(distance<0.5)

        terminated = reached or tipped
        truncated = self.step_count >= 1000
        return terminated, truncated, tipped, reached

    def _compute_reward(self, action, tipped, reached):
        linear_velocity, anuglar_velocity = p.getBaseVelocity(self.husky_id, physicsClientId=self.client)
        position, orientation = p.getBasePositionAndOrientation(self.husky_id, physicsClientId=self.client)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

        dx = self.goal_position[0] - position[0]
        dy = self.goal_position[1] - position[1]
        distance = np.sqrt(dx**2 + dy**2)
        goal_direction = np.array([dx,dy], dtype=np.float32) / (distance+ 1e-6)
        velocity_vector = np.array([linear_velocity[0], linear_velocity[1]], dtype=np.float32)
        progress = float(np.dot(velocity_vector, goal_direction))
        r_progress = 0.5 * progress

        r_survival = 0.01
        r_orientation = -0.3 * abs(roll) - 0.3 * abs(pitch)

        jerk = action - self.prev_action
        r_jerk = -0.05 * float(np.dot(jerk,jerk))

        r_goal = 2.0 if reached else 0.0
        r_tip = -5.0 if tipped else 0.0

        total = r_tip + r_goal + r_jerk + r_progress + r_survival + r_orientation
        details = {"r_progress": r_progress,
        "r_survival": r_survival,
        "r_orientation": r_orientation,
        "r_jerk": r_jerk,
        "r_goal": r_goal,
        "r_tip": r_tip,
        "reward_total": total}

        return float(total), details




