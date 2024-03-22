from unitree_sdk2.build import example

import time
import datetime
import numpy as np
import math
import torch
from scipy.spatial.transform import Rotation as R
from tensordict import TensorDict

from setproctitle import setproctitle


ORBIT_JOINT_ORDER = [
    'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 
    'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
    'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'
]

SDK_JOINT_ORDER = [
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'
]


class Robot:

    def __init__(self, cfg):
        self.cfg = cfg
        self._robot = example.RobotIface()
        self._robot.start_control()
        self.default_joint_pos = np.array(
            [
                0.1, -0.1,  0.1, -0.1,  
                0.8,  0.8,  1.0,  1.0, 
                -1.5, -1.5, -1.5, -1.5
            ], 
        )
        
        self.dt = 0.02
        self.latency = 0.0
        self.angvel_history = np.zeros((3, 3))
        self.action_buf = np.zeros((12, 4))
        self.command = np.zeros(4) # linvel_xy, angvel_z, base_height
    
    def reset(self):
        self.prev_actions = np.zeros((12,))
        self.rpy = self._robot.get_rpy()
        self.update()
        return self._compute_obs()
    
    def update(self):
        self.vel = self._robot.get_velocity()
        self.feet_pos_b = self._robot.get_feet_pos().reshape(4, 3)
        self.quat = self._robot.get_quat()
        self.rot = R.from_quat(self.quat[[1, 2, 3, 0]])
        self.jpos_sdk = self._robot.get_joint_pos()
        self.jvel_sdk = self._robot.get_joint_vel()
        
        self.prev_rpy = self.rpy
        self.rpy = self._robot.get_rpy()
        self.angvel = (self.rpy - self.prev_rpy) / self.dt
        self.angvel_history = np.roll(self.angvel_history, -1, axis=1)
        self.angvel_history[:, -1] = self.angvel
        
        self.projected_gravity = self._robot.get_projected_gravity()

        self.lxy = self._robot.lxy()
        self.rxy = self._robot.rxy()
        self.latency = (datetime.datetime.now() - self._robot.timestamp).total_seconds()
        self.command[0] = self.lxy[1]
        self.command[1] = -self.lxy[0]


        # self.command[2] = -self.rxy[0] * 1.2
        # self.command[3] = 0.3 + self.rxy[1] * 0.1
        heading = np.array([1., -self.rxy[0] * 1.3])
        heading = heading / np.linalg.norm(heading)
        self.command[2:4] = heading
    
    def step(self, action=None):
        if action is not None:
            self.action_buf[:, 1:] = self.action_buf[:, :-1]
            self.action_buf[:, 0] = action
            
            jpos_target = action * 0.5 + self.default_joint_pos
            self._robot.set_command(self.orbit_to_sdk(jpos_target))
        self.update()
        return self._compute_obs()

    def _compute_obs(self):
        angvel = self.angvel_history.mean(axis=1)
        angvel = self.rot.inv().apply(angvel)

        obs = [
            self.command,
            self.quat,
            angvel,
            self.projected_gravity,
            self.sdk_to_orbit(self.jpos_sdk),
            self.sdk_to_orbit(self.jvel_sdk),
            self.action_buf.reshape(-1),
        ]
        return np.concatenate(obs, dtype=np.float32)
    
    @staticmethod
    def orbit_to_sdk(joints: np.ndarray):
        return np.flip(joints.reshape(3, 2, 2), axis=2).transpose(1, 2, 0).reshape(-1)
    
    @staticmethod
    def sdk_to_orbit(joints: np.ndarray):
        return np.flip(joints.reshape(2, 2, 3), axis=1).transpose(2, 0, 1).reshape(-1)


class Runner:
    def __init__(self, policy):
        self.dt = 0.02

    @torch.inference_mode()
    def step(self):
        self.time = time.perf_counter()

def main():
    import yaml
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    setproctitle("play_go2")

    example.init_channel("enp3s0")

    init_pos = np.array([
        0.0, 0.9, -1.8, 
        0.0, 0.9, -1.8, 
        0.0, 0.9, -1.8, 
        0.0, 0.9, -1.8
    ])
    robot = Robot(cfg)
    
    robot._robot.set_kp(25.)
    robot._robot.set_kd(0.5)

    path = "/home/btx0424/isaac_ws/active-adaptation/scripts/policy.pt"
    policy = torch.load(path)
    # policy = lambda td: torch.zeros(12)

    robot._robot.set_command(init_pos)
    obs = robot.reset()
    obs = robot._compute_obs()
    print(obs.shape)

    try:
        td = TensorDict({
            "policy": torch.as_tensor(obs),
            "is_init": torch.tensor(1, dtype=bool),
            "context_adapt_hx": torch.zeros(128)
        }, []).unsqueeze(0)
        with torch.inference_mode():
            while True:
                start = time.perf_counter()
                # robot.update()
                # print(robot.feet_pos_b[:, 2])
                print(robot.command)
                policy(td)
                action = td["action"].cpu().numpy()
                # print(td["state_value"].item())
                # print(processed_actions)
                # print(robot._robot.get_joint_pos_target())
                # obs = torch.as_tensor(robot._compute_obs())
                obs = torch.as_tensor(robot.step(action))
                td["next", "policy"] = obs.unsqueeze(0)
                td["next", "is_init"] = torch.tensor([0], dtype=bool)
                td = td["next"]
                time.sleep(max(0, 0.02 - (time.perf_counter() - start)))

    except KeyboardInterrupt:
        pass
    finally:
        pass
        
if __name__ == "__main__":
    main()

    
