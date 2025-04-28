import numpy as np
import mujoco

class YourCtrl:
    def __init__(self, model, data, target_points, Kp=200.0, Kd=20.0, threshold=0.05):
        """
        Simplified PD controller in operational space (Jacobian-transpose) to touch each point.

        :param model: MuJoCo MjModel
        :param data: MuJoCo MjData
        :param target_points: (3,N) array or list of [x,y,z] points
        :param Kp: Task-space proportional gain
        :param Kd: Joint-space derivative gain
        :param threshold: Distance threshold to switch points
        """
        self.model = model
        self.data = data
        self.threshold = threshold  # meters
        self.Kp_task = Kp
        self.Kd = Kd
        self.eef_body = model.jnt_bodyid[5]
        # Load targets as list of (3,) arrays
        if isinstance(target_points, np.ndarray) and target_points.ndim == 2 and target_points.shape[0] == 3:
            self.targets = [target_points[:, i].astype(float).copy()
                            for i in range(target_points.shape[1])]
        else:
            self.targets = [np.array(pt, dtype=float) for pt in target_points]
        self.index = 0
        self.jacp = np.zeros((3, model.nv))
        print(f"Controller initialized with {len(self.targets)} targets, threshold={self.threshold}")

    def CtrlUpdate(self):
        """
        Apply Jacobian-transpose PD in task space to move EE to current target.
        """
        mujoco.mj_forward(self.model, self.data)
        ee_pos = self.data.xpos[self.eef_body].ravel()
        tgt = self.targets[self.index]
        err = tgt - ee_pos
        dist = np.linalg.norm(err)
        # Switch to next target if within threshold
        if dist < self.threshold:
            print(f"Reached target {self.index} at dist {dist:.3f}")
            self.index += 1
            if self.index < len(self.targets):
                print(f"Switching to target {self.index}: {self.targets[self.index]}")
                tgt = self.targets[self.index]
                err = tgt - ee_pos
            else:
                print("All targets reached. Holding final position.")
                self.index = len(self.targets) - 1
                err = np.zeros_like(err)
        # Compute Jacobian and PD torque
        mujoco.mj_jacBody(self.model, self.data, self.jacp, None, self.eef_body)
        J = self.jacp[:, :6]
        tau_task = J.T.dot(self.Kp_task * err)
        tau_damp = -self.Kd * self.data.qvel[:6]
        ctrl = tau_task + tau_damp
        self.data.ctrl[:6] = ctrl
        return self.data.ctrl.copy()
