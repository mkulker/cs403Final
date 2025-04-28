import numpy as np
import mujoco

class YourCtrl:
    def __init__(self, model, data, target_points, Kp=400.0, Kd=40.0, threshold=0.01, Kc=100.0):
        """
        PD controller in task space with decaying constant push for smooth, precise reaching.

        :param model: MuJoCo MjModel
        :param data: MuJoCo MjData
        :param target_points: (3,N) array or list of [x,y,z]
        :param Kp: Task-space proportional gain
        :param Kd: Joint-space derivative (damping) gain
        :param threshold: Distance threshold (m) to consider target reached
        :param Kc: Maximum constant task-space push gain
        """
        self.model = model
        self.data = data
        # Switching tolerance for reaching target exactly
        self.threshold = threshold
        self.Kp = Kp
        self.Kd = Kd
        self.Kc = Kc
        # End-effector body id (6th joint)
        self.eef_body = model.jnt_bodyid[5]
        # Load targets as list of (3,) arrays
        if isinstance(target_points, np.ndarray) and target_points.ndim == 2 and target_points.shape[0] == 3:
            self.targets = [target_points[:, i].astype(float).copy()
                            for i in range(target_points.shape[1])]
        else:
            self.targets = [np.array(pt, dtype=float) for pt in target_points]
        self.index = 0
        # Jacobian buffer
        self.jacp = np.zeros((3, model.nv))
        print(f"Controller init: {len(self.targets)} targets, threshold={self.threshold}")

    def CtrlUpdate(self):
        """
        Called each timestep to compute control torques.
        Terminates (zeros) after final target.
        """
        # Zero control if done
        if self.index >= len(self.targets):
            self.data.ctrl[:6] = np.zeros(6)
            return self.data.ctrl.copy()

        mujoco.mj_forward(self.model, self.data)
        ee_pos = self.data.xpos[self.eef_body].ravel()
        tgt = self.targets[self.index]
        err = tgt - ee_pos
        dist = np.linalg.norm(err)

        # Switch when error below threshold
        if dist < self.threshold:
            print(f"Reached target {self.index} at dist={dist:.4f}")
            self.index += 1
            if self.index >= len(self.targets):
                print("All targets reached. Terminating control.")
                self.data.ctrl[:6] = np.zeros(6)
                return self.data.ctrl.copy()
            print(f"Switching to target {self.index}")
            tgt = self.targets[self.index]
            err = tgt - ee_pos
            dist = np.linalg.norm(err)

        # Compute Jacobian
        mujoco.mj_jacBody(self.model, self.data, self.jacp, None, self.eef_body)
        J = self.jacp[:, :6]
        # Unit direction
        if dist > 1e-8:
            dir_unit = err / dist
        else:
            dir_unit = np.zeros(3)
        # Decaying constant push: scales from max Kc at far distances down to 0 at threshold
        push_scale = min(dist / self.threshold, 1.0)
        force = self.Kp * err + self.Kc * push_scale * dir_unit
        # Joint torques via J^T
        tau_task = J.T.dot(force)
        # Damping
        tau_damp = -self.Kd * self.data.qvel[:6]
        ctrl = tau_task + tau_damp
        self.data.ctrl[:6] = ctrl
        return self.data.ctrl.copy()

