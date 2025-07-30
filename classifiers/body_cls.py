import numpy as np
from collections import deque
from statistics import mean

class BodyEvaluator:
    def __init__(self, frame_window = 30):
        self.sb_angle           = None                          # side-bend angle
        self.fb_angle           = None                          # forward-bend angle
        self.body_yaw           = None                          # Horizontal body rotation
        
        self.leye_blocked       = False                         # left eye blocked or not
        self.reye_blocked       = False                         # right eye blocked or not

        self.rot_direction      = None                          # direction of the body (where it's facing)
        self.lean_direction     = None                          # direction of the lean

        self.prev_coord         = {}                   
        self.ub_movement        = deque(maxlen=frame_window)    # upper body movement (average of coordinates)
        self.avg_ub_movement    = None


    
    def calc_dir(self, coord):
        """
        Estimate body yaw based on left/right shoulder vector and categorizes the body's direction (which way it's facing => rot_direction)
        => calculates self.body_yaw, self.rot_direction 
        """
        ls           = coord["left_shoulder"]
        rs           = coord["right_shoulder"]
        shoulder_vec = np.array(ls) - np.array(rs)

        yaw_rad      = np.arctan2(shoulder_vec[2], shoulder_vec[0])
        yaw_deg      = np.degrees(yaw_rad)
        self.body_yaw     = yaw_deg
        rotation = ""
        if -40 <= self.body_yaw <= 40:
            rotation = "forward"
        elif -90 <= self.body_yaw < -40:
            rotation = "right"
        elif 40 < self.body_yaw <= 90:
            rotation = "left"
        else:
            rotation = "backward"
        self.rot_direction = rotation

    def eval_angle(self, coord):
        """
        Calculate forward/backward and side leaning angles 
        => calculates self.fb/sb_angle, self.lean_direction
        """
        l_shoulder = coord["left_shoulder"]
        r_shoulder = coord["right_shoulder"]
        l_hip      = coord["left_hip"]
        r_hip      = coord["right_hip"]

        # visibility/presence check
        pres_s = (l_shoulder[-2]>0.2 and r_shoulder[-2]>0.2)
        pres_h = (l_hip[-2]>0.2 and r_hip[-2]>0.2)
        if not (pres_s and pres_h):
            self.fb_angle = 0
            self.sb_angle = 0
            return

        # calculate midpoint => calculate midpoint vector
        ls, rs = l_shoulder[:3], r_shoulder[:3]
        lh, rh = l_hip[:3], r_hip[:3]
        mid_s  = (ls+rs)/2
        mid_h  = (lh+rh)/2
        v      = np.zeros(3, dtype=np.float32)
        v[0]   = mid_h[0] - mid_s[0]
        v[1]   = mid_h[1] - mid_s[1]
        v[2]   = mid_s[2] - mid_h[2]
        
        self.sb_angle = np.degrees(np.arctan2(v[2],v[1]))
        self.fb_angle = np.degrees(np.arctan2(v[0],v[1]))

        direction = "stationary"
        if self.fb_angle < -30:
            direction = "forward"
        elif self.fb_angle > 30:
            direction = "backward"
        if self.sb_angle > 20:
            direction += "-right"
        elif self.sb_angle < -20:
            direction += "-left"
        self.lean_direction = direction
        

        
    def check_eye_block(self, coord, tol=0.1):
        """
        Checks if eyes are blocked
        => calculates self.leye/reye_blocked
        """
        l_eye_keys = ["left_eye", "left_eye_inner", "left_eye_outer"]
        r_eye_keys = ["right_eye", "right_eye_inner", "right_eye_outer"]
        arm_keys   = ["left_shoulder", "left_elbow", "left_wrist", "left_index",
                      "right_shoulder", "right_elbow", "right_wrist", "right_index"]
        l_eye_pts  = [coord[k] for k in l_eye_keys]
        r_eye_pts  = [coord[k] for k in r_eye_keys]
        arm_pts    = np.array([coord[k] for k in arm_keys])

        def is_blocked(eye, parts):
            e = np.array(eye).ravel()
            if e.size == 5 and (e[3] < 0.5 or e[4] < 0.5):
                return True
            if parts.shape[1] >= 4:
                parts = parts[parts[:, 3] >= 0.2]
            if parts.size == 0:
                return False

            dx = np.abs(parts[:, 0] - e[0]) < tol
            dy = np.abs(parts[:, 1] - e[1]) < tol
            dz = parts[:, 2] <= e[2]
            return bool(np.any(dx & dy & dz))

        self.leye_blocked = any(is_blocked(eye, arm_pts) for eye in l_eye_pts)
        self.reye_blocked = any(is_blocked(eye, arm_pts) for eye in r_eye_pts)

    def calc_movement(self, coord):
        """
        Estimates total upper-body movement by frame-to-frame landmark difference.
        =>calculates self.ub_movement and self.avg_body_movement
        """
        movement = []
        for key, landmarks in coord.items():
            if landmarks[-2] > 0.2: 
                if key in self.prev_coord:
                    dist = np.linalg.norm(landmarks[:3] - self.prev_coord[key])
                    movement.append(dist)
                self.prev_coord[key] = landmarks[:3]
        total_movement = sum(movement)
        self.ub_movement.append(total_movement)
        self.avg_ub_movement = mean(self.ub_movement) if self.ub_movement else 0.0
        
    def get_nn_feature(self):
        """
        Return only features used for AttentionNet NN
        """
        return {"body_yaw": self.body_yaw,
                "avg_body_movement": self.avg_ub_movement,
                "left_eye_blocked": int(self.leye_blocked),
                "right_eye_blocked": int(self.reye_blocked)
                }

    def get_rule_feature(self):
        """
        Return only features used for Rule-based Inference
        """
        if self.rot_direction == "forward":
            rot_direction = 1
        else:
            rot_direction = 0
        return {"left_eye_blocked": int(self.leye_blocked),
                "right_eye_blocked": int(self.reye_blocked),
                "rot_direction": rot_direction
                }
    
    def get_stats(self):
        """
        Return only features for the "Visualization mode" (no attention score calculation)
        """
        return {"lean_direction": self.lean_direction,
                "rot_direction": self.rot_direction,
                "left_eye_blocked": self.leye_blocked,
                "right_eye_blocked": self.reye_blocked,
                "avg_ub_movement":self.avg_ub_movement
            }

    def update(self, coord):
        self.eval_angle(coord)
        self.check_eye_block(coord)
        self.calc_dir(coord)
        self.calc_movement(coord)


            
