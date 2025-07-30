import numpy as np
from collections import deque
import math
import time

class FaceEvaluator:
    def __init__(self, frame_window):
        self.yaw                 = None
        self.roll                = None 
        self.pitch               = None 

        self.frame_window        = frame_window

        self.on_cam_flags        = deque(maxlen = frame_window)     # deque of if face was on the came or not per frame
        self.face_on_cam         = False                            # if face is facing forward on camera
        self.face_ratio          = 0.0                              # ratio of True:False in self.on_cam_flags

        self.eye_closed            = False
        self.blink_start           = None
        self.blink_end_times       = deque()
        self.blink_rate            = 0.0
        self.eye_closed_start_time = None
        self.eye_closed_duration   = 0.0

        self.mouth_open     = False
        self.mar            = None                                # mouth-aspect-ratio

        self.gaze_dir       = None                                 # gaze direction (center, up/down, left/right)
        self.gaze_loc       = None                                 #gaze location (coordinates)


    def r_matrix_to_euler(self,R):
        """
        Converts a rotation matrix (uses matrix included in Face Landmarker) to Euler angles (yaw, pitch, roll).
        => helper function of self.eval_head_angle()
        """
        r_31 = R[2,0]
        if abs(r_31)!=1:
            theta_1 = - math.asin(r_31)
            c1      = math.cos(theta_1)
            psi1    = math.atan2(R[2,1]/c1, R[2,2]/c1)
            phi1    = math.atan2(R[1,0]/c1, R[0,0]/c1)
            return psi1, theta_1, phi1
        else:
            phi = 0
            if r_31 == -1:
                theta = math.pi/2
                psi   = phi + math.atan2(R[0,1], R[0,2])
            else:
                theta = -math.pi/2
                psi   = -phi + math.atan2(-R[0,1], -R[0,2])
            return psi, theta, phi
        
    #https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    def eval_head_angle(self, t_matrix):
        """
        Extract yaw, pitch, roll from transformation matrix
        """
        if t_matrix is not None:
            R                = t_matrix[:3,:3]
            yaw, pitch, roll = self.r_matrix_to_euler(R)
            self.yaw         = math.degrees(yaw)
            self.pitch       = math.degrees(pitch)
            self.roll        = math.degrees(roll)

    def check_face_dir(self, h_threshold=25, v_threshold=25):
        """
        Check if face is facing toward camera based on yaw/pitch thresholds.
        """
        if self.yaw is None or self.pitch is None:
            return
        on_cam = (abs(self.yaw)<=h_threshold and abs(self.pitch) <=v_threshold)
        self.on_cam_flags.append(on_cam)

        self.face_on_cam = on_cam
        self.face_ratio  = sum(self.on_cam_flags) / len(self.on_cam_flags)

    #https://www.mdpi.com/2079-9292/11/19/3183
    def ear_calculate(self, P):
        """
        Calculates Eye Aspect Ratio (helper function of eval_blink)
        """
        p1, p4 = P[-1], P[0]
        p2, p6 = P[1], P[2]
        p3, p5 = P[-3], P[-2]

        p2_p6 = np.linalg.norm(p2 - p6)
        p3_p5 = np.linalg.norm(p3 - p5)
        p1_p4 = np.linalg.norm(p1 - p4)
        if p1_p4 < 1e-6:
            return 0
        else:
            return (p2_p6 + p3_p5) / (2 * p1_p4)
        
    
    def eval_blink(self, coord, timestamp, threshold = 0.1):
        left_P    = np.asarray(coord["left_eye"])[:,:2]
        right_P   = np.asarray(coord["right_eye"])[:,:2]
        ear_left  = self.ear_calculate(left_P)
        ear_right = self.ear_calculate(right_P)
        avg_ear   = (ear_left + ear_right) /2
        closed = avg_ear < threshold

        if timestamp is None:   #start
            timestamp = time.time()

        if closed and not self.eye_closed:   #eye just closed
            self.blink_start = timestamp

        elif self.eye_closed and not closed and self.blink_start is not None:     #eye just opened
            self.blink_end_times.append(timestamp)
            self.blink_start = None
        
        self.eye_closed = closed
        self.update_blink_rate(timestamp)

        #eye is closed currently
        if self.eye_closed:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = timestamp
            self.eye_closed_duration = timestamp - self.eye_closed_start_time
        else:
            self.eye_closed_start_time = None
            self.eye_closed_duration = 0.0

    
    def update_blink_rate(self, now = None):
        if now is None:
            now = time.time()

        while self.blink_end_times and (now - self.blink_end_times[0]) > self.frame_window:
            self.blink_end_times.popleft()

        count = len(self.blink_end_times)
        self.blink_rate = (count / self.frame_window) * 60 if self.frame_window > 0 else 0.0

    #Mouth-aspect ratio
    #Real Time Driver Drowsiness Detection Using Opencv And Facial Landmarks(Thulasimani, 2021) 
    def eval_mouth(self, coord, threshold = 0.5):
        p1          = np.asarray(coord["r_end_lip"])[0][:2]
        p5          = np.asarray(coord["l_end_lip"])[0][:2]
        p2,p3,p4    = coord["upper_lips"]
        p8,p7,p6    = coord["lower_lips"]

        p2_p8 = (np.asarray(p2) - np.asarray(p8))[:2]
        p3_p7 = (np.asarray(p3) - np.asarray(p7))[:2]
        p4_p6 = (np.asarray(p4) - np.asarray(p6))[:2]
        p1_p5 = (p1) - np.asarray(p5)
        if np.linalg.norm(p1_p5) < 1e-6:
            self.mar = 0
            self.mouth_open = False
        else:
            self.mar = (np.linalg.norm(p2_p8) + np.linalg.norm(p3_p7) + np.linalg.norm(p4_p6)) / (3 * np.linalg.norm(p1_p5))
            self.mouth_open = self.mar > threshold


    def attain_hitpoint(self, coord, name, direction):
        """
        Project a gaze vector to a virtual 2D plane and return hitpoint (using ray-intersection)
        helper function of self.calculate_gaze()
        """
        eyes        = np.asarray(coord[name])
        P0          = np.mean(eyes,axis = 0)
        V           = direction
        norm_vec    = np.array([0.,0.,1.])
        plane_point = np.array([0, 0, -0.5])
        numerator   = np.dot(norm_vec, (plane_point - P0))
        denominator = np.dot(norm_vec, V)
        if denominator == 0:
            return None
        t = numerator / denominator
        return P0 + t * V

    def find_direction(self, coord, pupil_name, eye_name):
        """
        Estimate gaze direction based on head pose and eye-box midpoints
        helper function of self.find_direction()
        """
        yaw, pitch = np.deg2rad(self.yaw), np.deg2rad(self.pitch)
        direction  = np.array([-np.sin(pitch),
                               np.sin(yaw) * np.cos(pitch),
                               -np.cos(yaw) * np.cos(pitch)
                               ])

        pupil_mid = np.mean(np.asarray(coord[pupil_name]), axis=0)
        eyes_mid  = np.mean(np.asarray(coord[eye_name]), axis=0)

        if eye_name == "left_eye_box":
            side_direction = (eyes_mid - pupil_mid)
        else:
            side_direction = (pupil_mid - eyes_mid)
        return direction, side_direction[:2]
    
    
    def calculate_gaze(self, coord):
        """
        Estimate 2D gaze location from both eyes' projection onto a plane.
        """
        if self.yaw is None or self.pitch is None or self.eye_closed:
            self.gaze_dir = "OOB"
            self.gaze_loc = None
            return
        
        left_direction, l_sd = self.find_direction(coord, "left_pupil_box", 
                                                   "left_eye")
        right_direction, r_sd = self.find_direction(coord, "right_pupil_box", 
                                                    "right_eye")


        left_direction += 2 *np.array([l_sd[0], l_sd[1], 0])
        right_direction += 2*np.array([r_sd[0], r_sd[1], 0])
        
        l_hitpoint = self.attain_hitpoint(coord, "left_eye", left_direction)
        r_hitpoint = self.attain_hitpoint(coord, "right_eye",right_direction)
        if l_hitpoint is None or r_hitpoint is None:
            self.gaze_dir = "OOB"
            self.gaze_loc = None
            return
        
        self.gaze_loc = [l_hitpoint[:2],r_hitpoint[:2]]
        mid = (l_hitpoint + r_hitpoint) / 2

        if not (0 <= mid[0] <= 1 and 0 <= mid[1] <= 1):
            self.gaze_dir = "OOB"
        else:
            self.gaze_dir = (("Top" if mid[1] < 0.4 else "Bottom" if mid[1] > 0.6 else "Center") +
                                 ("-Left" if mid[0] < 0.4 else "-Right" if mid[0] > 0.6 else ""))

    def get_nn_feature(self):
        """
        Return only features used for AttentionNet NN
        """
        return {
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll,
            "face_ratio": self.face_ratio,
        }

    def get_rule_feature(self):
        """
        Return only features used for Rule-based Inference
        """
        return {"eye_closed": int(self.eye_closed),
                "blink_rate": self.blink_rate,
                "eye_closed_duration": self.eye_closed_duration,
                "mar": self.mar
                }
        
    def get_stats(self):
        """
        Return only features for the "Visualization mode" (no attention score calculation)
        """
        return {
            "blink rate": self.blink_rate,
            "face on cam": self.face_on_cam,
            "mouth open": self.mouth_open,
            "gaze direction": self.gaze_dir,
            "gaze_loc": self.gaze_loc,
            "eye closed": self.eye_closed
        }

    def update(self, coord, t_matrix, h_thresh, v_thresh, t_s):
        self.eval_head_angle(t_matrix)
        self.check_face_dir(h_thresh, v_thresh)
        self.eval_blink(coord, t_s)
        self.eval_mouth(coord)
        self.calculate_gaze(coord)