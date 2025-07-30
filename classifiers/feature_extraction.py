
from classifiers.body_cls import BodyEvaluator
from classifiers.face_cls import FaceEvaluator
from classifiers.emotion_cls import ExpressionEvaluator
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    PoseLandmarker, 
    PoseLandmarkerOptions,
    RunningMode
)
import cv2
import torch
from torchvision import transforms

class Features:
    def __init__(self, 
                 ub_model_path='models/pose_landmarker_full.task', 
                 ub_block_tol=0.05,
                 fc_model_path='models/face_landmarker.task', 
                 threshold=(35, 35), 
                 frame_window=30,
                 device='cuda', 
                 emote_model_path='models/mobilenet_model.pth'):

        self.bodyTrack = BodyEvaluator(frame_window=frame_window)
        self.faceTrack = FaceEvaluator(frame_window=frame_window)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.emoteTrack = ExpressionEvaluator(num_emotions=7)
        self.emoteTrack.load_state_dict(torch.load(emote_model_path, map_location=self.device))
        self.emoteTrack.to(self.device)
        self.emoteTrack.eval()
        self.emote_prob = None

        
        self.nn_input_feature = ["yaw", "pitch", "roll", "face_ratio", "body_yaw", "avg_body_movement", "Fear", "Surprise"]
        self.rule_input_feature = ["eye_closed","blink_rate", "eye_closed_duration","mar", "left_eye_blocked", "right_eye_blocked", "rot_direction", "Angry","Disgust"]
        self.emote_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

        self.all_features = self.nn_input_feature + self.rule_input_feature

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # key/used coordinates for Pose Landmark
        self.key_ub_landmark = {
            "nose": 0, "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
            "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
            "left_ear": 7, "right_ear": 8, "left_mouth": 9, "right_mouth": 10,
            "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
            "left_wrist": 15, "right_wrist": 16, "left_index": 19, "right_index": 20,
            "left_hip": 23, "right_hip": 24
        }

        #key/used coordinates for Faciallandmark
        self.key_fb_landmark = {
            'nose_tip': [1], 'chin': [199],
            'upper_lips': [11, 302, 72], 'lower_lips': [16, 316, 86],
            'l_end_lip': [61], 'r_end_lip': [291],
            'right_eye': [33, 160, 144, 159, 145, 158, 153, 133],
            'left_eye': [263, 387, 373, 386, 374, 385, 380, 369],
            'left_pupil': [468], 'right_pupil': [473],
            'left_pupil_box': [475, 476, 477, 474], 'right_pupil_box': [470, 471, 472, 469],
            'forehead': [9]
        }

        self.ub_coord = {name: np.zeros(5, dtype=float) for name in self.key_ub_landmark}
        self.face_coord = {name: np.zeros((len(indices), 3), dtype=float) for name, indices in self.key_fb_landmark.items()}
        self.fc_t_matrix = None
        self.last_landmark = None

        pose_options = PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(ub_model_path),
            running_mode=RunningMode.LIVE_STREAM,
            result_callback=self.return_result,
            num_poses=1
        )
        face_options = FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(fc_model_path),
            running_mode=RunningMode.LIVE_STREAM,
            num_faces=1,
            output_facial_transformation_matrixes=True,
            output_face_blendshapes=True,
            result_callback=self.return_result
        )

        self.face_landmarker = FaceLandmarker.create_from_options(face_options)
        self.body_landmarker = PoseLandmarker.create_from_options(pose_options)

        self.threshold = threshold
        self.ub_block_tol = ub_block_tol

    #callback from Mediapipe landmarker
    def return_result(self, result, output_image, timestamp):
        if hasattr(result, "pose_world_landmarks") and result.pose_world_landmarks:
            for name, idx in self.key_ub_landmark.items():
                part = result.pose_world_landmarks[0][idx]
                self.ub_coord[name] = np.array([part.x, part.y, part.z, part.presence, part.visibility])

        if hasattr(result, "face_landmarks") and result.face_landmarks:
            lm_3d = result.face_landmarks[0]
            for name, idx in self.key_fb_landmark.items():
                self.face_coord[name] = np.array([[lm_3d[i].x, lm_3d[i].y, lm_3d[i].z] for i in idx], dtype=float)
            self.last_landmark = lm_3d

        if hasattr(result, "facial_transformation_matrixes") and result.facial_transformation_matrixes:
            self.fc_t_matrix = np.array(result.facial_transformation_matrixes[0], dtype=float)

    # runs asynchronous landmark detection
    def process(self, frame, timestamp_ms):
        ts = int(timestamp_ms)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.body_landmarker.detect_async(mp_img, ts)
        self.face_landmarker.detect_async(mp_img, ts)

    # returns cropped face region (helper function for ExpressionEvaluator class)
    def get_face_roi(self, frame):
        if self.last_landmark is None:
            return None
        h, w, _ = frame.shape
        xs = [int(p.x * w) for p in self.last_landmark]
        ys = [int(p.y * h) for p in self.last_landmark]
        x_min, x_max = max(min(xs) - 10, 0), min(max(xs) + 10, w)
        y_min, y_max = max(min(ys) - 10, 0), min(max(ys) + 10, h)
        return frame[y_min:y_max, x_min:x_max], (x_min, y_min)

    #updates all evaluator modules using current landmarks and frames (FaceEvaluator, BodyEvaluator, ExpressionEvaluator)
    def update(self, timestamp, frame):
        t_s = timestamp / 1000.0
        self.faceTrack.update(self.face_coord, self.fc_t_matrix, self.threshold[0], self.threshold[1], t_s)
        self.bodyTrack.update(self.ub_coord)

        roi = self.get_face_roi(frame)
        if roi is not None:
            face_roi, _ = roi
            try:
                img_tensor = self.transform(face_roi).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.emoteTrack(img_tensor)
                    probs = torch.softmax(output, dim=1)
                    self.emote_prob = probs.squeeze(0).detach().cpu().numpy()
            except Exception:
                self.emote_prob = None

    def extract_feat(self):
        fc_dict = self.faceTrack.get_nn_feature()
        ub_dict = self.bodyTrack.get_nn_feature()
        emote = self.emote_prob if self.emote_prob is not None else np.zeros(7)
        emote_dict = dict(zip(self.emote_labels, emote))
        full_dict = {**fc_dict, **ub_dict, **emote_dict}
        feature_vec = [full_dict.get(i) for i in self.nn_input_feature]
        return feature_vec, full_dict


    def extract_rules(self):
        fc_dict = self.faceTrack.get_rule_feature()
        ub_dict = self.bodyTrack.get_rule_feature()
        emote = self.emote_prob if self.emote_prob is not None else np.zeros(7)
        emote_dict = dict(zip(self.emote_labels, emote))
        full_dict = {**fc_dict, **ub_dict, **emote_dict}
        filtered_dict = {k: full_dict[k] for k in self.rule_input_feature if k in full_dict}
        return filtered_dict

    #rule based inference function
    def check_rules(self):
        rules_dict = self.extract_rules()
        alpha = 1
        if rules_dict["eye_closed_duration"] >= 5:
            alpha*=0

        if rules_dict["right_eye_blocked"] == 1 or rules_dict["left_eye_blocked"] == 1:
            alpha*=0

        if rules_dict["blink_rate"] >= 12:
            alpha*=0.5

        if rules_dict["rot_direction"] == 0:
            alpha*=0.5

        if rules_dict["mar"] > 1.0:
            alpha*=0.5

        if rules_dict["Disgust"] >= 0.5:
            alpha*=0.8
        
        if rules_dict["Angry"] >=0.5:
            alpha*=0.9
        return alpha

    # just only shows particular features of the current person 
    def visualize(self):
        fc_dict = self.faceTrack.get_stats()
        ub_dict = self.bodyTrack.get_stats()
        emote = self.emote_prob if self.emote_prob is not None else np.zeros(7)
        max_idx = np.argmax(emote)
        current_emotion = self.emote_labels[max_idx]
        emote_dict = {"current_emotion": current_emotion, "confidence": emote[max_idx]}
        return {**fc_dict, **ub_dict, **emote_dict}

    def close(self):
        self.face_landmarker.close()
        self.body_landmarker.close()