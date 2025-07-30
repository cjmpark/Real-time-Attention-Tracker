import sys
import os
import numpy as np
import cv2
import torch
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    PoseLandmarker, 
    PoseLandmarkerOptions,
    RunningMode
)
from torchvision import transforms
from classifiers.body_cls import BodyEvaluator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class BodyOnlyFeatures:
    def __init__(self, 
                 ub_model_path='models/pose_landmarker_full.task', 
                 frame_window=30):

        self.bodyTrack = BodyEvaluator(frame_window=frame_window)
        self.key_ub_landmark = {
            "nose": 0, "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
            "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
            "left_ear": 7, "right_ear": 8, "left_mouth": 9, "right_mouth": 10,
            "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
            "left_wrist": 15, "right_wrist": 16, "left_index": 19, "right_index": 20,
            "left_hip": 23, "right_hip": 24
        }
        self.ub_coord = {name: np.zeros(5, dtype=float) for name in self.key_ub_landmark}
        self.ub_2dcoord = {name: np.zeros(5, dtype=float) for name in self.key_ub_landmark}

        pose_options = PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(ub_model_path),
            running_mode=RunningMode.LIVE_STREAM,
            result_callback=self.return_result,
            num_poses=1
        )
        self.body_landmarker = PoseLandmarker.create_from_options(pose_options)

    def return_result(self, result, output_image, timestamp):
        if hasattr(result, "pose_world_landmarks") and result.pose_world_landmarks:
            for name, idx in self.key_ub_landmark.items():
                part = result.pose_world_landmarks[0][idx]
                self.ub_coord[name] = np.array([part.x, part.y, part.z, part.presence, part.visibility])
        if hasattr(result, "pose_landmarks") and result.pose_landmarks:
            for name, idx in self.key_ub_landmark.items():
                part = result.pose_landmarks[0][idx]
                self.ub_2dcoord[name] = np.array([part.x, part.y, part.z, part.presence, part.visibility])

    def process(self, frame, timestamp_ms):
        ts = int(timestamp_ms)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.body_landmarker.detect_async(mp_img, ts)

    def update(self):
        self.bodyTrack.update(self.ub_coord)

    def get_nn_features(self):
        return self.bodyTrack.get_nn_feature()

    def get_rule_features(self):
        return self.bodyTrack.get_rule_feature()

    def get_stats(self):
        return self.bodyTrack.get_stats()

    def close(self):
        self.body_landmarker.close()
