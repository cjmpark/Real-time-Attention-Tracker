import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QWidget, QTextEdit, QHBoxLayout, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QGuiApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from collections import deque
import torch    
import joblib
import time
from classifiers.attn_cls import AttentionNet
from classifiers.feature_extraction import Features

import warnings
warnings.filterwarnings("ignore") 

class AttnApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Attention Tracker")
    
        self.mode = "Track Attention"

        self.model = AttentionNet(input_dim=8)
        self.model.load_state_dict(torch.load("models/chosen_model.pth", map_location=torch.device('cpu')))
        self.model.eval()
        self.scaler = joblib.load("data_set/minmax_scaler.pkl")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.extractor = Features()
        self.header = ["yaw", "pitch", "roll", "face_ratio", "body_yaw", "avg_body_movement", "Fear", "Surprise"]
        scale_needed = ["yaw", "pitch", "roll", "face_ratio", "body_yaw", "avg_body_movement"]
        self.eng_labels = {"yaw": "Head Turn",
                             "pitch": "Head Tilt",
                             "roll": "Head Roll",
                             "face_ratio": "Facing Forward Ratio",
                             "body_yaw": "Torso Rotation",
                             "avg_body_movement": "Body Motion",
                             "blink rate": "Blinking rate", 
                             "face on cam": "Face Visible",
                             "mouth open": "Mouth Open",
                             "gaze direction": "Gaze Direction",
                             "eye closed": "Eyes closed",
                             "lean_direction": "Leaning Direction",
                             "rot_direction": "Body Rotation Direction",
                             "left_eye_blocked": "Left Eye Covered",
                             "right_eye_blocked": "Right Eye Covered",
                             "avg_ub_movement": "Upper Body Motion",
                             "emotion": "Expression"
                            }
        self.kor_label = {"yaw": "좌우 머리 회전",
                             "pitch": "상하 머리 기울임",
                             "roll": "머리 회전",
                             "face_ratio": "정면 인식 비율",
                             "body_yaw": "몸통 회전",
                             "avg_body_movement": "상체 움직임",
                             "blink rate": "눈 깜빡임 빈도", 
                             "face on cam": "얼굴 감지",
                             "mouth open": "입 열림",
                             "gaze direction": "시선 방향",
                             "eye closed": "눈 감음",
                             "lean_direction": "상체 기울기 방향",
                             "rot_direction": "상체 회전 방향",
                             "left_eye_blocked": "왼쪽 눈 감지되지 않음",
                             "right_eye_blocked": "왼쪽 눈 감지되지 않음",
                             "avg_ub_movement": "상체 움직임",
                             "emotion": "감정"
                            }
        
        self.language = "eng"

        self.scale_idx = [i for i, col in enumerate(self.header) if col in scale_needed]

        self.init_ui()
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.started = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.fps         = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.att_history = deque(maxlen=int(self.fps*5))
        self.att_avg     = deque(maxlen=100)
        self.graphed_att = deque(maxlen=100)
    
    def init_ui(self):
        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(True)
        self.image_label.setMinimumSize(480, 270)

        self.feature_box = QTextEdit()
        self.feature_box.setReadOnly(True)
        self.feature_box.setFont(QFont("Courier", 13)) 

        self.att_button = QPushButton("Track Attention")
        self.att_button.clicked.connect(lambda: self.set_mode("Track Attention"))

        self.state_button = QPushButton("Track State")
        self.state_button.clicked.connect(lambda: self.set_mode("Track State"))

        self.lang_button = QPushButton("Switch to Korean")
        self.lang_button.clicked.connect(self.toggle_language)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_tracking)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.start_button)
        left_layout.addWidget(self.att_button)
        left_layout.addWidget(self.state_button)
        left_layout.addWidget(self.lang_button)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.canvas)
        right_layout.addWidget(self.feature_box)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout,5)
        main_layout.addLayout(right_layout,2)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def set_mode(self, mode):
        self.mode = mode
        if mode == "Track State":
            self.canvas.hide()
        else:
            self.canvas.show()

    def start_tracking(self):
        self.started = True
        self.start_button.setDisabled(True)

    def update_frame(self):
        if not self.started:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        timestamp = int(time.time() * 1000)
        self.extractor.process(frame, timestamp)
        self.extractor.update(timestamp, frame)

        if self.mode == "Track Attention":
            features, features_dict = self.extractor.extract_feat()
            features_scaled = np.array(features, dtype=float).reshape(1, -1)
            features_scaled[:, self.scale_idx] = self.scaler.transform(features_scaled[:, self.scale_idx])
            att_score = self.extractor.check_rules()
            if att_score != 0:
                input_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    logit = self.model(input_tensor)
                    att_score*=torch.sigmoid(logit).item()
            self.att_history.append(att_score)
            self.graphed_att.append(att_score)

            valid_scores = [s for s in self.att_history if not np.isnan(s)]
            avg_att_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            self.att_avg.append(avg_att_score)
            self.update_graph()
            cv2.putText(frame, f"Avg Attension Score: {avg_att_score:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)

            features_text = ""
            label_dict = self.eng_labels if self.language == "eng" else self.kor_label
            for key in self.header:
                val = features_dict.get(key) 
                if val is None:
                    val = 0.0
                if key == "Fear" or key == "Surprise":
                    continue    
                label = label_dict.get(key,key)           
                features_text += f"{label}: {val:.2f}\n" 
            self.feature_box.setText(features_text)


        else:
            stats = self.extractor.visualize()
            if "gaze_loc" in stats and stats["gaze_loc"] is not None:
                left, right = stats["gaze_loc"]
                h, w, _ = frame.shape
                cx1, cy1 = int(left[0] * w), int(left[1] * h)
                cx2, cy2 = int(right[0] * w), int(right[1] * h)
                cv2.circle(frame, (cx1, cy1), 8, (0, 255, 0), -1)
                cv2.circle(frame, (cx2, cy2), 8, (0, 255, 0), -1)

            label_dict = self.eng_labels if self.language == "eng" else self.kor_label
            text = ""
            for key, val in stats.items():
                if val is None:
                    val = ""
                if key == "gaze_loc" or key == "confidence":
                     continue
                label = label_dict.get(key,key)       
                text += f"{label}: {val}\n" 
            self.feature_box.setText(text)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def toggle_language(self):
        if self.language == "eng":
            self.language = "kor"
            self.lang_button.setText("Switch to English")
        else:
            self.language = "eng"
            self.lang_button.setText("Switch to Korean")


    def update_graph(self):
        self.ax.clear()
        x_val = list(range(len(self.graphed_att)))
        y_val = list(self.graphed_att)

        self.ax.plot(x_val, y_val)
        self.ax.set_xlim(0, max(100, len(self.graphed_att)))
        self.ax.set_ylim(0,1)

        if y_val:
            x,y = x_val[-1], y_val[-1]
            self.ax.annotate(f"{y:3f}", (x,y), xytext = (-30,10),fontsize = 8, color = 'red')
        
        x_val2 = list(range(len(self.att_avg)))
        y_val2 = list(self.att_avg)
        if y_val2:
            x,y = x_val2[-1], y_val2[-1]
            self.ax.plot(x_val2, y_val2)
            self.ax.annotate(f"{y:3f}", (x,y), xytext = (-30,10),fontsize = 8, color = 'blue')

        self.ax.set_title("Real-Time Attention Score")
        self.canvas.draw()
    
    def closeEvent(self,event):
        self.cap.release()
        self.extractor.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttnApp()
    window.show()
    sys.exit(app.exec_())

