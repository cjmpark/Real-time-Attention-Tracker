from classifiers.feature_extraction import Features
import cv2
import numpy as np
import time

extractor = Features()
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
font_color = (255, 255, 255)
line_type = cv2.LINE_AA

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = int(time.time()*1000)
        extractor.process(frame, timestamp)
        extractor.update(timestamp,frame)

        h, w = frame.shape[:2]
        for name, idx in extractor.key_fb_landmark.items():
            coord = extractor.face_coord[name][0]
            x_raw,y_raw = coord[0], coord[1]
            x_px = int(x_raw * w)
            y_px = int(y_raw * h)
            cv2.circle(frame, (x_px, y_px), 8, (0, 255, 0), -1)


        nn_dict = extractor.faceTrack.get_nn_feature()
        rule_dict = extractor.faceTrack.get_rule_feature()
        stat_dict = extractor.faceTrack.get_stats()
        pose_coord = extractor.key_fb_landmark

        info_panel = np.zeros((frame.shape[0],int(frame.shape[1]/2),3), dtype=np.uint8)
        y = 30

        def draw_dict(panel, label, dct, y):
            cv2.putText(panel, label, (10, y), font+2, 0.6, (0, 255, 255), 2)
            y += 40
            for k, v in dct.items():
                if k == "gaze_loc":
                    continue
                txt = f"{k}: {round(v, 3) if isinstance(v, float) else v}"
                cv2.putText(panel, txt, (10, y), font, font_scale, font_color, 1, line_type)
                y += 40
            return y + 20
        y = draw_dict(info_panel, "NN Features", nn_dict, y)
        y = draw_dict(info_panel, "Rule Features", rule_dict, y)
        y = draw_dict(info_panel, "Stats", stat_dict, y)
        combined = np.hstack((frame, info_panel))

        cv2.imshow("Real-time Feature Viewer", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
            

finally:
    cap.release()
    extractor.close()
    cv2.destroyAllWindows()
