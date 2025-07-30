from classifiers.feature_extraction import Features
import cv2
import numpy as np
import time

extractor = Features()
cap = cv2.VideoCapture(0)



font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 5.0
font_color = (255, 0, 0)
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
        
        emote_result = extractor.get_face_roi(frame)
        if emote_result:
            face_dim, (x_min, y_min) = emote_result
            h, w = face_dim.shape[:2]
            x_max = x_min + w
            y_max = y_min + h
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
            idx_emote = np.argmax(extractor.emote_prob)
            emotion = extractor.emote_labels[idx_emote]
            cv2.putText(frame, str(emotion), (x_max+10, y_min + 10), font, font_scale, font_color, 2, line_type)

        cv2.imshow("Real-time Feature Viewer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
            

finally:
    cap.release()
    extractor.close()
    cv2.destroyAllWindows()
