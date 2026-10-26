import cv2
import mediapipe as mp
import numpy as np
import math_utils
import math
import time
import winsound 

POSTURE = [1, 199, 33, 263, 61, 291]
LEFT_EYE = [33, 133, 160, 159, 144, 145]
RIGHT_EYE = [362, 263, 387, 386, 373, 374]

PROFILE = {"pitch": 0.0, "yaw": 0.0, "gaze": 0.5, "shoulder_angle": 0.0, "shoulder_y": 0.0}
STATE = {"bad_posture_ts": 0, "distracted_ts": 0, "closed_eye_ts": 0}

def init_tracker():
    mp_holistic = mp.solutions.holistic
    return mp_holistic.Holistic(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        refine_face_landmarks=True
    )

def gazePos(left_corner, right_corner, iris):
    w = math.dist(left_corner, right_corner)
    return math.dist(left_corner, iris) / w if w > 0 else 0.5

def calibrate(pitch, yaw, gaze, shoulder_angle, shoulder_y):
    PROFILE["pitch"] = pitch
    PROFILE["yaw"] = yaw
    PROFILE["gaze"] = gaze
    PROFILE["shoulder_angle"] = shoulder_angle
    PROFILE["shoulder_y"] = shoulder_y

def draw_hud_bar(panel, x, y, width, height, value, max_value, color):
    fill_width = int((value / max_value) * width) if value > 0 else 0
    fill_width = min(fill_width, width)
    cv2.rectangle(panel, (x, y), (x + width, y + height), (50, 50, 50), 1)
    cv2.rectangle(panel, (x, y), (x + fill_width, y + height), color, -1)

def frameProcessing(frame, tracker_obj):
    img_h, img_w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = tracker_obj.process(frame_rgb)
    
    pitch, yaw, gaze, shoulder_angle, shoulder_y_avg = 0, 0, 0.5, 0, 0
    status_posture, status_focus, status_eyes = "ABSENT", "DISTRAS", "ABSENT"
    face_detected = False
    
    current_time = time.time()
    
    if res.face_landmarks:
        face_detected = True
        faces = res.face_landmarks
        
        posture_points = [(int(faces.landmark[idx].x * img_w), int(faces.landmark[idx].y * img_h)) for idx in POSTURE]
        l_eye_pts = [(int(faces.landmark[idx].x * img_w), int(faces.landmark[idx].y * img_h)) for idx in LEFT_EYE]
        r_eye_pts = [(int(faces.landmark[idx].x * img_w), int(faces.landmark[idx].y * img_h)) for idx in RIGHT_EYE]
        
        l_out = (faces.landmark[33].x * img_w, faces.landmark[33].y * img_h)
        l_in = (faces.landmark[133].x * img_w, faces.landmark[133].y * img_h)
        l_iris = (faces.landmark[468].x * img_w, faces.landmark[468].y * img_h)
        
        r_in = (faces.landmark[362].x * img_w, faces.landmark[362].y * img_h)
        r_out = (faces.landmark[263].x * img_w, faces.landmark[263].y * img_h)
        r_iris = (faces.landmark[473].x * img_w, faces.landmark[473].y * img_h)
        
        pitch, yaw = math_utils.headPos(posture_points, img_w, img_h)
        ear_left = math_utils.eyePos(l_eye_pts)
        ear_right = math_utils.eyePos(r_eye_pts)
        
        gaze = (gazePos(l_out, l_in, l_iris) + gazePos(r_in, r_out, r_iris)) / 2.0
        
        if res.pose_landmarks:
            pose = res.pose_landmarks
            l_sh = pose.landmark[11]
            r_sh = pose.landmark[12]
            
            if l_sh.visibility > 0.6 and r_sh.visibility > 0.6:
                dy = r_sh.y - l_sh.y
                dx = r_sh.x - l_sh.x
                shoulder_angle = math.degrees(math.atan2(dy, dx))
                shoulder_y_avg = (l_sh.y + r_sh.y) / 2.0
            else:
                status_posture = "FĂRĂ UMERI"
        else:
            status_posture = "FĂRĂ UMERI"

        if min(ear_left, ear_right) < 0.22:
            if STATE["closed_eye_ts"] == 0:
                STATE["closed_eye_ts"] = current_time
            elif current_time - STATE["closed_eye_ts"] > 1.5:  
                status_eyes = "ALARMĂ SOMN"
                winsound.Beep(1500, 200) 
            else:
                status_eyes = "CLIPEȘTE"
        else:
            STATE["closed_eye_ts"] = 0
            status_eyes = "DESCHIȘI"
            
        if status_posture != "FĂRĂ UMERI":
            pitch_dev = abs(pitch - PROFILE["pitch"])
            shoulder_tilt = abs(shoulder_angle - PROFILE["shoulder_angle"])
            shoulder_drop = abs(shoulder_y_avg - PROFILE["shoulder_y"])
            
            status_posture = "CORECTĂ"
            if pitch_dev > 15.0 or shoulder_tilt > 5.0 or shoulder_drop > 0.05:
                if STATE["bad_posture_ts"] == 0:
                    STATE["bad_posture_ts"] = current_time
                elif current_time - STATE["bad_posture_ts"] > 2.0: 
                    status_posture = "INCORECTĂ"
            else:
                STATE["bad_posture_ts"] = 0

        status_focus = "FOCUSAT"
        if (
            abs(gaze - PROFILE["gaze"]) > 0.09
            or abs(yaw - PROFILE["yaw"]) > 18.0
            or abs(pitch - PROFILE["pitch"]) > 18.0
        ):
            if STATE["distracted_ts"] == 0:
                STATE["distracted_ts"] = current_time
            elif current_time - STATE["distracted_ts"] > 1.0: 
                status_focus = "DISTRAS"
        else:
            STATE["distracted_ts"] = 0

    hud_w = 280
    hud = np.full((img_h, hud_w, 3), (15, 15, 15), dtype=np.uint8)
    cv2.line(hud, (hud_w - 1, 0), (hud_w - 1, img_h), (70, 70, 70), 2)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(hud, "Eye2Posture", (20, 40), font, 0.75, (255, 255, 255), 1)
    cv2.line(hud, (20, 55), (260, 55), (100, 100, 100), 1)
    
    p_col = (0, 0, 255) if status_posture == "INCORECTĂ" else ((0, 165, 255) if status_posture in ("FĂRĂ UMERI", "ABSENT") else (0, 255, 0))
    f_col = (0, 0, 255) if status_focus == "DISTRAS" else (0, 255, 0)
    e_col = (0, 0, 255) if status_eyes == "ALARMĂ SOMN" else ((0, 255, 0) if status_eyes == "DESCHIȘI" else (0, 165, 255))
    
    cv2.putText(hud, "POSTURĂ:", (20, 100), font, 0.5, (200, 200, 200), 1)
    cv2.putText(hud, status_posture, (115, 100), font, 0.5, p_col, 1)
    
    cv2.putText(hud, "FOCUS:", (20, 150), font, 0.5, (200, 200, 200), 1)
    cv2.putText(hud, status_focus, (115, 150), font, 0.5, f_col, 1)
    
    cv2.putText(hud, "OCHII:", (20, 200), font, 0.5, (200, 200, 200), 1)
    cv2.putText(hud, status_eyes, (115, 200), font, 0.5, e_col, 1)
    
    if face_detected:
        cv2.putText(hud, "Deviație Cap:", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        draw_hud_bar(hud, 20, 290, 200, 10, abs(pitch - PROFILE["pitch"]), 30, (0, 165, 255))
        
        cv2.putText(hud, "Deviație Privire:", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        draw_hud_bar(hud, 20, 330, 200, 10, abs(gaze - PROFILE["gaze"]), 0.20, (0, 165, 255))
        
        cv2.putText(hud, "Înclinare Umeri:", (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        draw_hud_bar(hud, 20, 370, 200, 10, abs(shoulder_angle - PROFILE["shoulder_angle"]), 10, (255, 0, 255))
    
    combined_frame = cv2.hconcat([hud, frame])
    return combined_frame, pitch, yaw, gaze, shoulder_angle, shoulder_y_avg, status_focus, face_detected
