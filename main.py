import cv2
import tracker
import time
import os

def main():
    cap = cv2.VideoCapture(0)
    tracker_obj = tracker.init_tracker()
    
    print("Eye2Posture pornit. Asigură-te ca umerii sunt vizibili pe ecran.")
    print("Apasă 'c' pt calibrare. Apasă 'q' pentru a genera raportul si a ieși.")
    
    session_start = time.time()
    focused_frames = 0
    distracted_frames = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        
        frame, curr_p, curr_yaw, curr_g, curr_sa, curr_sy, status_focus, face_detected = tracker.frameProcessing(frame, tracker_obj)
        
        if status_focus == "FOCUSAT":
            focused_frames += 1
        elif status_focus == "DISTRAS":
            distracted_frames += 1
            
        cv2.rectangle(frame, (0, frame.shape[0] - 30), (frame.shape[1], frame.shape[0]), (15, 15, 15), -1)
        cv2.putText(frame, "[C] Calibrare Postură   |   [Q] Salvează Raport & Ieșire", (20, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow("Eye2Posture", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if face_detected:
                tracker.calibrate(curr_p, curr_yaw, curr_g, curr_sa, curr_sy)
                print(f"Profil Salvat -> Pitch: {curr_p:.0f} | Yaw: {curr_yaw:.0f} | Privire: {curr_g:.2f} | Unghi Umeri: {curr_sa:.2f}")
            else:
                print("Eroare calibrare: Asigură-te că fața și umerii sunt vizibili!")
                
    cap.release()
    cv2.destroyAllWindows()
    
    total_frames = focused_frames + distracted_frames
    if total_frames > 0:
        focus_percent = (focused_frames / total_frames) * 100
        session_duration = round(time.time() - session_start)
        
        log_path = os.path.join(os.getcwd(), "session_report.txt")
        with open(log_path, "w") as f:
            f.write("=== RAPORT SESIUNE Eye2Posture ===\n")
            f.write(f"Durată Sesiune: {session_duration} secunde\n")
            f.write(f"Rată de Focus: {focus_percent:.1f}%\n")
            f.write(f"Evaluare Finală: {'OPTIM' if focus_percent >= 80 else 'ATENTIE NECESARA'}\n")
            
        print(f"\nSesiune încheiată. Raport generat în: {log_path}")

if __name__ == "__main__":
    main()
