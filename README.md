# Eye2Posture

A real-time ergonomics and attention tracking application built with Python, OpenCV, and MediaPipe. It monitors posture, gaze direction, and eye states via webcam to help maintain healthy desk habits.



\#### Key Features

\- \*\*Holistic Tracking:\*\* Combines MediaPipe Holistic and Face Mesh to track head orientation (using PnP head pose estimation), eye aspect ratio (EAR) for blinking, and shoulder alignment.

\- \*\*Dynamic Calibration:\*\* Allows users to set their ideal posture baseline instantly with a single keypress (`C`), accommodating different camera angles and hardware setups.

\- \*\*HUD \& Alerts:\*\* Features a minimalist dark-mode HUD with real-time progress bars, visual indicators, and native system sound alerts (`winsound`) for prolonged slouching or extended eye closure.

\- \*\*Session Reporting:\*\* Automatically generates a performance summary (`session\_report.txt`) upon exit, logging session duration and focus metrics.

