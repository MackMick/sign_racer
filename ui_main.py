import sys
import cv2
import time
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QCheckBox, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QSizePolicy

from camera_mapping import open_camera_prefer_path

from prompt_display import PromptDisplay

from PyQt5.QtCore import Qt

import numpy as np

# Fix for missing Qt plugins
import os, PyQt5.QtCore
_base = os.path.join(os.path.dirname(PyQt5.QtCore.__file__), "plugins")
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_PLUGIN_PATH"] = _base
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(_base, "platforms")

# Import helper functions
from utilities import (
    Landmarker, FaceLandmarkerWrapper,
    predict_letter, draw_landmarks_on_image,
    get_text, model
)

import cv2, time, os
import numpy as np

def _looks_color(frame: np.ndarray) -> bool:
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        return False
    # IR greyscale => channels ~identical; color has variance between channels
    c01 = np.std(frame[:, :, 0] - frame[:, :, 1])
    c12 = np.std(frame[:, :, 1] - frame[:, :, 2])
    return (c01 > 1.0) or (c12 > 1.0)

def _warmup_and_check(cap, attempts=8, sleep=0.02) -> bool:
    ok = False
    for _ in range(attempts):
        ret, f = cap.read()
        if ret and _looks_color(f):
            ok = True
            break
        time.sleep(sleep)
    return ok

def _try_open_index(i: int) -> bool:
    """Try several backend/format combos on index i. Return True on color frames."""
    # 1) Default backend, MJPG 1280x720
    cap = cv2.VideoCapture(i)  # no CAP_V4L2 here
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if _warmup_and_check(cap):
            cap.release()
            return True
        cap.release()

    # 2) Default backend, YUYV 640x480 (very compatible)
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if _warmup_and_check(cap):
            cap.release()
            return True
        cap.release()

    # 3) Explicit V4L2 backend, MJPG 1280x720
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if _warmup_and_check(cap):
            cap.release()
            return True
        cap.release()

    # 4) Explicit V4L2 backend, YUYV 640x480
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if _warmup_and_check(cap):
            cap.release()
            return True
        cap.release()

    return False

def find_working_camera(max_tested=6):
    """Return the index of a camera that yields color frames."""
    # Optional: force an index (e.g., CAM_INDEX=0)
    env_idx = os.getenv("CAM_INDEX")
    if env_idx and env_idx.isdigit():
        i = int(env_idx)
        if _try_open_index(i):
            print(f"Using camera index {i} (env override)")
            return i

    # Prefer 0 first; IR is often 2
    order = list(range(min(max_tested, 6)))
    if 0 in order:
        order.remove(0)
        order.insert(0, 0)

    for i in order:
        if _try_open_index(i):
            print(f"Using camera index {i}")
            return i

    print("No suitable color camera found.")
    return None

class ASLApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Typeracer Demo")
        self.resize(2000, 700)

        # --- Video capture ---
        camera_index = find_working_camera()
        if camera_index is None:
            raise RuntimeError("No camera found — please check your device.")

        self.cap = cv2.VideoCapture(camera_index)      # let OpenCV pick backend
        
        self.last_time = time.time()

        self.start_time = None

        # --- Landmarkers ---
        self.hand_landmarker = Landmarker()
        self.face_landmarker = FaceLandmarkerWrapper()

        # --- Game text setup ---
        self.correct_string = get_text().upper()
        self.fullstring = ""
        self.colorvector = [0] * len(self.correct_string)

        self.prompt_done = False

        # --- UI Setup ---
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px;")
        self.video_label.setScaledContents(True)
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        self.status_label = QLabel("Status: Running")
        self.status_label.setFont(QFont("Arial", 16))
        self.status_label.setStyleSheet("color: #00FF88;")

        # --- Togglable buttons ---
        self.checkbox_style = """
            QCheckBox {
                font-size: 16px;
                color: #FFFFFF;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #4CAF50;   /* border color */
                border-radius: 4px;           /* slightly rounded */
                background-color: white;      /* fill color */
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;    /* fill when checked */
                border-color: #4CAF50;
            }
        """
        self.lefthanded_bool = False

        self.lefthanded_checkbox = QCheckBox("Sign with left hand?")
        self.lefthanded_checkbox.stateChanged.connect(self.toggle_left)
        self.lefthanded_checkbox.setStyleSheet(self.checkbox_style)

        self.draw_bool = True

        self.drawlandmarks_checkbox = QCheckBox("Draw Landmarks?")
        self.drawlandmarks_checkbox.setChecked(True)
        self.drawlandmarks_checkbox.stateChanged.connect(self.toggle_draw)
        self.drawlandmarks_checkbox.setStyleSheet(self.checkbox_style)

        # --- Spacebar input button
        self.spacebar_bool = False
        self.space_pressed = False
        self.input_processed = False

        self.setFocusPolicy(Qt.StrongFocus)

        self.spacebar_checkbox = QCheckBox("SPACE for input?")
        self.spacebar_checkbox.stateChanged.connect(self.toggle_spacebar)
        self.spacebar_checkbox.setStyleSheet(self.checkbox_style)

        # --- Display current prediction ---
        self.current_letter = ""

        self.show_current_letter = QLabel("Current letter:")
        self.show_current_letter.setFont(QFont("Monospace", 20))
        self.show_current_letter.setStyleSheet("color: #FFFFFF;")

        # --- Display current time --- 
        self.show_time = QLabel("Current time: ")
        self.show_time.setFont(QFont("Monospace", 20))
        self.show_time.setStyleSheet("color: #FFFFFF;")

        # --- Prompt display (colored text progress) ---
        self.prompt_display = PromptDisplay(self.correct_string)

        # --- Right panel ---
        right_frame = QFrame()
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.prompt_display)
        right_layout.addWidget(self.show_time)
        right_layout.addWidget(self.show_current_letter)
        right_layout.addWidget(self.spacebar_checkbox)
        right_layout.addWidget(self.lefthanded_checkbox)
        right_layout.addWidget(self.drawlandmarks_checkbox)
        right_layout.addStretch()
        right_frame.setLayout(right_layout)
        right_frame.setStyleSheet("background-color: #222; border-radius: 10px; padding: 20px;")

        # --- Main layout ---
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(right_frame)
        self.setLayout(main_layout)

        # --- Timer ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def toggle_draw(self, button):
        self.draw_bool = self.drawlandmarks_checkbox.isChecked()
        print("drawing landmarks? : ", self.draw_bool)
        self.video_label.setFocus()

    def toggle_left(self, button):
        self.lefthanded_bool = self.lefthanded_checkbox.isChecked()
        print("left handed?: ", self.lefthanded_bool)
        self.video_label.setFocus()

    # --- Spacebar handling --- #use for other buttons also
    def toggle_spacebar(self, button):
        self.spacebar_bool = self.spacebar_checkbox.isChecked()
        print("Spacebar mode:", self.spacebar_bool)
        # give focus back to the video area or main window
        self.video_label.setFocus()  # or self.setFocus()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.space_pressed = True
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.space_pressed = False
        else:
            super().keyReleaseEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.space_pressed = False  # reset when released
        else:
            super().keyReleaseEvent(event)

    def reset_prompt(self):
        self.correct_string = get_text().upper()
        self.fullstring = ""
        self.colorvector = [0] * len(self.correct_string)

        self.prompt_done = False

        self.start_time = None

        self.prompt_display.reset(self.correct_string)

    def process_input(self, prediction, face_landmarks):
        """Handles both blink-based and spacebar-based input modes."""
        if not self.spacebar_bool:
            # Blink-based trigger
            if self.face_landmarker.take_input(face_landmarks):
                if prediction:
                    self.fullstring += prediction
                    self.prompt_display.type_letter(prediction)

        else:
            # Spacebar-based trigger (only once per press)
            if self.space_pressed and not self.input_processed:
                if prediction:
                    self.fullstring += prediction
                    self.prompt_display.type_letter(prediction)
                    self.input_processed = True  # prevent multiple inputs per press

            # Reset when spacebar released
            if not self.space_pressed:
                self.input_processed = False


    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        hand_result = self.hand_landmarker.detect(frame)

        if self.lefthanded_bool:
            frame, prediction = predict_letter(frame, hand_result, -1) #flips the reading
        else:
            frame, prediction = predict_letter(frame, hand_result)

        self.current_letter = prediction

        letter_to_show = prediction if prediction != " " else "(SPACE)"
        self.show_current_letter.setText(f"Currently: {letter_to_show}")

        # --- Face detection ---
        face_landmarks = self.face_landmarker.detect(frame)
        if face_landmarks is not None and self.draw_bool:
            frame = self.face_landmarker.draw(frame, face_landmarks)

            # Use a separate method to handle input logic
            self.process_input(prediction, face_landmarks)
        

        if len(self.fullstring) == 1 and self.start_time == None:
            self.start_time = time.time()

        #update time indicator:
        if self.start_time and not self.prompt_done:
            self.show_time.setText("Time: " + str(round(time.time() - self.start_time, 2)))


        # --- Check if prompt is complete ---
        if len(self.fullstring) == len(self.correct_string) and not self.prompt_done:
            self.prompt_done = True
            prompt_time = time.time() - self.start_time 
            self.show_time.setText("FINAL: " + str(round(prompt_time, 2)) + " seconds")
            QTimer.singleShot(5000, self.reset_prompt)
            
        # --- Draw landmarks on hand ---
        if self.draw_bool:
            frame = draw_landmarks_on_image(frame, hand_result)

        # --- Convert to QImage for PyQt ---
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        # --- FPS ---
        now = time.time()
        fps = 1.0 / (now - self.last_time)
        self.last_time = now
        self.status_label.setText(f"Status: Running ({fps:.1f} FPS)")

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        self.hand_landmarker.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ASLApp()
    window.show()
    sys.exit(app.exec_())
