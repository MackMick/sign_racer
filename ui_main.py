import sys
import cv2
import time
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QFrame
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer

import os, PyQt5.QtCore
_base = os.path.join(os.path.dirname(PyQt5.QtCore.__file__), "plugins")
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_PLUGIN_PATH"] = _base
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(_base, "platforms")

# Import your existing logic
from testclassification import (
    Landmarker, FaceLandmarkerWrapper, 
    predict_letter, draw_landmarks_on_image, 
    get_input,
    get_text, model
)


class ASLApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Typeracer Demo")
        self.resize(1200, 700)

        # --- Video capture ---
        self.cap = cv2.VideoCapture(0)
        self.last_time = time.time()

        # --- Landmarkers ---
        self.hand_landmarker = Landmarker()
        self.face_landmarker = FaceLandmarkerWrapper()

        # --- Game text setup ---
        self.correct_string = get_text().upper()
        self.fullstring = ""
        self.colorvector = [0] * len(self.correct_string)
        self.skinny_set = {"I"}
        self.adjusted_placements = [20]
        for pos in range(1, len(self.correct_string)):
            if self.correct_string[pos - 1] in self.skinny_set:
                self.adjusted_placements.append(15)
            else:
                self.adjusted_placements.append(40)

        # --- Layout ---
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px;")
        self.video_label.setFixedSize(960, 720)

        self.status_label = QLabel("Status: Running")
        self.status_label.setFont(QFont("Arial", 16))
        self.status_label.setStyleSheet("color: #00FF88;")

        self.prompt_label = QLabel("Text to write")
        self.prompt_label.setFont(QFont("Monospace", 20))
        self.prompt_label.setStyleSheet("color: #450912;")

        right_frame = QFrame()
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.status_label)

        right_layout.addWidget(self.prompt_label)

        right_layout.addStretch()
        right_frame.setLayout(right_layout)
        right_frame.setStyleSheet("background-color: #222; border-radius: 10px; padding: 20px;")

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(right_frame)
        self.setLayout(main_layout)

        # --- Timer ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # about 30 FPS

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        hand_result = self.hand_landmarker.detect(frame)
        frame, prediction = predict_letter(frame, hand_result)

        face_landmarks = self.face_landmarker.detect(frame)
        if face_landmarks is not None:
            frame = self.face_landmarker.draw(frame, face_landmarks)
            if self.face_landmarker.take_input(face_landmarks):
                get_input(prediction)

        frame = draw_landmarks_on_image(frame, hand_result)
        #combined = draw_ui(frame, self.correct_string, self.fullstring)

        # --- Convert to QImage for display ---
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

        # --- FPS calculation ---
        now = time.time()
        fps = 1.0 / (now - self.last_time)
        self.last_time = now
        self.status_label.setText(f"Status: Running ({fps:.1f} FPS)")

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        self.hand_landmarker.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ASLApp()
    window.show()
    sys.exit(app.exec_())
