import sys
import cv2
import time
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QFrame
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QSizePolicy

from prompt_display import PromptDisplay

from PyQt5.QtCore import Qt

# Fix for missing Qt plugins
import os, PyQt5.QtCore
_base = os.path.join(os.path.dirname(PyQt5.QtCore.__file__), "plugins")
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_PLUGIN_PATH"] = _base
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(_base, "platforms")

# Import your existing logic
from testclassification import (
    Landmarker, FaceLandmarkerWrapper,
    predict_letter, draw_landmarks_on_image,
    get_text, model
)


class ASLApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Typeracer Demo")
        self.resize(2000, 700)

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

        # --- Display current prediction ---
        self.current_letter = ""

        self.show_current_letter = QLabel("Current letter:")
        self.show_current_letter.setFont(QFont("Monospace", 20))
        self.show_current_letter.setStyleSheet("color: #FFFFFF;")

        # --- Prompt display (colored text progress) ---
        self.prompt_display = PromptDisplay(self.correct_string)

        # --- Right panel ---
        right_frame = QFrame()
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.prompt_display)
        right_layout.addWidget(self.show_current_letter)
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

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        hand_result = self.hand_landmarker.detect(frame)
        frame, prediction = predict_letter(frame, hand_result)

        self.current_letter = prediction
        self.show_current_letter.setText(prediction)

        # Face detection & blink trigger
        face_landmarks = self.face_landmarker.detect(frame)
        if face_landmarks is not None:
            frame = self.face_landmarker.draw(frame, face_landmarks)
            if self.face_landmarker.take_input(face_landmarks):  # blink triggers input
                if prediction:
                    self.fullstring += prediction
                    self.prompt_display.type_letter(prediction)

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
