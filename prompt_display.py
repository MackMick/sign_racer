from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtGui import QTextCharFormat, QColor, QFont

class PromptDisplay(QTextEdit):
    """
    Displays the prompt text (the string to be typed) with color coding:
    - Green: correct letters
    - Red: incorrect letters
    - Yellow: current letter
    - Gray: remaining letters
    """

    def __init__(self, correct_text: str):
        super().__init__()
        self.correct_text = correct_text
        self.current_index = 0
        self.color_vector = [0] * len(correct_text)  # 0 = not typed, 1 = correct, 2 = wrong

        self.setReadOnly(True)
        self.setFont(QFont("Monospace", 20))
        self.setStyleSheet("background-color: #222; border: none; color: white;")
        self.update_display()

    def update_display(self):
        """Redraws the text with updated colors."""
        cursor = self.textCursor()
        self.clear()

        for i, char in enumerate(self.correct_text):
            fmt = QTextCharFormat()

            if i < self.current_index:
                if self.color_vector[i] == 1:
                    fmt.setForeground(QColor("#00FF00"))  # green = correct
                elif self.color_vector[i] == 2:
                    fmt.setForeground(QColor("#D11919"))  # red = wrong
            elif i == self.current_index:
                fmt.setForeground(QColor("#FFFF00"))  # yellow = current
                fmt.setBackground(QColor("#9C9C9C"))
            else:
                fmt.setForeground(QColor("#AAAAAA"))  # gray = not reached

            cursor.insertText(char, fmt)

    def type_letter(self, letter: str):
        """Call this when a new letter has been 'typed'."""
        if self.current_index >= len(self.correct_text):
            return

        expected = self.correct_text[self.current_index]
        self.color_vector[self.current_index] = 1 if letter == expected else 2
        self.current_index += 1
        self.update_display()

    def reset(self, new_text: str):
        """Resets the display for a new prompt."""
        self.correct_text = new_text
        self.current_index = 0
        self.color_vector = [0] * len(new_text)
        self.update_display()
