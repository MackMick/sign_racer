import random
import sqlite3


def get_text():
    conn = sqlite3.connect("prompts.db")
    cursor = conn.cursor()

    cursor.execute("SELECT text FROM prompts")
    prompts = [row[0] for row in cursor.fetchall()]

    selected = random.choice(prompts)

    return selected
