FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    # X11 / xcb stack
    libx11-6 libx11-xcb1 libxcb1 libxcb-render0 libxcb-shm0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-shape0 libxcb-sync1 \
    libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 libxcb-render-util0 \
    libxkbcommon0 libxkbcommon-x11-0 \
    # GL + common Qt deps
    libgl1 libgl1-mesa-dri libsm6 libxext6 libxrender1 libfontconfig1 libfreetype6 \
    libglib2.0-0 \
    mesa-utils \
 && rm -rf /var/lib/apt/lists/*

# Optional: default to XCB; you can override at runtime
ENV QT_QPA_PLATFORM=xcb


WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "ui_main.py"]


