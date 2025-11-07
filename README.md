# ASL TypeRacer

Demo project — a TypeRacer-like demo application for American Sign Language (ASL) with a live webcam feed and sign recognition.

The machine learning model is trained on an open-source dataset to recognize ASL signs. Users record signs in the prompt by blinking, or use the spacebar if performance is low.

## Installation

```bash
git clone https://github.com/mackmick/sign_racer.git
cd sign_racer
pip install -r requirements.txt
python3 ui_main.py
```

## Docker Image

Pull the latest image:

```bash
docker pull strombologni/sign_racer:latest
```

Run it (Linux example):

```bash
xhost +local:docker
docker run -it --rm \
  --device /dev/video0:/dev/video0 \
  --device /dev/dri:/dev/dri \
  --group-add video \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  strombologni/sign_racer:latest
```
