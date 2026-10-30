# ============================================================================
# AR Head-Tilt Racing - a tiny racing game you steer by tilting your head.
#
# HOW IT PLAYS: your webcam watches your face. Tilt your head left or right
# and the yellow car at the bottom of the screen slides across the road with
# you. Dodge the falling blocks. The longer you survive, the faster it gets
# and the higher your score. Press R to start/restart, Q to quit.
#
# HOW IT SEES YOUR HEAD TILT: Google's MediaPipe face AI (the
# face_landmarker.task model file, auto-downloaded on first run) finds the
# corners of your two eyes in each webcam frame. The angle of the line
# between your eyes IS your head tilt - that is the whole trick.
#
# THE ONE FILE: everything (download helper, Obstacle class, game loop)
# lives in this single file.
#
# BEFORE YOU RUN: pip install opencv-python mediapipe numpy
# WHEN DEBUGGING (reasoned from the code, not re-tested):
#   - "face_landmarker.task not found": the auto-download failed; check your
#     internet connection and run again.
#   - Car drifts without tilting: sit square to the camera when pressing R;
#     bright even lighting helps the eye tracking.
# ============================================================================

# OpenCV: webcam frames and all the drawing
import cv2
# built-in (imported, currently unused)
import sys
# Google's face-tracking AI library
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# arrays: the game screen is built as number grids
import numpy as np
# built-in: atan2/degrees for the head-tilt angle
import math
# built-in: checks whether the model file exists yet
import os
# built-in: downloads the AI model on first run
import urllib.request
# built-in: frame timing and score keeping
import time
# built-in: where and when obstacles appear
import random

# Where the face-tracking AI model lives online, and where we save it.
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

# Downloads the AI model once (a few MB). After the first run the file is
# already there and this does nothing.
def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading MediaPipe Face model to {MODEL_PATH}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Download complete.")
        except Exception as e:
            print(f"Download failed: {e}")

# One falling block on the road. Knows where it is, how big and fast it is,
# and how to draw itself.
class Obstacle:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        # Road boundaries
        self.road_left = w * 0.2
        self.road_right = w * 0.8
        self.x = random.uniform(self.road_left + 50, self.road_right - 50)
        self.y = -50
        self.width = random.randint(40, 100)
        self.height = 40
        self.speed = random.uniform(5, 12)
        self.color = (random.randint(50, 100), random.randint(50, 100), random.randint(150, 255))
        self.passed = False

    # Slides the block down the screen. speed_multiplier rises over time,
    # which is what makes the game get faster the longer you survive.
    def update(self, speed_multiplier):
        self.y += self.speed * speed_multiplier

    def draw(self, img):
        cv2.rectangle(img, (int(self.x - self.width/2), int(self.y - self.height/2)), 
                      (int(self.x + self.width/2), int(self.y + self.height/2)), self.color, cv2.FILLED)
        cv2.rectangle(img, (int(self.x - self.width/2), int(self.y - self.height/2)), 
                      (int(self.x + self.width/2), int(self.y + self.height/2)), (255, 255, 255), 2)

# The whole game: camera setup, face tracking, steering, obstacles, drawing.
def main():
    download_model()
    
    if not os.path.exists(MODEL_PATH):
        print("ERROR: face_landmarker.task not found!")
        return

    # Set up the face AI: track one face, only report when at least 50% sure.
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Open the webcam and ask it for a 1280x720 picture.
    cap = cv2.VideoCapture(0)
    # Give higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Game state
    state = "START"
    score = 0
    speed_multiplier = 1.0
    obstacles = []
    
    w, h = 1280, 720
    ret, frame = cap.read()
    if ret:
        h, w, _ = frame.shape
        
    car_x = w / 2
    car_width = 60
    car_height = 100
    car_y = h - 120
    
    smoothed_angle = 0.0

    print("====================================")
    print("      AR Head-Tilt Racing           ")
    print("====================================")
    print("Controls:")
    print(" - Tilt your head LEFT/RIGHT to steer")
    print(" - Press 'R' to Start / Restart")
    print(" - Press 'Q' to Quit")
    print("====================================")

    last_time = time.time()
    
    # The main loop: once per webcam frame.
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        
        frame = cv2.flip(frame, 1) # selfie view
        h, w, _ = frame.shape
        
        # Darken the background to make neon UI pop
        frame = cv2.convertScaleAbs(frame, alpha=0.5, beta=20)
        overlay = np.zeros_like(frame, dtype=np.uint8)
        
        # Detect Face
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)

        face_angle = 0.0
        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            # Left Eye (Outer corner 33), Right Eye (Outer corner 263)
            # In flipped frame, 263 is on the left visually, 33 is on the right visually
            lx = landmarks[263].x * w
            ly = landmarks[263].y * h
            rx = landmarks[33].x * w
            ry = landmarks[33].y * h
            
            # Draw eyes
            cv2.circle(frame, (int(lx), int(ly)), 5, (0, 255, 255), -1)
            cv2.circle(frame, (int(rx), int(ry)), 5, (0, 255, 255), -1)
            cv2.line(frame, (int(lx), int(ly)), (int(rx), int(ry)), (255, 255, 255), 2)
            
            dx = rx - lx
            dy = ry - ly
            if dx != 0:
                face_angle = math.degrees(math.atan2(dy, dx))
            
            # Smooth angle
            # Blend 70% old angle + 30% new measurement so the steering does
            # not jitter with every tiny head wobble.
            smoothed_angle = smoothed_angle * 0.7 + face_angle * 0.3
            
            # Map angle to Car X target
            # Angle roughly goes from -30 (tilt left) to +30 (tilt right)
            road_left = w * 0.2
            road_right = w * 0.8
            road_center = w / 2
            
            max_tilt = 30.0
            tilt_clamped = max(-max_tilt, min(max_tilt, smoothed_angle))
            normalized_tilt = tilt_clamped / max_tilt # -1.0 to 1.0
            
            target_x = road_center + normalized_tilt * (road_right - road_left) / 2
            
            if state == "PLAYING":
                car_x += (target_x - car_x) * 0.3
                car_x = max(road_left + car_width/2, min(road_right - car_width/2, car_x))

        # Game Logic
        # dt = seconds since the last frame; using real time (not frame
        # count) keeps the game speed steady even if the camera slows down.
        curr_time = time.time()
        dt = curr_time - last_time
        last_time = curr_time
        
        if state == "PLAYING":
            score += dt * 10 * speed_multiplier
            speed_multiplier += dt * 0.02 # Gets faster!
            
            # Spawn obstacles
            if random.random() < 0.03 * speed_multiplier:
                obstacles.append(Obstacle(w, h))
                
            for obs in obstacles[:]:
                obs.update(speed_multiplier)
                obs.draw(overlay)
                
                # Collision check
                # Box-overlap test: crash if the car's box and the obstacle's
                # box overlap (with a 10-pixel mercy margin).
                cx, cy = car_x, car_y
                ox, oy = obs.x, obs.y
                if abs(cx - ox) < (car_width/2 + obs.width/2 - 10) and \
                   abs(cy - oy) < (car_height/2 + obs.height/2 - 10):
                    state = "GAME OVER"
                    
                if obs.y > h + 50:
                    obstacles.remove(obs)
        
        # Draw Road
        cv2.line(overlay, (int(w*0.2), 0), (int(w*0.2), h), (255, 255, 255), 3)
        cv2.line(overlay, (int(w*0.8), 0), (int(w*0.8), h), (255, 255, 255), 3)
        
        # Moving road lines
        offset = int((time.time() * 300 * speed_multiplier) % 100)
        for y in range(offset, h + 100, 100):
            cv2.line(overlay, (int(w/2), y), (int(w/2), y + 50), (100, 100, 100), 5)
            
        # Draw Car
        # Body
        cv2.rectangle(overlay, (int(car_x - car_width/2), int(car_y - car_height/2)), 
                      (int(car_x + car_width/2), int(car_y + car_height/2)), (0, 255, 255), cv2.FILLED)
        # Windows
        cv2.rectangle(overlay, (int(car_x - car_width/2 + 10), int(car_y - car_height/4)), 
                      (int(car_x + car_width/2 - 10), int(car_y + car_height/4)), (50, 50, 50), cv2.FILLED)
        # Thrusters glowing
        if state == "PLAYING":
            pulse = int(5 + 5 * math.sin(time.time() * 20))
            cv2.circle(overlay, (int(car_x - 15), int(car_y + car_height/2)), pulse + 5, (0, 150, 255), -1)
            cv2.circle(overlay, (int(car_x + 15), int(car_y + car_height/2)), pulse + 5, (0, 150, 255), -1)
            cv2.circle(overlay, (int(car_x - 15), int(car_y + car_height/2)), 5, (255, 255, 255), -1)
            cv2.circle(overlay, (int(car_x + 15), int(car_y + car_height/2)), 5, (255, 255, 255), -1)

        # Composite HUD
        # Blend the game drawings (overlay) over the dimmed webcam picture.
        frame = cv2.addWeighted(overlay, 0.9, frame, 1.0, 0)
        
        # UI Text
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"SCORE: {int(score)}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(frame, f"TILT ANGLE: {int(smoothed_angle)}", (w - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        if state == "START":
            # Glow text
            box_x, box_y = w//2 - 250, h//2 - 50
            cv2.rectangle(frame, (box_x - 20, box_y - 40), (box_x + 500, box_y + 40), (0, 0, 0), -1)
            cv2.rectangle(frame, (box_x - 20, box_y - 40), (box_x + 500, box_y + 40), (0, 255, 255), 2)
            cv2.putText(frame, "Press 'R' to Start Racing!", (box_x, box_y + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                        
        elif state == "GAME OVER":
            box_x, box_y = w//2 - 250, h//2 - 50
            cv2.rectangle(frame, (box_x - 20, box_y - 80), (box_x + 500, box_y + 40), (0, 0, 0), -1)
            cv2.rectangle(frame, (box_x - 20, box_y - 80), (box_x + 500, box_y + 40), (0, 0, 255), 3)
            cv2.putText(frame, "CRASHED!", (box_x + 120, box_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            cv2.putText(frame, "Press 'R' to Restart", (box_x + 50, box_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        cv2.imshow("Head-Tilt AR Racing", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            state = "PLAYING"
            score = 0
            speed_multiplier = 1.0
            obstacles.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
