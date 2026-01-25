#!/usr/bin/env python3
"""
SF Security Camera Backend - Multi-Person Pose Detection
- YOLOv8-Pose for multi-person skeleton detection
- Action recognition based on body keypoints
- Audio analysis (speech detection, volume levels, sound classification)
- Real-time streaming of all events
"""

import asyncio
import websockets
import ssl
import json
import base64
import numpy as np
import cv2
import time
from collections import defaultdict
from datetime import datetime
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# YOLOv8 Pose (works on CPU/GPU, no NVIDIA required)
from ultralytics import YOLO

# Configuration
CONFIDENCE_THRESHOLD = 0.5
AUDIO_THRESHOLD = 0.02

# YOLO Pose keypoint indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


class MultiPersonPoseDetector:
    """Detect multiple people and their poses using YOLOv8-Pose"""

    def __init__(self):
        print("[DETECTOR] Loading YOLOv8-Pose model...")
        self.model = YOLO('yolov8n-pose.pt')
        self.next_person_id = 0
        self.tracked_persons = {}  # Store previous frame positions for ID tracking
        print("[DETECTOR] YOLOv8-Pose multi-person detector loaded")

    def detect(self, frame):
        """Detect all people and their poses in frame"""
        h, w = frame.shape[:2]

        # Run YOLO pose detection
        results = self.model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)

        detections = []

        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints
            boxes = results[0].boxes

            if keypoints.xy is not None and len(keypoints.xy) > 0:
                for i in range(len(keypoints.xy)):
                    kpts = keypoints.xy[i].cpu().numpy()  # Shape: (17, 2)
                    conf = keypoints.conf[i].cpu().numpy() if keypoints.conf is not None else np.ones(17)

                    # Get bounding box
                    if boxes is not None and len(boxes.xyxy) > i:
                        box = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = box
                        box_conf = float(boxes.conf[i].cpu().numpy())
                    else:
                        # Calculate from keypoints
                        valid_pts = kpts[conf > 0.3]
                        if len(valid_pts) < 3:
                            continue
                        x1, y1 = valid_pts.min(axis=0)
                        x2, y2 = valid_pts.max(axis=0)
                        box_conf = float(np.mean(conf))

                    # Normalize coordinates
                    x_norm = x1 / w
                    y_norm = y1 / h
                    w_norm = (x2 - x1) / w
                    h_norm = (y2 - y1) / h

                    # Assign person ID based on position tracking
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    person_id = self._assign_person_id(center, w, h)

                    # Extract landmarks
                    landmarks = self._extract_landmarks(kpts, conf, w, h)

                    detection = {
                        "id": person_id,
                        "x": float(x_norm),
                        "y": float(y_norm),
                        "width": float(w_norm),
                        "height": float(h_norm),
                        "confidence": box_conf,
                        "class": "person",
                        "landmarks": landmarks
                    }
                    detections.append(detection)

        # Clean old tracked persons
        self._cleanup_tracks()

        return detections

    def _assign_person_id(self, center, frame_w, frame_h):
        """Assign consistent person ID based on position tracking"""
        min_dist = float('inf')
        best_id = None
        threshold = 0.15 * max(frame_w, frame_h)  # 15% of frame size

        current_time = time.time()

        for pid, data in self.tracked_persons.items():
            dist = np.sqrt((center[0] - data['center'][0])**2 + (center[1] - data['center'][1])**2)
            if dist < threshold and dist < min_dist:
                min_dist = dist
                best_id = pid

        if best_id is not None:
            self.tracked_persons[best_id] = {'center': center, 'last_seen': current_time}
            return best_id
        else:
            new_id = self.next_person_id
            self.next_person_id += 1
            self.tracked_persons[new_id] = {'center': center, 'last_seen': current_time}
            return new_id

    def _cleanup_tracks(self):
        """Remove old tracked persons"""
        current_time = time.time()
        to_remove = [pid for pid, data in self.tracked_persons.items()
                     if current_time - data['last_seen'] > 2.0]
        for pid in to_remove:
            del self.tracked_persons[pid]

    def _extract_landmarks(self, kpts, conf, frame_w, frame_h):
        """Extract normalized landmarks from keypoints"""
        def get_lm(idx):
            if idx < len(kpts) and conf[idx] > 0.3:
                return {
                    "x": float(kpts[idx][0] / frame_w),
                    "y": float(kpts[idx][1] / frame_h),
                    "v": float(conf[idx])
                }
            return {"x": 0.5, "y": 0.5, "v": 0.0}

        return {
            "nose": get_lm(NOSE),
            "left_shoulder": get_lm(LEFT_SHOULDER),
            "right_shoulder": get_lm(RIGHT_SHOULDER),
            "left_elbow": get_lm(LEFT_ELBOW),
            "right_elbow": get_lm(RIGHT_ELBOW),
            "left_wrist": get_lm(LEFT_WRIST),
            "right_wrist": get_lm(RIGHT_WRIST),
            "left_hip": get_lm(LEFT_HIP),
            "right_hip": get_lm(RIGHT_HIP),
            "left_knee": get_lm(LEFT_KNEE),
            "right_knee": get_lm(RIGHT_KNEE),
            "left_ankle": get_lm(LEFT_ANKLE),
            "right_ankle": get_lm(RIGHT_ANKLE)
        }


class ActionRecognizer:
    """Recognize actions based on pose landmarks and motion for multiple people"""

    def __init__(self):
        self.tracks = defaultdict(lambda: {
            "positions": [],
            "landmarks_history": [],
            "velocities": [],
            "last_seen": 0,
            "action": "unknown",
            "action_confidence": 0.0,
            "prev_action": "unknown"
        })
        self.frame_count = 0

    def analyze(self, detections):
        self.frame_count += 1
        results = []

        for det in detections:
            track_id = det["id"]
            landmarks = det.get("landmarks")

            if not landmarks:
                continue

            track = self.tracks[track_id]

            # Store landmark history
            track["landmarks_history"].append(landmarks)
            if len(track["landmarks_history"]) > 30:
                track["landmarks_history"] = track["landmarks_history"][-30:]

            # Calculate center position
            cx = det["x"] + det["width"] / 2
            cy = det["y"] + det["height"] / 2
            track["positions"].append((cx, cy, self.frame_count))
            if len(track["positions"]) > 30:
                track["positions"] = track["positions"][-30:]

            track["last_seen"] = self.frame_count

            # Calculate velocity
            velocity = 0
            if len(track["positions"]) >= 2:
                p1 = track["positions"][-2]
                p2 = track["positions"][-1]
                velocity = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
                track["velocities"].append(velocity)
                if len(track["velocities"]) > 30:
                    track["velocities"] = track["velocities"][-30:]

            # Classify action using landmarks
            action, confidence = self._classify_action(track, landmarks, velocity, det)

            track["prev_action"] = track["action"]
            track["action"] = action
            track["action_confidence"] = confidence

            results.append({
                "person_id": track_id,
                "action": action,
                "confidence": confidence,
                "velocity": velocity,
                "bbox": {
                    "x": det["x"],
                    "y": det["y"],
                    "width": det["width"],
                    "height": det["height"]
                }
            })

        # Clean old tracks
        to_remove = [k for k, v in self.tracks.items() if self.frame_count - v["last_seen"] > 30]
        for k in to_remove:
            del self.tracks[k]

        return results

    def _classify_action(self, track, lm, velocity, det):
        """Classify action based on pose landmarks"""

        # Check visibility
        def is_visible(pt):
            return pt.get("v", 0) > 0.3

        # Get key points
        l_shoulder = lm["left_shoulder"]
        r_shoulder = lm["right_shoulder"]
        l_hip = lm["left_hip"]
        r_hip = lm["right_hip"]
        l_knee = lm["left_knee"]
        r_knee = lm["right_knee"]
        l_ankle = lm["left_ankle"]
        r_ankle = lm["right_ankle"]
        l_wrist = lm["left_wrist"]
        r_wrist = lm["right_wrist"]

        # Check if we have enough visible points
        key_points = [l_shoulder, r_shoulder, l_hip, r_hip]
        visible_count = sum(1 for p in key_points if is_visible(p))
        if visible_count < 2:
            return "unknown", 0.3

        # Calculate body metrics
        shoulder_y = (l_shoulder["y"] + r_shoulder["y"]) / 2
        hip_y = (l_hip["y"] + r_hip["y"]) / 2
        knee_y = (l_knee["y"] + r_knee["y"]) / 2 if is_visible(l_knee) or is_visible(r_knee) else hip_y + 0.15
        ankle_y = (l_ankle["y"] + r_ankle["y"]) / 2 if is_visible(l_ankle) or is_visible(r_ankle) else knee_y + 0.15

        shoulder_x = (l_shoulder["x"] + r_shoulder["x"]) / 2
        hip_x = (l_hip["x"] + r_hip["x"]) / 2

        # Torso angle (vertical = 0, horizontal = 90)
        torso_dx = hip_x - shoulder_x
        torso_dy = hip_y - shoulder_y
        torso_angle = abs(np.degrees(np.arctan2(torso_dx, max(torso_dy, 0.001))))

        # Leg bend angle
        def angle_3pts(p1, p2, p3):
            if not (is_visible(p1) and is_visible(p2) and is_visible(p3)):
                return 180  # Assume straight if not visible
            v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]])
            v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm < 1e-6:
                return 180
            cos_angle = np.dot(v1, v2) / norm
            return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        l_knee_angle = angle_3pts(l_hip, l_knee, l_ankle)
        r_knee_angle = angle_3pts(r_hip, r_knee, r_ankle)
        avg_knee_angle = (l_knee_angle + r_knee_angle) / 2

        avg_velocity = np.mean(track["velocities"][-10:]) if track["velocities"] else 0

        # Check for falling (rapid change + becoming horizontal)
        if len(track["landmarks_history"]) >= 5:
            old_lm = track["landmarks_history"][-5]
            old_shoulder_y = (old_lm["left_shoulder"]["y"] + old_lm["right_shoulder"]["y"]) / 2
            shoulder_drop = shoulder_y - old_shoulder_y

            if shoulder_drop > 0.06 and torso_angle > 25:
                return "falling", 0.85

        # Lying down (torso nearly horizontal, low position)
        if torso_angle > 50 and hip_y > 0.55:
            return "lying", 0.9

        # Sitting (bent knees, hips low, torso upright)
        if avg_knee_angle < 130 and hip_y > 0.45 and torso_angle < 35:
            return "sitting", 0.8

        # Crouching/Squatting (very bent knees, torso upright)
        if avg_knee_angle < 100 and torso_angle < 45:
            return "crouching", 0.75

        # Running (high velocity, upright)
        if avg_velocity > 0.035 and torso_angle < 35:
            return "running", 0.8

        # Walking (medium velocity, upright)
        if avg_velocity > 0.012 and torso_angle < 30:
            return "walking", 0.85

        # Bending (torso tilted forward)
        if 25 < torso_angle < 55 and avg_velocity < 0.02:
            return "bending", 0.7

        # Hand raised detection
        if is_visible(l_wrist) and is_visible(l_shoulder):
            if l_wrist["y"] < l_shoulder["y"] - 0.08:
                return "hand_raised", 0.8
        if is_visible(r_wrist) and is_visible(r_shoulder):
            if r_wrist["y"] < r_shoulder["y"] - 0.08:
                return "hand_raised", 0.8

        # Standing (upright, low velocity, straight legs)
        if torso_angle < 25 and avg_velocity < 0.015:
            return "standing", 0.9

        return "standing", 0.5


class AudioAnalyzer:
    """Analyze audio for speech, sounds, and events"""

    def __init__(self):
        self.volume_history = []
        self.speech_duration = 0
        self.silence_duration = 0
        print("[AUDIO] Audio analyzer initialized")

    def analyze(self, audio_data):
        if audio_data is None or len(audio_data) == 0:
            return None

        if isinstance(audio_data, list):
            audio_data = np.array(audio_data, dtype=np.float32)

        volume = float(np.sqrt(np.mean(audio_data ** 2)))
        peak = float(np.max(np.abs(audio_data)))

        self.volume_history.append(volume)
        if len(self.volume_history) > 50:
            self.volume_history = self.volume_history[-50:]

        avg_volume = np.mean(self.volume_history)
        is_sound = volume > AUDIO_THRESHOLD

        if is_sound:
            self.speech_duration += 1
            self.silence_duration = 0
        else:
            self.silence_duration += 1
            if self.silence_duration > 10:
                self.speech_duration = 0

        event = None
        event_confidence = 0.0

        if is_sound:
            if volume > avg_volume * 3 and peak > 0.5:
                event = "loud_noise"
                event_confidence = 0.8
            elif self.speech_duration > 5:
                event = "speech"
                event_confidence = 0.7
            else:
                event = "sound"
                event_confidence = 0.5

        if volume > 0.3 and self.speech_duration > 3:
            event = "shouting"
            event_confidence = 0.75

        return {
            "volume": volume,
            "peak": peak,
            "avg_volume": avg_volume,
            "is_sound": is_sound,
            "speech_detected": self.speech_duration > 5,
            "event": event,
            "event_confidence": event_confidence
        }


class IncidentDetector:
    """Detect and track incidents for multiple people"""

    def __init__(self):
        self.incidents = []
        self.active_incidents = {}
        self.audio_alerts = {"help": 0, "cough": 0, "talking": 0}

    def check_incidents(self, actions, audio):
        new_incidents = []

        for action in actions:
            person_id = action["person_id"]

            # Fall detection
            if action["action"] == "falling" and action["confidence"] > 0.7:
                key = f"fall_{person_id}"
                if key not in self.active_incidents:
                    incident = {
                        "type": "FALL",
                        "description": f"Person {person_id} detected falling",
                        "timestamp": datetime.now().isoformat(),
                        "confidence": action["confidence"],
                        "person_id": person_id
                    }
                    new_incidents.append(incident)
                    self.incidents.append(incident)
                    self.active_incidents[key] = time.time()
                    print(f"[INCIDENT] FALL: Person {person_id}")

            # Person down
            elif action["action"] == "lying" and action["confidence"] > 0.8:
                key = f"lying_{person_id}"
                if key not in self.active_incidents:
                    incident = {
                        "type": "PERSON_DOWN",
                        "description": f"Person {person_id} lying on ground",
                        "timestamp": datetime.now().isoformat(),
                        "confidence": action["confidence"],
                        "person_id": person_id
                    }
                    new_incidents.append(incident)
                    self.incidents.append(incident)
                    self.active_incidents[key] = time.time()
                    print(f"[INCIDENT] PERSON DOWN: Person {person_id}")

            # Hand raised (distress signal)
            elif action["action"] == "hand_raised" and action["confidence"] > 0.7:
                key = f"hand_{person_id}"
                if key not in self.active_incidents:
                    incident = {
                        "type": "HAND_RAISED",
                        "description": f"Person {person_id} raising hand",
                        "timestamp": datetime.now().isoformat(),
                        "confidence": action["confidence"],
                        "person_id": person_id
                    }
                    new_incidents.append(incident)
                    self.incidents.append(incident)
                    self.active_incidents[key] = time.time()

            # Clear incidents if person recovered
            elif action["action"] in ["standing", "walking", "running"]:
                for key in [f"fall_{person_id}", f"lying_{person_id}"]:
                    if key in self.active_incidents:
                        del self.active_incidents[key]

        # Check proximity for fights (between multiple people)
        if len(actions) >= 2:
            for i, a1 in enumerate(actions):
                for a2 in actions[i+1:]:
                    b1, b2 = a1["bbox"], a2["bbox"]
                    c1 = (b1["x"] + b1["width"]/2, b1["y"] + b1["height"]/2)
                    c2 = (b2["x"] + b2["width"]/2, b2["y"] + b2["height"]/2)
                    dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5

                    # Close proximity + high velocity = potential fight
                    if dist < 0.12 and a1["velocity"] > 0.025 and a2["velocity"] > 0.025:
                        key = f"fight_{min(a1['person_id'], a2['person_id'])}_{max(a1['person_id'], a2['person_id'])}"
                        if key not in self.active_incidents:
                            incident = {
                                "type": "FIGHT",
                                "description": f"Altercation between persons {a1['person_id']} and {a2['person_id']}",
                                "timestamp": datetime.now().isoformat(),
                                "confidence": 0.75,
                                "persons": [a1['person_id'], a2['person_id']]
                            }
                            new_incidents.append(incident)
                            self.incidents.append(incident)
                            self.active_incidents[key] = time.time()
                            print(f"[INCIDENT] FIGHT: Persons {a1['person_id']} & {a2['person_id']}")

        # Audio incidents
        if audio and audio.get("event") == "shouting":
            if "shouting" not in self.active_incidents:
                incident = {
                    "type": "AUDIO_ALERT",
                    "description": "Shouting detected",
                    "timestamp": datetime.now().isoformat(),
                    "confidence": audio["event_confidence"]
                }
                new_incidents.append(incident)
                self.incidents.append(incident)
                self.active_incidents["shouting"] = time.time()

        # Cleanup old incidents (expire after 3 seconds)
        to_remove = []
        current_time = time.time()
        for key, val in self.active_incidents.items():
            if isinstance(val, float) and current_time - val > 3.0:
                to_remove.append(key)
        for key in to_remove:
            del self.active_incidents[key]

        return new_incidents

    def handle_audio_alert(self, alert_type, description):
        self.audio_alerts[alert_type] = self.audio_alerts.get(alert_type, 0) + 1

        incident = None
        if alert_type == "help":
            incident = {
                "type": "HELP_CALL",
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.9,
                "priority": "HIGH"
            }
            print(f"[ALERT] HELP: {description}")
        elif alert_type == "cough":
            incident = {
                "type": "COUGH",
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.7,
                "priority": "MEDIUM"
            }
            print(f"[ALERT] COUGH: {description}")
        elif alert_type == "talking":
            print(f"[AUDIO] Speech detected")
            return None

        if incident:
            self.incidents.append(incident)
            return incident
        return None

    def get_stats(self):
        return {
            "total_incidents": len(self.incidents),
            "falls": len([i for i in self.incidents if i["type"] == "FALL"]),
            "fights": len([i for i in self.incidents if i["type"] == "FIGHT"]),
            "audio_alerts": len([i for i in self.incidents if i["type"] in ["AUDIO_ALERT", "HELP_CALL", "COUGH"]]),
            "persons_down": len([i for i in self.incidents if i["type"] == "PERSON_DOWN"]),
            "help_calls": self.audio_alerts.get("help", 0),
            "coughs": self.audio_alerts.get("cough", 0)
        }


# Global instances
detector = MultiPersonPoseDetector()
action_recognizer = ActionRecognizer()
audio_analyzer = AudioAnalyzer()
incident_detector = IncidentDetector()

async def process_client(websocket):
    print(f"[CONNECTION] Client connected: {websocket.remote_address}")

    frame_count = 0
    fps_start = time.time()

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                response = {"type": "update", "timestamp": datetime.now().isoformat()}

                # Handle audio alerts
                if data.get("type") == "audio_alert":
                    alert_type = data.get("alert_type")
                    description = data.get("description", "")
                    incident = incident_detector.handle_audio_alert(alert_type, description)
                    if incident:
                        response["incidents"] = [incident]
                        response["stats"] = {**incident_detector.get_stats(), "people": 0}
                    await websocket.send(json.dumps(response))
                    continue

                # Process video frame
                if data.get("type") == "frame" and data.get("data"):
                    # Include camera ID in response
                    cam_id = data.get("cam_id", 0)
                    cam_name = data.get("cam_name", "CAM 1")
                    response["cam_id"] = cam_id
                    response["cam_name"] = cam_name

                    img_data = base64.b64decode(data["data"].split(",")[1])
                    nparr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        detections = detector.detect(frame)
                        actions = action_recognizer.analyze(detections)

                        # Remove landmarks from response (too large)
                        for det in detections:
                            if "landmarks" in det:
                                del det["landmarks"]

                        response["detections"] = detections
                        response["actions"] = actions
                        response["people_count"] = len(detections)

                        frame_count += 1

                # Process audio
                audio_result = None
                if data.get("audio"):
                    audio_result = audio_analyzer.analyze(data["audio"])
                    if audio_result:
                        response["audio"] = audio_result

                # Check incidents
                actions = response.get("actions", [])
                incidents = incident_detector.check_incidents(actions, audio_result)
                if incidents:
                    response["incidents"] = incidents

                # Stats
                response["stats"] = {
                    **incident_detector.get_stats(),
                    "people": response.get("people_count", 0)
                }

                # Log FPS and detections
                if frame_count % 30 == 0 and frame_count > 0:
                    elapsed = time.time() - fps_start
                    fps = frame_count / elapsed
                    people = response.get("people_count", 0)
                    actions_list = response.get("actions", [])
                    actions_str = ", ".join([f"P{a['person_id']}:{a['action']}" for a in actions_list])
                    print(f"[STREAM] FPS: {fps:.1f}, People: {people}, Actions: [{actions_str}]")

                await websocket.send(json.dumps(response))

            except json.JSONDecodeError:
                print("[ERROR] Invalid JSON")
            except Exception as e:
                print(f"[ERROR] {e}")
                import traceback
                traceback.print_exc()

    except websockets.exceptions.ConnectionClosed:
        print(f"[CONNECTION] Client disconnected")


from websockets.datastructures import Headers
from websockets.http11 import Response

async def health_check(connection, request):
    """Handle direct browser visits to accept the SSL certificate"""
    if request.headers.get("Upgrade", "").lower() != "websocket":
        # Regular HTTP request - serve a simple page for cert acceptance
        body = b"""<!DOCTYPE html>
<html><head><title>AI Backend</title></head>
<body style="font-family:system-ui;display:flex;justify-content:center;align-items:center;height:100vh;margin:0;background:#0f1419;color:#22c55e;">
<div style="text-align:center">
<h1>Certificate Accepted</h1>
<p>You can close this tab and return to the app.</p>
</div>
</body></html>"""
        return Response(200, "OK", Headers([("Content-Type", "text/html")]), body)
    return None  # Continue with WebSocket handshake


async def main():
    print("=" * 60)
    print("SF Security Camera - Multi-Person Pose Detection")
    print("=" * 60)
    print("Features:")
    print("  - YOLOv8-Pose multi-person detection")
    print("  - Per-person tracking with consistent IDs")
    print("  - Actions: standing, walking, running, sitting, crouching,")
    print("             falling, lying, bending, hand_raised")
    print("  - Audio: speech, shouting, help keyword, cough")
    print("  - Incidents: falls, fights, person down, distress")
    print("=" * 60)

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    cert_path = os.path.join(os.path.dirname(__file__), "cert.pem")
    key_path = os.path.join(os.path.dirname(__file__), "key.pem")

    if os.path.exists(cert_path) and os.path.exists(key_path):
        ssl_context.load_cert_chain(cert_path, key_path)
        print("[SERVER] SSL enabled")
        server = await websockets.serve(process_client, "0.0.0.0", 8765, ssl=ssl_context, process_request=health_check)
    else:
        print("[SERVER] No SSL (development mode)")
        server = await websockets.serve(process_client, "0.0.0.0", 8765, process_request=health_check)

    print("[SERVER] WebSocket listening on port 8765")
    print("=" * 60)

    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
