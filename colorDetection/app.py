# app.py
# FINAL VERSION - No Skeleton Logic

import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
import datetime
import csv
import os
import traceback

from flask import Flask, render_template, request, jsonify

# Import your existing utilities
import color_utils as cu
import detection_utils as du
# mediapipe import might still be needed if detection_utils uses its enums internally, keep it for safety
import mediapipe as mp

# Initialize the Flask application
app = Flask(__name__)

LOG_FILE_PATH = "color_log.csv"

# --- HSV Color Range Definitions ---
HSV_RANGES = {
    "Red":    [[np.array([0, 70, 50]), np.array([10, 255, 255])],
               [np.array([170, 70, 50]), np.array([180, 255, 255])]],
    "Orange": [[np.array([11, 100, 100]), np.array([25, 255, 255])]],
    "Yellow": [[np.array([26, 100, 100]), np.array([34, 255, 255])]],
    "Green":  [[np.array([35, 50, 50]), np.array([85, 255, 255])]],
    "Cyan":   [[np.array([86, 50, 50]), np.array([100, 255, 255])]],
    "Blue":   [[np.array([101, 50, 50]), np.array([130, 255, 255])]],
    "Purple": [[np.array([131, 50, 50]), np.array([160, 255, 255])]],
    "Pink":   [[np.array([161, 40, 100]), np.array([175, 255, 255])]],
    "Black":  [[np.array([0, 0, 0]), np.array([180, 255, 50])]],
    "Gray":   [[np.array([0, 0, 51]), np.array([180, 50, 200])]],
    "White":  [[np.array([0, 0, 201]), np.array([180, 50, 255])]],
    "venu_color": [[np.array([0, 0, 62]), np.array([179, 255, 255])]],
    "venu_phone": [[np.array([0, 0, 62]), np.array([179, 255, 255])]]
}
HSV_CHECK_ORDER = ["Black", "White", "Gray", "Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple", "Pink", "venu_color", "venu_phone"]
HSV_MATCH_THRESHOLD_PERCENT = 25

# --- Helper function to check HSV ranges ---
def check_hsv_color(roi_bgr_frame):
    if roi_bgr_frame is None or roi_bgr_frame.size == 0: return None
    try:
        hsv_roi = cv2.cvtColor(roi_bgr_frame, cv2.COLOR_BGR2HSV)
        total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]
        if total_pixels == 0: return None
        for color_name in HSV_CHECK_ORDER:
            if color_name in HSV_RANGES:
                combined_mask = None
                for hsv_range in HSV_RANGES[color_name]:
                    mask = cv2.inRange(hsv_roi, hsv_range[0], hsv_range[1])
                    if combined_mask is None: combined_mask = mask
                    else: combined_mask = cv2.bitwise_or(combined_mask, mask)
                if combined_mask is not None:
                    match_count = cv2.countNonZero(combined_mask)
                    match_percentage = (match_count * 100) / total_pixels if total_pixels > 0 else 0
                    if match_percentage >= HSV_MATCH_THRESHOLD_PERCENT:
                        return color_name # Return base name like "Black", "Red"
    except Exception as e: print(f"Error during HSV check: {e}")
    return None

# --- Other Helper Functions ---
def tuple_to_serializable(t):
    if t is None: return None
    try: return tuple(int(c) for c in t)
    except (TypeError, ValueError): return None

def rgb_to_hex(rgb):
    rgb_serializable = tuple_to_serializable(rgb)
    if rgb_serializable is None or len(rgb_serializable) != 3: return "N/A"
    try:
        r, g, b = [max(0, min(255, val)) for val in rgb_serializable]
        return "#{:02x}{:02x}{:02x}".format(r, g, b)
    except Exception: return "N/A"

# --- Logging Functions ---
def initialize_log_file():
    if not os.path.exists(LOG_FILE_PATH):
        try:
            with open(LOG_FILE_PATH, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Part', 'Detected Color Name', 'Matched Hex', 'Average BGR', 'Detection Method'])
            print(f"Created log file: {LOG_FILE_PATH}")
        except Exception as e: print(f"Error creating log file: {e}")

def log_color_data(log_entry):
    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(LOG_FILE_PATH, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([ timestamp_str, log_entry.get('part'), log_entry.get('name'),
                log_entry.get('hex'), str(log_entry.get('bgr')), log_entry.get('method') ])
    except Exception as e: print(f"Error writing to log file: {e}")

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if not request.json: return jsonify({"error": "Invalid request"}), 400
    image_data = request.json.get('image_data')
    manual_roi_coords = request.json.get('manual_roi')
    if not image_data: return jsonify({"error": "No image data found"}), 400

    try:
        base64_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(base64_data)
        image_pil = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        frame_h, frame_w = frame.shape[:2]

        detection_results = []
        mode = "ROI Selection" # Default mode

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Only get landmarks needed for box detection
        face_landmarks, pose_landmarks = du.detect_face_and_pose(rgb_frame)
        person_detected = bool(face_landmarks or pose_landmarks)

        # Centralized function to process an ROI
        def analyze_roi(roi_frame, part_name_for_log):
            # ... (analyze_roi function remains the same as previous version) ...
            hsv_color_name_base = check_hsv_color(roi_frame)
            avg_bgr = cu.get_average_color(roi_frame, (0, 0, roi_frame.shape[1], roi_frame.shape[0]))
            color_name, hex_value, method, avg_bgr_serializable = "Color Error", "N/A", "Error", None
            color_rgb_match = None
            if hsv_color_name_base:
                color_name, hex_value, method = f"{hsv_color_name_base} (HSV)", "N/A (HSV)", "HSV Range"
                if avg_bgr: avg_bgr_serializable = tuple_to_serializable(avg_bgr)
            elif avg_bgr:
                color_name_base, color_rgb_match = cu.get_color_name_and_rgb(avg_bgr[::-1])
                color_name, hex_value, method = color_name_base, rgb_to_hex(color_rgb_match), "Avg BGR"
                avg_bgr_serializable = tuple_to_serializable(avg_bgr)
            return { "part": part_name_for_log, "color_name": color_name, "hex": hex_value,
                     "avg_bgr": avg_bgr_serializable, "method": method }

        if not person_detected and manual_roi_coords:
            # --- Process Manual ROI ---
            mode = "ROI Selection"
            x, y = int(manual_roi_coords.get('x', 0)), int(manual_roi_coords.get('y', 0))
            w_roi, h_roi = int(manual_roi_coords.get('w', 0)), int(manual_roi_coords.get('h', 0))
            if w_roi > 0 and h_roi > 0:
                x1, y1 = max(0, x), max(0, y); x2, y2 = min(frame_w, x + w_roi), min(frame_h, y + h_roi)
                clamped_w, clamped_h = x2 - x1, y2 - y1
                if clamped_w > 0 and clamped_h > 0:
                    roi_frame = frame[y1:y2, x1:x2]
                    analysis_result = analyze_roi(roi_frame, "Manual ROI")
                    analysis_result["roi"] = tuple_to_serializable((x, y, w_roi, h_roi))
                    analysis_result["draw_color"] = tuple_to_serializable((0, 255, 0)) # Green
                    detection_results.append(analysis_result)

        elif person_detected:
            # --- Process Body Parts ---
            mode = "Human Detected"
            # NO need to serialize pose_landmarks if not drawing skeleton
            # pose_landmarks_serializable = landmark_list_to_serializable(...)

            parts_to_check = { # Define parts
                "Face": {"getter": du.get_face_roi_from_mesh, "args": (face_landmarks,), "color": (0, 255, 255)},
                "Hair": {"getter": du.get_hair_roi_from_mesh, "args": (face_landmarks,), "color": (255, 0, 255)},
                "Torso": {"getter": du.get_estimated_body_part_roi, "args": (pose_landmarks, 'torso'), "color": (255, 0, 0)},
                "Left Arm": {"getter": du.get_estimated_body_part_roi, "args": (pose_landmarks, 'left_arm'), "color": (0, 255, 0)},
                "Right Arm": {"getter": du.get_estimated_body_part_roi, "args": (pose_landmarks, 'right_arm'), "color": (0, 0, 255)}
            }

            for part_name, details in parts_to_check.items():
                required_landmarks = details["args"][0]
                if required_landmarks is None: continue
                roi_rect = details["getter"](frame.shape, *details["args"])
                if roi_rect:
                    x, y, w_roi, h_roi = roi_rect
                    if w_roi > 0 and h_roi > 0:
                        roi_frame = frame[int(y):int(y+h_roi), int(x):int(x+w_roi)]
                        analysis_result = analyze_roi(roi_frame, part_name)
                        analysis_result["roi"] = tuple_to_serializable(roi_rect)
                        analysis_result["draw_color"] = tuple_to_serializable(details["color"])
                        detection_results.append(analysis_result)
        # else: mode remains "ROI Selection"

        # --- Prepare Response (No pose_landmarks needed now) ---
        response_data = {
            "mode": mode,
            "detections": detection_results
            # "pose_landmarks": None # Explicitly set to None or remove key
        }
        # --- DEBUG PRINT (Keep this active) ---
        print("--- Sending Response ---")
        try: print(json.dumps(response_data, indent=2, default=str))
        except Exception as dump_error: print(f"Could not dump response_data: {dump_error}\n{response_data}")
        print("------------------------")
        # --- END DEBUG PRINT ---

        return jsonify(response_data)

    except Exception as e:
        print(f"Error processing frame in /process_frame: {e}")
        traceback.print_exc()
        return jsonify({"error": "Processing error occurred", "details": str(e)}), 500

@app.route('/log_detections', methods=['POST'])
def log_detections_route():
    # ... (logging route remains the same) ...
    if not request.json or 'detections' not in request.json: return jsonify({"error": "No detection data provided"}), 400
    detections_to_log = request.json['detections']
    log_count = 0
    if isinstance(detections_to_log, list):
        for detection in detections_to_log:
            if isinstance(detection, dict) and 'part' in detection: log_color_data(detection); log_count += 1
        print(f"Logged {log_count} detection(s) from frontend request.")
        return jsonify({"message": f"Logged {log_count} detections."}), 200
    else: return jsonify({"error": "Invalid data format for detections"}), 400

# --- Boilerplate to Run ---
if __name__ == '__main__':
    initialize_log_file()
    print("Starting Flask server... Go to http://127.0.0.1:5000 or http://localhost:5000")
    try: from PIL import Image
    except ImportError: print("\n*** Pillow library not found. Install: pip install Pillow ***\n"); exit()
    if cu.color_kdtree is None and not (hasattr(cu, '_CSS3_NAMES_TO_HEX') and cu._CSS3_NAMES_TO_HEX):
        print("\n*** WARNING: Color database (CSV/webcolors) failed to load. ***\n")
    app.run(host='0.0.0.0', port=5000, debug=True)