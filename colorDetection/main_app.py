# main_app.py
import cv2
import numpy as np
import color_utils as cu
import detection_utils as du
import csv
import datetime
import os

import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.pose as mp_pose

# --- Configuration ---
WEBCAM_ID = 0
WINDOW_NAME = "Advanced Color Detection"
DISPLAY_INFO_HEIGHT = 180
LOG_FILE_PATH = "color_log.csv"
HSV_MATCH_THRESHOLD_PERCENT = 25 # % of pixels needed to match an HSV range

# --- NEW: HSV Color Range Definitions (OpenCV H=0-179, S=0-255, V=0-255) ---
# These ranges often need tuning based on lighting and camera!
HSV_RANGES = {
    # Color Name: [lower_bound, upper_bound]
    # Note: Red wraps around 0/180
    "Red":    [[np.array([0, 70, 50]), np.array([10, 255, 255])],
               [np.array([170, 70, 50]), np.array([180, 255, 255])]],
    "Orange": [[np.array([11, 100, 100]), np.array([25, 255, 255])]],
    "Yellow": [[np.array([26, 100, 100]), np.array([34, 255, 255])]],
    "Green":  [[np.array([35, 50, 50]), np.array([85, 255, 255])]],
    "Cyan":   [[np.array([86, 50, 50]), np.array([100, 255, 255])]], # Added Cyan
    "Blue":   [[np.array([101, 50, 50]), np.array([130, 255, 255])]],
    "Purple": [[np.array([131, 50, 50]), np.array([160, 255, 255])]],
    # Pink can be tricky - often desaturated red/purple range
    "Pink":   [[np.array([161, 40, 100]), np.array([175, 255, 255])]], # Adjust S/V ranges
    # Non-Hue specific colors (based on S and V)
    "Black":  [[np.array([0, 0, 0]), np.array([180, 255, 50])]],
    "Gray":   [[np.array([0, 0, 51]), np.array([180, 50, 200])]], # Low Saturation, Mid Value
    "White":  [[np.array([0, 0, 201]), np.array([180, 50, 255])]], # Low Saturation, High Value
}
# Order matters for checks like Black/Gray/White vs specific hues if overlap exists
HSV_CHECK_ORDER = ["Black", "White", "Gray", "Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple", "Pink"]
# --- End HSV Definitions ---


# --- Global variables ---
selecting_roi = False
roi_start_point = None
roi_end_point = None
selected_roi = None
draw_pose = False
last_logged_data = {}

# --- Function Definitions ---

def select_roi_callback(event, x, y, flags, param):
    """Handles mouse events for selecting ROI."""
    global selecting_roi, roi_start_point, roi_end_point, selected_roi
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting_roi = True; roi_start_point = (x, y); roi_end_point = (x, y); selected_roi = None
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting_roi: roi_end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        selecting_roi = False; x1, y1 = roi_start_point; x2, y2 = roi_end_point
        roi_x, roi_y = min(x1, x2), min(y1, y2); roi_w, roi_h = abs(x1 - x2), abs(y1 - y2)
        if roi_w > 5 and roi_h > 5: selected_roi = (roi_x, roi_y, roi_w, roi_h)
        else: selected_roi = None

def rgb_to_hex(rgb):
    """Converts an RGB tuple (r, g, b) to a hex string #RRGGBB."""
    if rgb is None or len(rgb) != 3: return "N/A"
    try:
        r, g, b = [max(0, min(255, int(val))) for val in rgb]
        return "#{:02x}{:02x}{:02x}".format(r, g, b)
    except Exception: return "N/A"

def initialize_log_file():
    """Creates log file with header if it doesn't exist."""
    if not os.path.exists(LOG_FILE_PATH):
        try:
            with open(LOG_FILE_PATH, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Added 'Detection Method' column
                writer.writerow(['Timestamp', 'Part', 'Detected Color Name', 'Matched Hex', 'Average BGR', 'Detection Method'])
            print(f"Created log file: {LOG_FILE_PATH}")
        except Exception as e:
            print(f"Error creating log file: {e}")


def log_color_data(timestamp, part, color_name, hex_value, avg_bgr, method):
    """Appends a row of detected color data to the CSV log file."""
    global last_logged_data
    # Added method to the log entry
    current_log_entry = (timestamp.strftime('%Y-%m-%d %H:%M:%S'), part, color_name, hex_value, str(avg_bgr), method)
    try:
        with open(LOG_FILE_PATH, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(current_log_entry)
    except Exception as e:
        print(f"Error writing to log file: {e}")

# --- NEW: Helper function to check HSV ranges ---
def check_hsv_color(roi_bgr_frame):
    """
    Checks an ROI against predefined HSV ranges.
    Returns the name of the first matching color (e.g., 'Red (HSV)'), or None.
    """
    if roi_bgr_frame is None or roi_bgr_frame.size == 0:
        return None

    hsv_roi = cv2.cvtColor(roi_bgr_frame, cv2.COLOR_BGR2HSV)
    total_pixels = hsv_roi.size / 3 # Size includes H, S, V channels

    if total_pixels == 0: return None

    for color_name in HSV_CHECK_ORDER:
        if color_name in HSV_RANGES:
            combined_mask = None
            # Iterate through potentially multiple ranges for a color (like Red)
            for hsv_range in HSV_RANGES[color_name]:
                lower_bound = hsv_range[0]
                upper_bound = hsv_range[1]
                mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, mask) # Combine masks for wrap-around

            if combined_mask is not None:
                match_count = cv2.countNonZero(combined_mask)
                match_percentage = (match_count * 100) / total_pixels
                if match_percentage >= HSV_MATCH_THRESHOLD_PERCENT:
                    return f"{color_name} (HSV)" # Return name indicating HSV match

    return None # No significant HSV range match found
# --- End HSV Helper ---


# --- Main Application Logic ---
def main():
    global selected_roi, draw_pose

    initialize_log_file()

    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened(): print(f"Error: Could not open webcam {WEBCAM_ID}."); return

    ret, frame = cap.read()
    if not ret: print("Error: Failed to get initial frame."); cap.release(); return
    frame_h, frame_w = frame.shape[:2]

    display_h = frame_h + DISPLAY_INFO_HEIGHT
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, select_roi_callback)

    print("Application started. Press 'q' to quit.")
    print(" - Press 'p' to toggle Pose landmarks drawing.")
    print(" - Press 'l' to log currently detected colors to CSV.")
    print("Detecting face/pose. If none detected, click/drag for manual ROI.")

    current_detections_for_log = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        display_canvas = np.zeros((display_h, frame_w, 3), dtype=np.uint8)
        display_canvas[0:frame_h, 0:frame_w] = frame

        detected_info = []
        current_detections_for_log.clear()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_landmarks, pose_landmarks = du.detect_face_and_pose(rgb_frame) # Keep writeable flags inside function

        if draw_pose and pose_landmarks:
            mp_drawing.draw_landmarks(
                display_canvas[0:frame_h, 0:frame_w], pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        person_detected = face_landmarks or pose_landmarks

        if person_detected:
            detected_info.append("Mode: Human Detected")
            selected_roi = None

            parts_to_check = { # Using tuple for args now
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
                     # Check for valid ROI dimensions before proceeding
                     if w_roi > 0 and h_roi > 0:
                        cv2.rectangle(display_canvas[0:frame_h, 0:frame_w], (x, y), (x + w_roi, y + h_roi), details["color"], 1)
                        # --- Get ROI Frame & Check HSV First ---
                        roi_frame = frame[y:y+h_roi, x:x+w_roi]
                        hsv_color_name = check_hsv_color(roi_frame)
                        avg_bgr = cu.get_average_color(roi_frame, (0, 0, w_roi, h_roi)) # Get avg BGR anyway for info

                        if hsv_color_name: # HSV Match!
                            color_name = hsv_color_name
                            hex_value = "N/A (HSV)" # No single hex for range
                            method = "HSV Range"
                        elif avg_bgr: # No HSV match, fallback to average BGR + nearest name
                            color_name, color_rgb = cu.get_color_name_and_rgb(avg_bgr[::-1]) # Query with RGB
                            hex_value = rgb_to_hex(color_rgb)
                            method = "Avg BGR"
                        else: # Failed to get average color
                            color_name = "Color Error"
                            hex_value = "N/A"
                            method = "Error"

                        # Display and Log
                        if color_name != "Color Error":
                            display_text = f" {part_name}: {color_name} ({hex_value}) (AvgBGR:{avg_bgr})"
                            detected_info.append(display_text)
                            current_detections_for_log.append({'part': part_name, 'name': color_name, 'hex': hex_value, 'bgr': avg_bgr, 'method': method})
                        else:
                            detected_info.append(f" {part_name}: Color Error")
                     # else: print(f"Skipping {part_name}: Invalid ROI dimensions {w_roi}x{h_roi}") # Debug
        else:
            # --- ROI Selection Mode ---
            detected_info.append("Mode: ROI Selection")
            if selected_roi:
                x, y, w_roi, h_roi = selected_roi
                if w_roi > 0 and h_roi > 0:
                    cv2.rectangle(display_canvas[0:frame_h, 0:frame_w], (x, y), (x + w_roi, y + h_roi), (0, 255, 0), 2)
                    # --- Check HSV First for Manual ROI ---
                    roi_frame = frame[y:y+h_roi, x:x+w_roi]
                    hsv_color_name = check_hsv_color(roi_frame)
                    avg_bgr_roi = cu.get_average_color(roi_frame, (0, 0, w_roi, h_roi))

                    if hsv_color_name: # HSV Match
                        color_name_roi = hsv_color_name
                        hex_value_roi = "N/A (HSV)"
                        method = "HSV Range"
                    elif avg_bgr_roi: # Fallback
                        color_name_roi, color_rgb_roi = cu.get_color_name_and_rgb(avg_bgr_roi[::-1])
                        hex_value_roi = rgb_to_hex(color_rgb_roi)
                        method = "Avg BGR"
                    else: # Error
                        color_name_roi = "Color Error"
                        hex_value_roi = "N/A"
                        method = "Error"

                    if color_name_roi != "Color Error":
                        display_text = f" ROI: {color_name_roi} ({hex_value_roi}) (AvgBGR:{avg_bgr_roi})"
                        detected_info.append(display_text)
                        current_detections_for_log.append({'part': 'Manual ROI', 'name': color_name_roi, 'hex': hex_value_roi, 'bgr': avg_bgr_roi, 'method': method})
                    else:
                        detected_info.append(" ROI Color: N/A")
                # else: print("Skipping Manual ROI: Invalid dimensions") # Debug
            elif selecting_roi and roi_start_point and roi_end_point:
                 cv2.rectangle(display_canvas[0:frame_h, 0:frame_w], roi_start_point, roi_end_point, (0, 0, 255), 2)
                 detected_info.append(" Selecting...")
            else:
                 detected_info.append(" Click and drag mouse to select ROI.")


        # --- Display Status Info ---
        pose_status = "ON" if draw_pose else "OFF"
        detected_info.insert(0, f"Pose Landmarks ('p'): {pose_status} | Log ('l') | HSV Thresh: {HSV_MATCH_THRESHOLD_PERCENT}%") # Show threshold

        # --- Display Detected Information Text ---
        y_offset = frame_h + 20; line_height = 18
        for i, line in enumerate(detected_info):
             current_y = y_offset + i * line_height
             if current_y + line_height > display_h: break
             cv2.putText(display_canvas, line, (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Show final canvas ---
        cv2.imshow(WINDOW_NAME, display_canvas)

        # --- Keypress Handling ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): draw_pose = not draw_pose; print(f"Pose drawing toggled {'ON' if draw_pose else 'OFF'}")
        elif key == ord('l'):
            timestamp = datetime.datetime.now()
            if not current_detections_for_log: print("Nothing to log currently.")
            else:
                 print(f"Logging {len(current_detections_for_log)} detection(s)...")
                 for detection in current_detections_for_log:
                      log_color_data(timestamp, detection['part'], detection['name'], detection['hex'], detection['bgr'], detection['method'])

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")

# --- Script Execution Start ---
if __name__ == "__main__":
    main()