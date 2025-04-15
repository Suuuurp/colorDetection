# detection_utils.py
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
# Pose Landmark Enum for easier indexing
lm_pose_enum = mp_pose.PoseLandmark


# Function to run both detectors
def detect_face_and_pose(rgb_frame):
    """Detects both face landmarks and pose landmarks."""
    face_landmarks = None
    pose_landmarks = None

    # Process for Face Mesh
    try:
        results_face = face_mesh.process(rgb_frame)
        if results_face.multi_face_landmarks:
            # Assuming only one face due to FaceMesh config
            face_landmarks = results_face.multi_face_landmarks[0]
    except Exception as e:
        print(f"Error during Face Mesh processing: {e}")


    # Process for Pose
    try:
        results_pose = pose.process(rgb_frame)
        if results_pose.pose_landmarks:
            pose_landmarks = results_pose.pose_landmarks
    except Exception as e:
        print(f"Error during Pose processing: {e}")


    return face_landmarks, pose_landmarks


# --- ROI Estimation Functions ---

def get_face_roi_from_mesh(image_shape, landmarks):
    """Calculates a bounding box around the face using Face Mesh landmarks."""
    if not landmarks: return None
    h, w = image_shape[:2]
    try:
        all_x = [lm.x * w for lm in landmarks.landmark]
        all_y = [lm.y * h for lm in landmarks.landmark]
        # Filter out potential invalid coords if landmarks are bad
        all_x = [val for val in all_x if 0 <= val <= w]
        all_y = [val for val in all_y if 0 <= val <= h]
        if not all_x or not all_y: return None # No valid points

        x = int(min(all_x))
        y = int(min(all_y))
        w_box = int(max(all_x) - x)
        h_box = int(max(all_y) - y)

        padding = 5 # pixels
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + w_box + padding)
        y2 = min(h, y + h_box + padding)

        final_w = x2-x1
        final_h = y2-y1
        if final_w <= 0 or final_h <= 0: return None

        return (x1, y1, final_w, final_h) # Return x, y, w, h format
    except Exception as e:
        print(f"Error getting face ROI from mesh: {e}")
        return None


def get_hair_roi_from_mesh(image_shape, face_landmarks):
    """Estimates a hair region (forehead hairline) based on Face Mesh landmarks."""
    if not face_landmarks: return None

    h, w = image_shape[:2]
    lm = face_landmarks.landmark

    try:
        forehead_indices = [10, 109, 338, 67, 297]
        points_x = []
        points_y = []
        for index in forehead_indices:
            if index < len(lm):
                 # Ensure landmark coordinates are valid before using
                 lx, ly = lm[index].x * w, lm[index].y * h
                 if 0 <= lx <= w and 0 <= ly <= h:
                     points_x.append(lx)
                     points_y.append(ly)
                 else: return None # Invalid coordinate encountered
            else: return None # Index out of bounds

        if not points_x or not points_y: return None # No valid points collected

        forehead_min_x = min(points_x)
        forehead_max_x = max(points_x)
        forehead_min_y = min(points_y)
        forehead_width = forehead_max_x - forehead_min_x

        roi_height = int(forehead_width * 0.3); roi_height = max(10, roi_height)
        roi_y = int(forehead_min_y - roi_height * 0.9); roi_y = max(0, roi_y)
        roi_x = int(forehead_min_x)
        roi_w = int(forehead_width)

        x1 = max(0, roi_x); y1 = max(0, roi_y)
        x2 = min(w, roi_x + roi_w); y2 = min(h, roi_y + roi_height)
        final_w = x2 - x1; final_h = y2 - y1

        if final_w < 10 or final_h < 10: return None

        return (x1, y1, final_w, final_h)

    except Exception as e:
        print(f"Error getting hair ROI: {e}")
        return None


def get_estimated_body_part_roi(image_shape, pose_landmarks, part_name):
    """Estimates an ROI for a body part based on MediaPipe Pose landmarks."""
    if not pose_landmarks: return None

    h, w = image_shape[:2]
    lm = pose_landmarks.landmark
    min_visibility = 0.6

    required_indices_map = {
        'torso': [lm_pose_enum.LEFT_SHOULDER, lm_pose_enum.RIGHT_SHOULDER, lm_pose_enum.LEFT_HIP, lm_pose_enum.RIGHT_HIP],
        'left_arm': [lm_pose_enum.LEFT_SHOULDER, lm_pose_enum.LEFT_ELBOW, lm_pose_enum.LEFT_WRIST],
        'right_arm': [lm_pose_enum.RIGHT_SHOULDER, lm_pose_enum.RIGHT_ELBOW, lm_pose_enum.RIGHT_WRIST]
    }

    if part_name not in required_indices_map: return None
    required_indices = required_indices_map[part_name]

    points_x = []
    points_y = []
    try:
        for index in required_indices:
            landmark = lm[index] # Use the enum value as index
            if landmark.visibility > min_visibility:
                lx, ly = landmark.x * w, landmark.y * h
                # Check coordinates are valid
                if 0 <= lx <= w and 0 <= ly <= h:
                    points_x.append(lx)
                    points_y.append(ly)
                else: return None # Invalid coord
            else: return None # Landmark not visible enough

        if not points_x or not points_y: return None

        x_min = int(min(points_x)); y_min = int(min(points_y))
        x_max = int(max(points_x)); y_max = int(max(points_y))

        padding_x = int((x_max - x_min) * 0.15)
        padding_y = int((y_max - y_min) * 0.15)

        if 'arm' in part_name:
             padding_x = max(10, padding_x); padding_y = max(5, padding_y)
        else: # Torso
             padding_x = max(5, padding_x); padding_y = max(5, padding_y)

        x1 = max(0, x_min - padding_x); y1 = max(0, y_min - padding_y)
        x2 = min(w, x_max + padding_x); y2 = min(h, y_max + padding_y)
        roi_w = x2 - x1; roi_h = y2 - y1

        if roi_w < 10 or roi_h < 10: return None

        return (x1, y1, roi_w, roi_h)

    except IndexError: # Should not happen with enum but good practice
        print(f"Landmark index error for {part_name}.")
        return None
    except Exception as e:
        print(f"Error getting estimated {part_name} ROI: {e}")
        return None