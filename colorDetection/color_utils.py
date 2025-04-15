# color_utils.py
import pandas as pd
import numpy as np
from scipy.spatial import KDTree

# --- Direct Import Attempt (Suitable for older webcolors versions like 1.13) ---
try:
    import webcolors
    # Access constants directly using the module name
    # In v1.13, these should exist directly under 'webcolors'
    _CSS3_NAMES_TO_HEX = webcolors.CSS3_NAMES_TO_HEX
    _hex_to_rgb = webcolors.hex_to_rgb
    print("INFO: Attempting to access webcolors constants directly...")
except ImportError:
    print("ERROR: Failed to import the 'webcolors' module.")
    _CSS3_NAMES_TO_HEX = {}
    _hex_to_rgb = lambda x: (0, 0, 0) # Dummy function
except AttributeError as e:
    print(f"ERROR: 'webcolors' imported, but failed to find constants directly: {e}")
    _CSS3_NAMES_TO_HEX = {}
    _hex_to_rgb = lambda x: (0, 0, 0) # Dummy function
# --- End Direct Import Attempt ---


# Global variables
color_data = None
color_kdtree = None
color_names = None
color_rgb_values = None

# --- MODIFIED: Changed default CSV path ---
def load_colors_from_csv(csv_path="common_colors.csv"):
    """Loads color data from the CSV file."""
    global color_data, color_kdtree, color_names, color_rgb_values
    try:
        color_data = pd.read_csv(csv_path)
        color_data.columns = [col.strip() for col in color_data.columns]
        if not all(col in color_data.columns for col in ['Color Name', 'R', 'G', 'B']):
            raise ValueError("CSV must contain 'Color Name', 'R', 'G', 'B' columns.")

        rgb_values = color_data[['R', 'G', 'B']].values
        color_kdtree = KDTree(rgb_values)
        color_names = color_data['Color Name'].tolist()
        color_rgb_values = rgb_values
        print(f"Loaded {len(color_names)} colors from {csv_path}") # Will now print 'common_colors.csv'
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please create it or check the path.")
        print("Using webcolors fallback for color naming (if available).")
        color_data, color_kdtree, color_names, color_rgb_values = None, None, None, None
    except Exception as e:
        print(f"Error loading colors from CSV: {e}")
        print("Using webcolors fallback for color naming (if available).")
        color_data, color_kdtree, color_names, color_rgb_values = None, None, None, None
# --- End MODIFIED ---


def get_color_name_and_rgb(rgb_tuple_query):
    """
    Finds the closest color name and its corresponding RGB value using KDTree fallback.
    Returns (color_name, (R, G, B)) or (fallback_name, None) or (error_name, None).
    """
    # --- Primary Method: Use KDTree (now defaults to common_colors.csv) ---
    if color_kdtree is not None and color_names is not None and color_rgb_values is not None:
        try:
            distance, index = color_kdtree.query(rgb_tuple_query)
            matched_name = color_names[index]
            matched_rgb = tuple(color_rgb_values[index])
            return matched_name, matched_rgb
        except Exception as e:
            print(f"Error querying KDTree: {e}")
            pass # Fall through to webcolors

    # --- Fallback Method: Use webcolors (if KDTree failed or CSV didn't load) ---
    # Check if the constants were loaded correctly earlier
    if not _CSS3_NAMES_TO_HEX:
        # print("Webcolors fallback unavailable (constants not loaded).") # Optional debug
        return "Color DB Error", None

    try:
        min_dist = float('inf')
        closest_name = "Unknown"
        closest_rgb = None
        requested_rgb = np.array(rgb_tuple_query)

        # Use the constants loaded via direct access
        for name, hex_code in _CSS3_NAMES_TO_HEX.items():
            # webcolors v1.x hex codes don't always have '#', add it
            if not hex_code.startswith('#'):
                 hex_code_corrected = '#' + hex_code
            else:
                 hex_code_corrected = hex_code

            try:
                # Use the directly accessed function
                css_rgb_tuple = _hex_to_rgb(hex_code_corrected)
                css_rgb = np.array(css_rgb_tuple)
                dist = np.linalg.norm(requested_rgb - css_rgb)

                if dist < min_dist:
                    min_dist = dist
                    closest_name = name
                    closest_rgb = css_rgb_tuple

            except ValueError: continue # Skip invalid hex from older spec?
            except TypeError: # If _hex_to_rgb is the dummy lambda
                 print("ERROR: _hex_to_rgb function not properly loaded.")
                 return "Color Func Error", None

        # Prepend ~ to indicate webcolors fallback result
        return f"~ {closest_name}", closest_rgb

    except Exception as e:
         print(f"Webcolors fallback error: {e}")
         return "Color Lookup Error", None


def get_average_color(frame, roi):
    """Calculates the average BGR color within a given ROI (x, y, w, h)."""
    x, y, w, h = roi
    # Ensure ROI dimensions are integers
    x, y, w, h = int(x), int(y), int(w), int(h)

    if w <= 0 or h <= 0: return None
    # Clamp coordinates to frame boundaries
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

    # Check if clamped ROI is valid
    if x2 <= x1 or y2 <= y1: return None

    roi_region = frame[y1:y2, x1:x2]
    if roi_region.size == 0: return None

    # Calculate the average BGR color
    average_color_bgr = np.mean(roi_region, axis=(0, 1))
    return tuple(average_color_bgr.astype(int))


# Load colors automatically when the module is imported (will now try common_colors.csv first)
load_colors_from_csv()