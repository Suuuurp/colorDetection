# hsv_tuner.py
import cv2
import numpy as np

# Callback function for trackbars (does nothing, but required by createTrackbar)
def nothing(x):
    """Placeholder callback function for trackbars."""
    pass

# --- Configuration ---
WEBCAM_ID = 0 # Change this if your desired webcam is not the default (0)

# --- Initialization ---
cap = cv2.VideoCapture(WEBCAM_ID)
if not cap.isOpened():
    print(f"Error: Could not open webcam with ID {WEBCAM_ID}.")
    exit()

# Create windows
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('Result (HSV Filtered)', cv2.WINDOW_NORMAL)
cv2.namedWindow('Trackbars')

# Resize trackbar window for better layout (optional)
cv2.resizeWindow('Trackbars', 640, 250) # Adjust size as needed

# --- Create Trackbars ---
# OpenCV HSV Range: H: 0-179, S: 0-255, V: 0-255

# Lower HSV Bounds
cv2.createTrackbar('H Low', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('S Low', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('V Low', 'Trackbars', 0, 255, nothing)

# Upper HSV Bounds
cv2.createTrackbar('H High', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('S High', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('V High', 'Trackbars', 255, 255, nothing)

print("--- HSV Tuner ---")
print("Instructions:")
print("1. Place the object of the target color in front of the camera.")
print("2. Adjust the trackbar sliders.")
print("3. Goal: Make ONLY the target color appear WHITE in the 'Mask' window.")
print("   - Everything else should be BLACK.")
print("   - Use the 'Result' window to see which original colors remain.")
print("4. Tuning Strategy:")
print("   - Start with wide S and V ranges (S Low=0, V Low=0, S High=255, V High=255).")
print("   - Narrow the Hue (H Low/H High) range first to isolate the basic color.")
print("     (Remember Red wraps around: find 0-15 AND 170-179 separately).")
print("   - Increase 'S Low' to remove grayish areas.")
print("   - Increase 'V Low' to remove shadows/dark areas.")
print("   - Decrease 'V High' to remove highlights/bright areas.")
print("5. Press 'q' to quit when you have isolated the color well.")
print("-----------------")

# --- Main Loop ---
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Flip frame horizontally for a more intuitive mirror view
    frame = cv2.flip(frame, 1)

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions from trackbars
    h_low = cv2.getTrackbarPos('H Low', 'Trackbars')
    s_low = cv2.getTrackbarPos('S Low', 'Trackbars')
    v_low = cv2.getTrackbarPos('V Low', 'Trackbars')
    h_high = cv2.getTrackbarPos('H High', 'Trackbars')
    s_high = cv2.getTrackbarPos('S High', 'Trackbars')
    v_high = cv2.getTrackbarPos('V High', 'Trackbars')

    # Create the numpy arrays for lower and upper bounds
    lower_bound = np.array([h_low, s_low, v_low])
    upper_bound = np.array([h_high, s_high, v_high])

    # Create the mask based on the current bounds
    # Pixels within the range become white (255), others black (0)
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Optional: Apply morphological operations to reduce noise in the mask
    # kernel = np.ones((5,5),np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Removes small noise spots
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fills small holes

    # Apply the mask to the original frame to see the result visually
    result_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # --- Display the frames ---
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result (HSV Filtered)', result_frame)

    # --- Exit condition ---
    # Wait for 1 millisecond for a key press. If 'q' is pressed, break loop.
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()

# --- Print Final Values ---
print("\n--- Tuning Complete ---")
print("Final HSV bounds from trackbars:")
print(f"Lower: np.array([{h_low}, {s_low}, {v_low}])")
print(f"Upper: np.array([{h_high}, {s_high}, {v_high}])")
print("\nCopy these arrays into the HSV_RANGES dictionary in app.py.")
print("Remember Red might need two ranges (e.g., H 0-10 and H 170-180).")
print("Tune Black/White/Gray mainly using S and V sliders.")
print("-----------------------")