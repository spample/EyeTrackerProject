import cv2
import urllib.request
import numpy as np
import dlib
from math import hypot

# Initialize variables
keyboard = np.zeros((600, 1000, 3), np.uint8)
url = 'http://192.168.86.212/cam-lo.jpg'
cv2.namedWindow("live Cam Testing", cv2.WINDOW_AUTOSIZE)

# Create VideoCapture object (though not used directly here)
cap = cv2.VideoCapture(url)

# Initialize facial feature detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def draw_menu():
    rows, cols, _ = keyboard.shape
    th_lines = 4  # thickness lines
    cv2.line(keyboard, (int(cols / 2) - int(th_lines), 0), (int(cols / 2) - int(th_lines / 2), rows),
             (51, 51, 51), th_lines)
    cv2.putText(keyboard, "LEFT", (80, 300), cv2.FONT_HERSHEY_PLAIN, 6, (255, 255, 255), 5)
    cv2.putText(keyboard, "RIGHT", (80 + int(cols / 2), 300), cv2.FONT_HERSHEY_PLAIN, 6, (255, 255, 255), 5)
    cv2.putText(keyboard, "UP", (430, 150), cv2.FONT_HERSHEY_PLAIN, 6, (255, 255, 255), 5)
    cv2.putText(keyboard, "Down", (380, 500), cv2.FONT_HERSHEY_PLAIN, 6, (255, 255, 255), 5)


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                               np.int32)

    height, width, _ = img.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio


# Function to get eye contours
def eyes_contour_points(facial_landmarks):
    left_eye = []
    right_eye = []
    for n in range(36, 42):  # Left eye landmark points
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        left_eye.append([x, y])
    for n in range(42, 48):  # Right eye landmark points
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        right_eye.append([x, y])
    left_eye = np.array(left_eye, np.int32)
    right_eye = np.array(right_eye, np.int32)
    return left_eye, right_eye


# Function to calculate blinking ratio (EAR)
def get_blinking_ratio(eye_points, facial_landmarks):
    # Get the coordinates of the eye landmarks
    eye = []
    for n in eye_points:
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        eye.append([x, y])

    # Convert to numpy array
    eye = np.array(eye, np.int32)

    # Compute the vertical and horizontal distances
    A = hypot(eye[1][0] - eye[5][0], eye[1][1] - eye[5][1])  # Vertical distance (top-bottom)
    B = hypot(eye[2][0] - eye[4][0], eye[2][1] - eye[4][1])  # Vertical distance (left-right)
    C = hypot(eye[0][0] - eye[3][0], eye[0][1] - eye[3][1])  # Horizontal distance (left-right)

    # Calculate the eye aspect ratio (EAR)
    ear = (A + B) / (2.0 * C)

    return ear


# Main loop
while True:
    try:
        # Fetch the image from the URL
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)


        # Decode the image into an OpenCV image
        img = cv2.imdecode(imgnp, cv2.IMREAD_COLOR)

        # Check if the image was decoded successfully
        if img is None:
            print("Failed to decode image.")
            continue

        # Ensure the image is valid, now you can access 'shape'
        rows, cols, _ = img.shape

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Optionally draw a white space for the loading bar at the bottom of the frame
        img[rows - 50: rows, 0: cols] = (255, 255, 255)

        # Detect faces in the grayscale image
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            left_eye, right_eye = eyes_contour_points(landmarks)

            # Detect blinking ratio for both eyes
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

            # Draw contours around the eyes
            cv2.polylines(img, [left_eye], True, (0, 0, 255), 2)
            cv2.polylines(img, [right_eye], True, (0, 0, 255), 2)

            # Detect the gaze ratio and select keyboard menu (if needed)
            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

            print("Gaze ratio:", gaze_ratio)

            # Perform actions based on gaze ratio (not shown in full code for brevity)
            if gaze_ratio <= 1.3:  # Example logic for gaze-based selection
                print("Looking right.")
            elif gaze_ratio > 1.3:
                print("Looking left.")
            else:
                print("Looking center.")

        # Display the frame with annotations
        cv2.imshow('live Cam Testing', img)


        # Wait for 'q' key press to quit
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    except Exception as e:
        print(f"Error fetching or decoding frame: {e}")
        continue

cv2.destroyAllWindows()
