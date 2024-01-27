import mediapipe as mp
import cv2
import random
import numpy as np

def find_essential_matrix(left_camera_points, right_camera_points):
    """
    Find the essential matrix given corresponding points in two stereo cameras.

    Parameters:
    - left_camera_points: List of corresponding points in the left camera.
    - right_camera_points: List of corresponding points in the right camera.

    Returns:
    - Essential matrix (3x3 matrix).
    """

    # Convert lists to numpy arrays
    points_left = np.array(left_camera_points)
    points_right = np.array(right_camera_points)

    # Normalize points
    points_left_normalized = cv2.normalize(points_left, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    points_right_normalized = cv2.normalize(points_right, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Find essential matrix using RANSAC
    E, _ = cv2.findEssentialMat(points_left_normalized, points_right_normalized, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    return E




# Initialize the face mesh solution
mp_face_mesh = mp.solutions.face_mesh

# Create a face mesh instance
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

c = []
for i in range(1000):
    c.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

r_camera = cv2.imread('data/khodam/C.jpg')
l_camera = cv2.imread('data/khodam/R.jpg')
# Convert the frame to RGB
rgb_r_camera = cv2.cvtColor(r_camera, cv2.COLOR_BGR2RGB)
rgb_l_camera = cv2.cvtColor(l_camera, cv2.COLOR_BGR2RGB)

# Preprocess the frame for face detection
r_results = face_mesh.process(rgb_r_camera)

# Check if any faces were detected
if r_results.multi_face_landmarks:
    # Get the landmarks for the first detected face
    face_landmarks = r_results.multi_face_landmarks[0]

    # Draw the landmarks on the frame
    for i, landmark in enumerate(face_landmarks.landmark):
        x = int(landmark.x * r_camera.shape[1])
        y = int(landmark.y * r_camera.shape[0])
        cv2.circle(r_camera, (x, y), 5, (c[i][0], c[i][1], c[i][2]), -1)

# Display the frame with detected facial landmarks
output = cv2.resize(r_camera, (int(r_camera.shape[1] * 0.5), int(l_camera.shape[0] * 0.5)))
cv2.imshow('R', output)

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
l_results = face_mesh.process(rgb_l_camera)

# Check if any faces were detected
if l_results.multi_face_landmarks:
    # Get the landmarks for the first detected face
    face_landmarks = l_results.multi_face_landmarks[0]

    # Draw the landmarks on the frame
    for i, landmark in enumerate(face_landmarks.landmark):
        x = int(landmark.x * l_camera.shape[1])
        y = int(landmark.y * l_camera.shape[0])
        cv2.circle(l_camera, (x, y), 5, (c[i][0], c[i][1], c[i][2]), -1)


 
# Display the frame with detected facial landmarks
output = cv2.resize(l_camera, (int(l_camera.shape[1] * 0.5), int(l_camera.shape[0] * 0.5)))
cv2.imshow('L', output)


rc_points = []
for landmark in r_results.multi_face_landmarks[0].landmark:
    rc_points.append([landmark.x * r_camera.shape[1],
                      landmark.y * r_camera.shape[0]])

lc_points = []
for landmark in l_results.multi_face_landmarks[0].landmark:
    lc_points.append([landmark.x * l_camera.shape[1],
                      landmark.y * l_camera.shape[0]])


E = find_essential_matrix(lc_points, rc_points)
