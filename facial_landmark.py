import mediapipe as mp
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

r_camera = cv2.imread('IMG_6290.png')
l_camera = cv2.imread('IMG_6291.png')
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
cv2.waitKey(1000)


rc_points = []
for landmark in r_results.multi_face_landmarks[0].landmark:
    rc_points.append([landmark.x * r_camera.shape[1],
                      landmark.y * r_camera.shape[0]])

lc_points = []
for landmark in l_results.multi_face_landmarks[0].landmark:
    lc_points.append([landmark.x * l_camera.shape[1],
                      landmark.y * l_camera.shape[0]])
    


indexes = []
apd_R = []
apd_L = []
N = 100
for i in range(0, 468, 4):
    indexes.append(random.randint(0, 467))
    apd_R.append(rc_points[i])
    apd_L.append(lc_points[i])

#E = find_essential_matrix(apd_R, apd_L)


E = find_essential_matrix(lc_points, rc_points)


def decompose_essential_matrix(E):
    """
    Decompose the essential matrix into possible rotation and translation matrices.

    Parameters:
    - E: Essential matrix (3x3 matrix).

    Returns:
    - Rotation matrices (R1, R2) and translation vector (t).
    """
    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

    # Two possible rotations
    R1 = np.dot(U, np.dot(W, Vt))
    R2 = np.dot(U, np.dot(W.T, Vt))

    # Ensure proper rotation matrix
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    # Translation
    t = U[:, 2]

    return R1, R2, t


def triangulate_points(P1, P2, pts1, pts2):
    """
    Triangulate corresponding points from two images with projection matrices.

    Parameters:
    - P1: Projection matrix for the first camera.
    - P2: Projection matrix for the second camera.
    - pts1: Corresponding points in the first image.
    - pts2: Corresponding points in the second image.

    Returns:
    - 3D points in homogeneous coordinates.
    """

    print(pts1.shape, pts2.shape)
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = pts4D / pts4D[3]
    return pts3D[:3]

K = np.array([[3.05447534e+04, 0.00000000e+00, 3.35472082e+02],
 [0.00000000e+00, 1.74067924e+04, 4.58258041e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Find rotation and translation from the essential matrix
R1, R2, t = decompose_essential_matrix(E)

# Choose the correct R and t here (there are 4 possibilities)
# This often requires additional constraints or information

# Create projection matrices for the two cameras
P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2 = np.dot(K, np.hstack((R1, t.reshape(3, 1))))  # Replace R1 and t with chosen values

# Convert points to homogeneous coordinates
pts1_homo = cv2.convertPointsToHomogeneous(np.array(rc_points))[:, 0, :2] # Extract x, y and transpose
pts2_homo = cv2.convertPointsToHomogeneous(np.array(lc_points))[:, 0, :2]


# Triangulate points
points_3D = triangulate_points(P1, P2, pts1_homo, pts2_homo)

X = points_3D[0]
Y = points_3D[1]
Z = points_3D[2]

# Creating a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for 3D points
ax.scatter(X, Y, Z)

# Setting labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Setting title
ax.set_title('3D Facial Mask Points Visualization')

# Show plot


print(R1, R2)


from scipy.spatial import Delaunay

# ... [your existing code for extracting 3D points] ...

# Perform Delaunay triangulation on the 2D projection of 3D points
# Projecting onto the XY plane (ignoring Z)
points_2D = np.vstack([X, Y]).T
tri = Delaunay(points_2D)

# Creating a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plotting the triangulated surface
ax.plot_trisurf(X, Y, Z, triangles=tri.simplices, cmap='viridis')

# Setting labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Setting title
ax.set_title('3D Triangulated Facial Surface')

# Show plot
plt.show()