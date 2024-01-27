import mediapipe as mp
import cv2
import random
# Initialize the face mesh solution
mp_face_mesh = mp.solutions.face_mesh

# Create a face mesh instance
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

c = []
for i in range(1000):
    c.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture the current frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess the frame for face detection
    results = face_mesh.process(rgb_frame)

    # Check if any faces were detected
    if results.multi_face_landmarks:
        # Get the landmarks for the first detected face
        face_landmarks = results.multi_face_landmarks[0]

        # Draw the landmarks on the frame
        for i, landmark in enumerate(face_landmarks.landmark):
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (c[i][0], c[i][1], c[i][2]), -1)

    # Display the frame with detected facial landmarks
    output = cv2.resize(frame, (int(frame.shape[1] * 1.5), int(frame.shape[0] * 1.5)))
    cv2.imshow('Frame with facial landmarks', output)

    # Check if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
