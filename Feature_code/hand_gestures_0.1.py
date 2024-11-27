import cv2
import mediapipe as mp
import csv
import matplotlib.pyplot as plt
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open video capture
video_path = r'C:\Users\kostas\Desktop\thesis\Feature_code\video\Car Accident .mp4'
cap = cv2.VideoCapture(video_path)

# Prepare CSV file for writing
csv_file_path = 'hand_landmarks.csv'
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write header with clear descriptions
    csv_writer.writerow(['Word', 'Landmark Index', 'X Coordinate', 'Y Coordinate', 'Z Coordinate'])

    # Define the word you want to associate with the landmarks
    word_to_save = os.path.splitext(os.path.basename(video_path))[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Save landmark positions with clearer formatting
                for index, landmark in enumerate(hand_landmarks.landmark):
                    x = landmark.x  # Normalized x coordinate (0 to 1)
                    y = landmark.y  # Normalized y coordinate (0 to 1)
                    z = landmark.z  # Normalized z coordinate (0 to 1)

                    # Write to CSV file (word, landmark index, x, y, z)
                    csv_writer.writerow([word_to_save, index, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])

                # Draw a circle at the index finger tip position
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = frame.shape
                x_tip, y_tip = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                cv2.circle(frame, (x_tip, y_tip), 10, (0, 255, 0), -1)  # Green circle

        # Convert BGR frame to RGB for Matplotlib display
        rgb_frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the processed frame using Matplotlib
        plt.imshow(rgb_frame_display)
        plt.axis('off')  # Hide axes
        plt.pause(0.001)  # Pause to allow for rendering

        if plt.waitforbuttonpress(timeout=0.01):  # Check for button press to exit
            break

cap.release()
plt.close()

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        h, w, _ = frame.shape
        x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
        
        # Draw a circle at the index finger tip position
        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)  # Green circle