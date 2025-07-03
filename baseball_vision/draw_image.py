import cv2
import mediapipe as mp
import numpy as np
import config

def draw_diff(frame1, frame2):
    result = np.zeros_like(frame1)
    # Compare pixels: if they're different, it means something was drawn.
    identical_pixels_mask = np.all(frame1 == frame2, axis=2)
    
    # Copy only the pixels that are different (where pose or name was drawn)
    result[~identical_pixels_mask] = frame2[~identical_pixels_mask]
    return result
    
def draw_landmarks_custom(image, landmarks, image_width, image_height):
    """
    Draws custom-styled pose landmarks on the image.
    - Simplifies face to a single nose node.
    - Draws other body points as small gray dots.

    Args:
        image (np.array): The OpenCV BGR image frame to draw on.
        landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
                    The pose landmarks detected by MediaPipe.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    """
    for idx, landmark in enumerate(landmarks.landmark):
        # Only draw if landmark visibility is good
        if landmark.visibility < config.MIN_DRAW_VISIBILITY:
            continue
        
        center_coordinates = (int(landmark.x * image_width), int(landmark.y * image_height))

        if idx == mp.solutions.pose.PoseLandmark.NOSE.value: # Nose: single face node
            cv2.circle(image, center_coordinates, 5, (255, 255, 255), -1) # White circle
        elif 1 <= idx <= 10: # Other facial landmarks (eyes, ears, mouth): don't draw
            pass
        else: # Body, arm, leg landmarks: small gray dot
            cv2.circle(image, center_coordinates, 2, (100, 100, 100), -1)

def draw_connections_custom(image, landmarks, image_width, image_height):
    """
    Draws custom color-coded pose connections (bones) on the image.

    Args:
        image (np.array): The OpenCV BGR image frame to draw on.
        landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
                    The pose landmarks detected by MediaPipe.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    """
    for connection in mp.solutions.pose.POSE_CONNECTIONS:
        idx1, idx2 = connection
        
        if landmarks.landmark[idx1].visibility < config.MIN_DRAW_VISIBILITY \
            or landmarks.landmark[idx2].visibility < config.MIN_DRAW_VISIBILITY:
            continue
        
        # Get color for the connection from the predefined map
        color = config.CONNECTIONS_COLORS.get(connection, None)
        if color is None: # Check if tuple order is reversed in map
            color = config.CONNECTIONS_COLORS.get((idx2, idx1), None)

        if color is not None:
            point1 = (int(landmarks.landmark[idx1].x * image_width),
                      int(landmarks.landmark[idx1].y * image_height))
            point2 = (int(landmarks.landmark[idx2].x * image_width),
                      int(landmarks.landmark[idx2].y * image_height))
            cv2.line(image, point1, point2, color, 2) # Line thickness 2

def draw_pose_on_frame(frame, pose_landmarks):
    """
    Orchestrates drawing pose landmarks and connections on a frame with custom styling.

    Args:
        frame (np.array): The OpenCV BGR image frame.
        pose_landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
                        The pose landmarks detected by MediaPipe.

    Returns:
        np.array: The frame with the custom-drawn pose.
    """
    frame_with_pose = frame.copy()
    h, w, _ = frame.shape
    
    # Call functions to draw landmarks and connections
    draw_landmarks_custom(frame_with_pose, pose_landmarks, w, h)
    draw_connections_custom(frame_with_pose, pose_landmarks, w, h)
    
    return frame_with_pose