import cv2
import mediapipe as mp
import numpy as np
import config # Assuming config.py exists with MIN_DRAW_VISIBILITY and CONNECTIONS_COLORS

def draw_diff(frame1, frame2):
    """
    Compares two frames and returns a frame showing only the differences.
    Useful for visualizing what was newly drawn onto a frame.

    Args:
        frame1 (np.array): The base OpenCV BGR image frame.
        frame2 (np.array): The modified OpenCV BGR image frame.

    Returns:
        np.array: A new frame with only the differing pixels from frame2.
    """
    result = np.zeros_like(frame1)
    # Compare pixels: if they're different, it means something was drawn.
    identical_pixels_mask = np.all(frame1 == frame2, axis=2)
    
    # Copy only the pixels that are different (where pose or name was drawn)
    result[~identical_pixels_mask] = frame2[~identical_pixels_mask]
    return result
    
def draw_landmarks_custom(image, landmarks_array, image_width, image_height, visibility_array=None):
    """
    Draws custom-styled pose landmarks on the image from a NumPy array.
    - Simplifies face to a single nose node.
    - Draws other body points as small gray dots.

    Args:
        image (np.array): The OpenCV BGR image frame to draw on.
        landmarks_array (np.array): A NumPy array of pose landmarks, shape (num_landmarks, 3)
                                    where each row is [x, y, z] (normalized coordinates).
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        visibility_array (np.array, optional): An array of visibility scores for each landmark,
                                                shape (num_landmarks,). Defaults to None,
                                                in which case all landmarks are considered visible.
    """
    num_landmarks = landmarks_array.shape[0]
    
    # Create a default visibility array if not provided
    if visibility_array is None:
        visibility_array = np.ones(num_landmarks)

    for idx in range(num_landmarks):
        landmark_x = landmarks_array[idx, 0]
        landmark_y = landmarks_array[idx, 1]
        landmark_visibility = visibility_array[idx]
        
        # Only draw if landmark visibility is good
        if landmark_visibility < config.MIN_DRAW_VISIBILITY:
            continue
        
        center_coordinates = (int(landmark_x * image_width), int(landmark_y * image_height))

        # Accessing MediaPipe's PoseLandmark enum values for comparison
        if idx == mp.solutions.pose.PoseLandmark.NOSE.value: # Nose: single face node
            cv2.circle(image, center_coordinates, 5, (255, 255, 255), -1) # White circle
        # Facial landmarks (eyes, ears, mouth) are typically from 1 to 10
        elif mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER.value <= idx <= mp.solutions.pose.PoseLandmark.MOUTH_RIGHT.value:
            pass # Don't draw these facial landmarks
        else: # Body, arm, leg landmarks: small gray dot
            cv2.circle(image, center_coordinates, 2, (100, 100, 100), -1)

def draw_connections_custom(image, landmarks_array, image_width, image_height, visibility_array=None):
    """
    Draws custom color-coded pose connections (bones) on the image from a NumPy array.

    Args:
        image (np.array): The OpenCV BGR image frame to draw on.
        landmarks_array (np.array): A NumPy array of pose landmarks, shape (num_landmarks, 3)
                                    where each row is [x, y, z] (normalized coordinates).
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        visibility_array (np.array, optional): An array of visibility scores for each landmark,
                                                shape (num_landmarks,). Defaults to None,
                                                in which case all landmarks are considered visible.
    """
    num_landmarks = landmarks_array.shape[0]

    # Create a default visibility array if not provided
    if visibility_array is None:
        visibility_array = np.ones(num_landmarks)

    for connection in mp.solutions.pose.POSE_CONNECTIONS:
        idx1, idx2 = connection
        
        # Ensure indices are within bounds for the landmarks_array
        if not (0 <= idx1 < num_landmarks and 0 <= idx2 < num_landmarks):
            continue

        if visibility_array[idx1] < config.MIN_DRAW_VISIBILITY \
           or visibility_array[idx2] < config.MIN_DRAW_VISIBILITY:
            continue
        
        # Get color for the connection from the predefined map
        color = config.CONNECTIONS_COLORS.get(connection, None)
        if color is None: # Check if tuple order is reversed in map (for robustness)
            color = config.CONNECTIONS_COLORS.get((idx2, idx1), None)

        if color is not None:
            point1 = (int(landmarks_array[idx1, 0] * image_width),
                      int(landmarks_array[idx1, 1] * image_height))
            point2 = (int(landmarks_array[idx2, 0] * image_width),
                      int(landmarks_array[idx2, 1] * image_height))
            cv2.line(image, point1, point2, color, 2) # Line thickness 2

def draw_pose_on_frame(frame, pose_landmarks_array, pose_visibility_array=None):
    """
    Orchestrates drawing pose landmarks and connections on a frame with custom styling.

    Args:
        frame (np.array): The OpenCV BGR image frame.
        pose_landmarks_array (np.array): A NumPy array of pose landmarks, shape (num_landmarks, 3)
                                        where each row is [x, y, z] (normalized coordinates).
        pose_visibility_array (np.array, optional): An array of visibility scores for each landmark,
                                                     shape (num_landmarks,). Defaults to None.

    Returns:
        np.array: The frame with the custom-drawn pose.
    """
    if pose_landmarks_array is None:
        return frame
    frame_with_pose = frame.copy()
    h, w, _ = frame.shape
    
    # Call functions to draw landmarks and connections
    draw_landmarks_custom(frame_with_pose, pose_landmarks_array, w, h, pose_visibility_array)
    draw_connections_custom(frame_with_pose, pose_landmarks_array, w, h, pose_visibility_array)
    
    return frame_with_pose