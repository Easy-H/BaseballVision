import numpy as np

def only_bone(frame, frame_with_pose):
    
    final_pose_only_frame = np.zeros_like(frame)
    # Compare pixels: if they're different, it means something was drawn.
    identical_pixels_mask = np.all(frame == frame_with_pose, axis=2)
    
    # Copy only the pixels that are different (where pose or name was drawn)
    final_pose_only_frame[~identical_pixels_mask] = frame_with_pose[~identical_pixels_mask]
    return final_pose_only_frame