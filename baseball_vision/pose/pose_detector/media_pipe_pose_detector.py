from baseball_vision.LandmarkKalmanFilter import LandmarkKalmanFilter
from .pose_detector import IPoseDetector
from .fusion_method import MedianFusion
import config

import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def NoneList(size):
    retval = []
    for i in range(size):
        retval.append(None)
    return retval

# MediaPipe Landmark Names for C3D Marker Mapping
MEDIAPIPE_LANDMARK_NAMES = {
    mp_pose.PoseLandmark.NOSE.value: 'NOSE',
    mp_pose.PoseLandmark.LEFT_EYE_INNER.value: 'L_EYE_INNER',
    mp_pose.PoseLandmark.LEFT_EYE.value: 'L_EYE',
    mp_pose.PoseLandmark.LEFT_EYE_OUTER.value: 'L_EYE_OUTER',
    mp_pose.PoseLandmark.RIGHT_EYE_INNER.value: 'R_EYE_INNER',
    mp_pose.PoseLandmark.RIGHT_EYE.value: 'R_EYE',
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value: 'R_EYE_OUTER',
    mp_pose.PoseLandmark.LEFT_EAR.value: 'L_EAR',
    mp_pose.PoseLandmark.RIGHT_EAR.value: 'R_EAR',
    mp_pose.PoseLandmark.MOUTH_LEFT.value: 'L_MOUTH',
    mp_pose.PoseLandmark.MOUTH_RIGHT.value: 'R_MOUTH',
    mp_pose.PoseLandmark.LEFT_SHOULDER.value: 'L_SHOULDER',
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value: 'R_SHOULDER',
    mp_pose.PoseLandmark.LEFT_ELBOW.value: 'L_ELBOW',
    mp_pose.PoseLandmark.RIGHT_ELBOW.value: 'R_ELBOW',
    mp_pose.PoseLandmark.LEFT_WRIST.value: 'L_WRIST',
    mp_pose.PoseLandmark.RIGHT_WRIST.value: 'R_WRIST',
    mp_pose.PoseLandmark.LEFT_PINKY.value: 'L_PINKY',
    mp_pose.PoseLandmark.RIGHT_PINKY.value: 'R_PINKY',
    mp_pose.PoseLandmark.LEFT_INDEX.value: 'L_INDEX',
    mp_pose.PoseLandmark.RIGHT_INDEX.value: 'R_INDEX',
    mp_pose.PoseLandmark.LEFT_THUMB.value: 'L_THUMB',
    mp_pose.PoseLandmark.RIGHT_THUMB.value: 'R_THUMB',
    mp_pose.PoseLandmark.LEFT_HIP.value: 'L_HIP',
    mp_pose.PoseLandmark.RIGHT_HIP.value: 'R_HIP',
    mp_pose.PoseLandmark.LEFT_KNEE.value: 'L_KNEE',
    mp_pose.PoseLandmark.RIGHT_KNEE.value: 'R_KNEE',
    mp_pose.PoseLandmark.LEFT_ANKLE.value: 'L_ANKLE',
    mp_pose.PoseLandmark.RIGHT_ANKLE.value: 'R_ANKLE',
    mp_pose.PoseLandmark.LEFT_HEEL.value: 'L_HEEL',
    mp_pose.PoseLandmark.RIGHT_HEEL.value: 'R_HEEL',
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value: 'L_FOOT_INDEX',
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value: 'R_FOOT_INDEX',
}

class MediaPipePoseDetector(IPoseDetector):
    def __init__(self, fusion_method=MedianFusion):
        self.pose_detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=config.MODEL_CONFIG["MODEL_COMPLEXITY"],
            enable_segmentation=False,
            min_detection_confidence=config.MODEL_CONFIG["MIN_DETECTION_CONFIDENCE"],
            min_tracking_confidence=config.MODEL_CONFIG["MIN_TRACKING_CONFIDENCE"]
        )
        
        self.kalman_filter_processor = LandmarkKalmanFilter(
            num_landmarks=len(mp.solutions.pose.PoseLandmark),
            process_noise_std=config.MODEL_CONFIG["PROCESS_NOISE_STD"],
            measurement_noise_std=config.MODEL_CONFIG["MEASUREMENT_NOISE_STD"]
        )

        self.fusion_method = fusion_method()
    
    def process(self, images: list):

        raw_landmarks_3d_per_view, all_raw_landmarks_2d, \
            all_visibility_scores = self._process_image(images)

        if not raw_landmarks_3d_per_view:
            return self._process_failed(len(images))

        # --- Fusion Method Selection ---
        fused_landmarks_3d = self.fusion_method.fusion(
            raw_landmarks_3d_per_view, all_visibility_scores)
        
        if fused_landmarks_3d is None:
            return self._process_failed(len(images))

        processed_landmarks_3d = self._apply_kalman_filter(
            fused_landmarks_3d, all_visibility_scores)
        
        return self._numpy_to_dict(processed_landmarks_3d), \
            self._numpy_list_to_dict_list(all_raw_landmarks_2d), \
            self._numpy_list_to_dict_list(all_visibility_scores)
    
    def _numpy_to_dict(self, arr:np.array):
        ret = {}
        for idx in range(arr.shape[0]):
            ret[MEDIAPIPE_LANDMARK_NAMES[idx]] = arr[idx]
        return ret

    def _numpy_list_to_dict_list(self, arr:list[np.array]):
        ret = []
        for a in arr:
            ret.append(self._numpy_to_dict(a))
        return ret
    
    def _process_failed(self, length):
        return None, NoneList(length), NoneList(length)
    
    def _process_image(self, images: list):

        raw_landmarks_3d_per_view = []
        all_raw_landmarks_2d = []
        all_visibility_scores = []

        for image in images: # Iterate through each view for the current time step
            results = self.pose_detector.process(image)
            
            if results.pose_world_landmarks is None:
                print("Warning: Pose not detected in one of the views.\
                       Skipping this view's data.")
                continue

            raw_landmarks_3d_per_view.append(np.array([
                [lmk.x, lmk.y, lmk.z]
                for lmk in results.pose_world_landmarks.landmark
            ]))

            all_raw_landmarks_2d.append(np.array([
                [lmk.x, lmk.y, lmk.z]
                for lmk in results.pose_landmarks.landmark
            ]))

            all_visibility_scores.append(np.array([
                lmk.visibility
                for lmk in results.pose_world_landmarks.landmark
            ]))
        
        return raw_landmarks_3d_per_view,\
              all_raw_landmarks_2d, all_visibility_scores

    def _apply_kalman_filter(self, fused_landmarks_3d, all_visibility_scores):
        # Apply Kalman filter to the fused 3D landmarks
        if not config.MODEL_CONFIG["APPLY_KALMAN_FILTER"]:
            return fused_landmarks_3d
        
        # Use mean visibility for Kalman filter regardless of fusion method
        combined_visibility_for_kalman = \
            np.mean(all_visibility_scores, axis=0) \
                if all_visibility_scores \
                    else np.zeros(len(mp.solutions.pose.PoseLandmark))

        return self.kalman_filter_processor.filter(
            fused_landmarks_3d,
            visibility_scores=combined_visibility_for_kalman,
            min_visibility_threshold=0.6 
        )
    
    def close(self):
        self.pose_detector.close()