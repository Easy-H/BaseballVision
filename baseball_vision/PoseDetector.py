from baseball_vision.LandmarkKalmanFilter import LandmarkKalmanFilter
import config
import mediapipe as mp
import numpy as np

def NoneList(size):
        retval = []
        for i in range(size):
            retval.append(None)
        return retval
    
class PoseDetector:
    def __init__(self):
        pass
    def process(self, images: list, fusion_method: str = 'weighted_average'):
        pass
    def get_all_frames_landmarks_3d(self):
        pass
    def close(self):
        pass
    @staticmethod
    def _calculate_median_3d_landmarks(raw_landmarks_3d_per_view: list) -> np.ndarray:
        """
        Calculates the median 3D landmarks from a list of 3D landmark arrays (from different views).

        Args:
            raw_landmarks_3d_per_view: A list of NumPy arrays, where each array is (num_landmarks, 3)
                                       representing 3D landmarks from a single view.

        Returns:
            A NumPy array (num_landmarks, 3) of the median 3D landmarks.
        """
        if not raw_landmarks_3d_per_view:
            return None # Or raise an error, depending on desired error handling

        landmarks_3d_stack = np.array(raw_landmarks_3d_per_view)
        median_landmarks_3d = np.median(landmarks_3d_stack, axis=0)
        return median_landmarks_3d

    @staticmethod
    def _calculate_weighted_average_3d_landmarks(
        raw_landmarks_3d_per_view: list, all_visibility_scores: list
    ) -> np.ndarray:
        """
        Calculates the weighted average 3D landmarks based on visibility scores.

        Args:
            raw_landmarks_3d_per_view: A list of NumPy arrays, where each array is (num_landmarks, 3)
                                       representing 3D landmarks from a single view.
            all_visibility_scores: A list of NumPy arrays, where each array is (num_landmarks,)
                                   representing visibility scores for landmarks in each view.

        Returns:
            A NumPy array (num_landmarks, 3) of the weighted average 3D landmarks.
        """
        if not raw_landmarks_3d_per_view or not all_visibility_scores:
            return None

        # Assuming all landmark arrays have the same shape
        num_landmarks = raw_landmarks_3d_per_view[0].shape[0]

        weighted_avg_landmarks_3d = np.zeros((num_landmarks, 3))
        total_weights_per_landmark = np.zeros(num_landmarks)

        for i, landmarks_3d_view in enumerate(raw_landmarks_3d_per_view):
            visibility_scores_view = all_visibility_scores[i]
            
            # Expand visibility scores to match the (num_landmarks, 3) shape for element-wise multiplication
            weights_expanded = np.expand_dims(visibility_scores_view, axis=-1)

            weighted_avg_landmarks_3d += landmarks_3d_view * weights_expanded
            total_weights_per_landmark += visibility_scores_view
        
        # Avoid division by zero for landmarks with zero total weight
        total_weights_per_landmark[total_weights_per_landmark == 0] = 1e-6 # Small non-zero value

        weighted_avg_landmarks_3d /= np.expand_dims(total_weights_per_landmark, axis=-1)
        return weighted_avg_landmarks_3d
        
class MediaPipePoseDetector(PoseDetector):
    def __init__(self):
        self.pose_detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=config.MODEL_COMPLEXITY,
            enable_segmentation=False,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        self.kalman_filter_processor = LandmarkKalmanFilter(
            num_landmarks=len(mp.solutions.pose.PoseLandmark),
            process_noise_std=config.PROCESS_NOISE_STD,
            measurement_noise_std=config.MEASUREMENT_NOISE_STD
        )
    
    def process(self, images: list, fusion_method: str = 'weighted_average'):
        """
        Processes a list of images (multiple views for a single moment in time)
        to calculate fused 3D landmarks (median or weighted average),
        optionally followed by Kalman filtering.

        Args:
            images: A list of image frames, where each image is a NumPy array.
            fusion_method: The method to use for fusing 3D landmarks ('median' or 'weighted_average').

        Returns:
            A tuple containing:
            - processed_landmarks_3d: NumPy array of the fused 3D landmarks for the current frame,
                                      optionally Kalman filtered.
            - all_raw_landmarks_2d: A list of NumPy arrays, where each array contains
                                    2D landmarks for a view (image).
            - all_visibility_scores: A list of NumPy arrays, where each array contains
                                     visibility scores for a view (image).
        """
        raw_landmarks_3d_per_view = []
        all_raw_landmarks_2d = []
        all_visibility_scores = []

        for image in images: # Iterate through each view for the current time step
            results = self.pose_detector.process(image)
            
            if results.pose_world_landmarks is None:
                print("Warning: Pose not detected in one of the views. Skipping this view's data.")
                continue 
            
            current_view_landmarks_3d = np.array([
                [lmk.x, lmk.y, lmk.z]
                for lmk in results.pose_world_landmarks.landmark
            ])
            raw_landmarks_3d_per_view.append(current_view_landmarks_3d)

            current_view_landmarks_2d = np.array([
                [lmk.x, lmk.y, lmk.z]
                for lmk in results.pose_landmarks.landmark
            ])
            all_raw_landmarks_2d.append(current_view_landmarks_2d)

            current_view_visibility_scores = np.array([
                lmk.visibility
                for lmk in results.pose_world_landmarks.landmark
            ])
            all_visibility_scores.append(current_view_visibility_scores)

        if not raw_landmarks_3d_per_view:
            return None, NoneList(len(images)), NoneList(len(images))

        # --- Fusion Method Selection ---
        fused_landmarks_3d = None
        if fusion_method == 'median':
            fused_landmarks_3d = self._calculate_median_3d_landmarks(raw_landmarks_3d_per_view)
        elif fusion_method == 'weighted_average':
            fused_landmarks_3d = self._calculate_weighted_average_3d_landmarks(
                raw_landmarks_3d_per_view, all_visibility_scores
            )
        else:
            raise ValueError("Invalid fusion_method. Choose 'median' or 'weighted_average'.")

        if fused_landmarks_3d is None: # Handle cases where fusion functions might return None
            return None, NoneList(len(images)), NoneList(len(images))

        # Apply Kalman filter to the fused 3D landmarks
        if config.APPLY_KALMAN_FILTER:
            # Use mean visibility for Kalman filter regardless of fusion method
            combined_visibility_for_kalman = np.mean(all_visibility_scores, axis=0) if all_visibility_scores else np.zeros(len(mp.solutions.pose.PoseLandmark))

            if not self.kalman_filter_processor.initialized:
                self.kalman_filter_processor.initialize_state(fused_landmarks_3d)
                processed_landmarks_3d = fused_landmarks_3d # First frame uses raw fused data
            else:
                processed_landmarks_3d = self.kalman_filter_processor.filter(
                    fused_landmarks_3d,
                    visibility_scores=combined_visibility_for_kalman,
                    min_visibility_threshold=0.6 
                )
        else:
            processed_landmarks_3d = fused_landmarks_3d
        
        return processed_landmarks_3d, all_raw_landmarks_2d, all_visibility_scores
    def close(self):
        self.pose_detector.close()