import numpy as np

class IFusionMethod:
    def fusion(self, raw_landmarks_3d_per_view: list,
               all_visibility_scores: list) -> np.ndarray:
        pass

class MedianFusion(IFusionMethod):
    def fusion(self, raw_landmarks_3d_per_view: list,
               all_visibility_scores: list) -> np.ndarray:
        
        if not raw_landmarks_3d_per_view:
            return None # Or raise an error, depending on desired error handling

        landmarks_3d_stack = np.array(raw_landmarks_3d_per_view)
        median_landmarks_3d = np.median(landmarks_3d_stack, axis=0)
        return median_landmarks_3d
    
class WeightedAverageFusion(IFusionMethod):
    def fusion(self, raw_landmarks_3d_per_view: list,
               all_visibility_scores: list) -> np.ndarray:
        
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