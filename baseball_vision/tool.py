import mediapipe as mp
import baseball_vision.angle_calc as ac
import pandas as pd
import matplotlib.pyplot as plt

def get_joints(landmarks):
     return { "R_shoulder": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER],
                "R_elbow": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW],
                "R_wrist": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST],
                "R_hip": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP],
                "R_knee": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE],
                "R_ankle": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE],
                "L_shoulder": landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER],
                "L_elbow": landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW],
                "L_wrist": landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST],
                "L_hip": landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP],
                "L_knee": landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE],
                "L_ankle": landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE],
                "R_pinky_tip": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX],
                "L_pinky_tip": landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX]}

class AnalysisTool:
    def __init__(self):
        self.results = []
    def calc(self, landmarks):
        result = self.calc_joints(get_joints(landmarks))
        self.results.append(result)
        return result
    def calc_joints(self, joints):
        pass
    def skip(self):
        self.results.append([])
    def run(self):
        self.df_results = pd.DataFrame(self.results)
    def save(self, output_name):
        self.df_results.to_csv(output_name + ".csv", index=True)
    def show_dataframe(self):
        print(self.df_results)
    def show_graph(self, label=[]):
        remove_idx = []
        for i in range(len(label)):
            if not label[i] in self.df_results.columns.tolist():
                print(label[i] + "is not correct name")
                remove_idx.append(i)
        remove_idx.reverse()
        for i in range(len(remove_idx)):
            del label[remove_idx[i]]
            
        if len(label)==0:
            label = self.df_results.columns.tolist()
        plt.figure(figsize=(12, 8))

        for i in range(len(label)):
            plt.plot(self.df_results.index, self.df_results[label[i]], label=label[i])
        
        plt.title('Change in each joint angle (over time)', fontsize=16)
        plt.xlabel('frame idx', fontsize=12)
        plt.ylabel('degree', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class PitcherAnalysisTool(AnalysisTool):
    def calc_joints(self, joints):
        # --- 각도 계산 ---
        # 1. 오른쪽 팔꿈치 각도
        angle_R_elbow = ac.calculate_angle_3(
            joints["R_shoulder"], joints["R_elbow"], joints["R_wrist"])
        angle_L_elbow = ac.calculate_angle_3(
            joints["L_shoulder"], joints["L_elbow"], joints["L_wrist"])
    
        # 2. 오른쪽 어깨 각도 (몸통-어깨-팔꿈치"])
        angle_R_shoulder = ac.calculate_angle_3(
            joints["L_shoulder"], joints["R_shoulder"], joints["R_elbow"])
        angle_L_shoulder = ac.calculate_angle_3(
            joints["R_shoulder"], joints["L_shoulder"], joints["L_elbow"])
    
        # 3. 오른쪽 골반 각도 (몸통-골반-무릎"])
        angle_body_twist = ac.calculate_angle_4(
            joints["R_shoulder"], joints["L_shoulder"], joints["R_hip"], joints["L_hip"])
    
        # 4. 오른쪽 무릎 각도
        angle_R_knee = ac.calculate_angle_3(
            joints["R_hip"], joints["R_knee"], joints["R_ankle"])
        angle_L_knee = ac.calculate_angle_3(
            joints["L_hip"], joints["L_knee"], joints["L_ankle"])

        return { "R Elbow": round(angle_R_elbow, 2),
              "L Elbow": round(angle_L_elbow, 2),
              "R Shoulder": round(angle_R_shoulder, 2),
              "L Shoulder": round(angle_L_shoulder, 2),
              "Body Twist": round(angle_body_twist, 2),
              "R Knee": round(angle_R_knee, 2),
              "L Knee": round(angle_L_knee, 2)}

class BatterAnalysisTool(PitcherAnalysisTool):
    def calc_joints(self, joints):
        result = super().calc_joints(joints)
        angle_R_wrist = ac.calculate_angle_3(
            joints["R_elbow"], joints["R_wrist"], joints["R_pinky_tip"])
        angle_L_wrist = ac.calculate_angle_3(
            joints["L_elbow"], joints["L_wrist"], joints["L_pinky_tip"])
        result["R wrist"] = angle_R_wrist
        result["L wrist"] = angle_L_wrist
        return result
    