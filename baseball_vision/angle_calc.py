import numpy as np

def calculate_angle(vec1, vec2):
    # 코사인 값 계산
    # np.dot(ba, bc)는 내적, np.linalg.norm()은 벡터의 크기
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # 아크코사인으로 라디안 각도 계산
    angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) # clip으로 부동 소수점 오차 방지

    # 라디안을 도로 변환
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def calculate_angle_3(a, b, c):
    return calculate_angle(a - b, c - b)

def calculate_angle_4(a, b, c, d):
    v1 = np.array([a[0] - b[0], a[2] - b[2]]);
    v2 = np.array([c[0] - d[0], c[2] - d[2]])
    
    # 아크코사인으로 라디안 각도 계산
    angle_radians = np.arctan2(v1[0]*v2[1] - v1[1]*v2[0], v1[0]*v2[0] + v1[1]*v2[1])
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees