import cv2
import mediapipe as mp
import math

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                    min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# 기울기 계산 함수 (어깨 각도)
def calculate_angle(point1, point2):
    angle = math.degrees(math.atan2(point2.y - point1.y, point2.x - point1.x))
    return angle


# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 영상을 가져올 수 없습니다.")
        break

    # BGR 이미지를 RGB로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 자세 감지
    results = pose.process(image)

    # BGR로 다시 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # 랜드마크 그리기
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 좌표 가져오기
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # 기울기 계산 (어깨의 수평 기울기)
        angle = calculate_angle(left_shoulder, right_shoulder)

        # 기울기 화면에 표시
        cv2.putText(image, f'Tilt: {int(angle)} degrees', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

    # 이미지 출력
    cv2.imshow('Upper Body Posture Detection', image)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
