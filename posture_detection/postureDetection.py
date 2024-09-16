import cv2
import mediapipe as mp
import math

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                    min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# 기울기 계산 함수 (가슴과 머리 사이의 각도)
def calculate_angle(point1, point2):
    angle = math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))
    return angle


# 가슴의 중앙 좌표 계산 함수
def get_chest_center(left_shoulder, right_shoulder):
    center_x = (left_shoulder.x + right_shoulder.x) / 2
    center_y = (left_shoulder.y + right_shoulder.y) / 2
    return (center_x, center_y)


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

        # 랜드마크 좌표 가져오기
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = landmarks[mp_pose.PoseLandmark.NOSE]

        # 가슴의 중앙 좌표 계산
        chest_center = get_chest_center(left_shoulder, right_shoulder)

        # 기울기 계산 (가슴과 머리 사이의 각도)
        angle = abs(calculate_angle((nose.x, nose.y), chest_center))

        # 이미지 크기와 비율에 맞춰 좌표 변환
        image_height, image_width, _ = image.shape
        chest_point = (int(chest_center[0] * image_width), int(chest_center[1] * image_height))
        nose_point = (int(nose.x * image_width), int(nose.y * image_height))

        # 가슴 중앙과 코 사이에 선 그리기
        cv2.line(image, chest_point, nose_point, (0, 255, 0), 2)

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
