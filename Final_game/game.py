import sys
import pygame
import random
import cv2
import mediapipe as mp
import math
import pyaudio
import numpy as np

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                    min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# PyAudio 초기화 (음성 인식을 위한 설정)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# 기울기 계산 함수 (가슴과 머리 사이의 각도)
def calculate_angle(point1, point2):
    angle = math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))
    return angle

# 가슴의 중앙 좌표 계산 함수
def get_chest_center(left_shoulder, right_shoulder):
    center_x = (left_shoulder.x + right_shoulder.x) / 2
    center_y = (left_shoulder.y + right_shoulder.y) / 2
    return (center_x, center_y)

# 게임 화면 크기
WINDOW_WIDTH = 550
WINDOW_HEIGHT = 800

# 색상
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)

# 소스 디렉토리
DIRCARS = "cars/"
DIRSOUND = "sound/"

# 기본 변수
STAGE = 1
CAR_COUNT = 5
SCORE = 0
STAGESCORE = 0
STAGESTAIR = 1000
PNUMBER = 5
CARS = []

class Car:
    car_image = [f'Car{i:02d}.png' for i in range(1, 41)]

    def __init__(self, x=0, y=0, dx=0, dy=0):
        self.image = ""
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.rect = ""

    def load_car(self, p=""):
        if p == "p":
            self.image = pygame.image.load(DIRCARS + "Player.png")
            self.image = pygame.transform.scale(self.image, (40, 102))
            self.rect = self.image.get_rect()
            self.rect.x = self.x
            self.rect.y = self.y
        else:
            self.image = pygame.image.load(DIRCARS + random.choice(self.car_image))
            self.rect = self.image.get_rect()
            carwidth = max(self.rect.width - 15, 55)
            carheight = round((self.rect.height * carwidth) / self.rect.width)
            self.image = pygame.transform.scale(self.image, (carwidth, carheight))
            self.rect.x = random.randrange(0, WINDOW_WIDTH - self.rect.width)
            self.rect.y = random.randrange(-150, -50)
            speed = min(STAGE + 5, 15)
            self.dy = random.randint(5, speed)

    def draw_car(self):
        SCREEN.blit(self.image, [self.rect.x, self.rect.y])

    def move_x(self):
        self.rect.x += self.dx

    def move_y(self):
        self.rect.y += self.dy

    def check_screen(self):
        if self.rect.right > WINDOW_WIDTH or self.rect.x < 0:
            self.rect.x -= self.dx
        if self.rect.bottom > WINDOW_HEIGHT or self.rect.y < 0:
            self.rect.y -= self.dy

    def check_collision(self, car, distance=0):
        if (self.rect.top + distance < car.rect.bottom) and (car.rect.top < self.rect.bottom - distance) and (
                self.rect.left + distance < car.rect.right) and (car.rect.left < self.rect.right - distance):
            return True
        else:
            return False

def draw_score():
    font_01 = pygame.font.SysFont("FixedSsy", 30, True, False)
    text_score = font_01.render("Score : " + str(SCORE), True, BLACK)
    SCREEN.blit(text_score, [15, 15])

    text_stage = font_01.render("STAGE : " + str(STAGE), True, BLACK)
    text_stage_rect = text_stage.get_rect()
    text_stage_rect.centerx = round(WINDOW_WIDTH / 2)
    SCREEN.blit(text_stage, [text_stage_rect.x, 15])

    for i in range(PNUMBER):
        if i < 5:
            pimage = pygame.image.load(DIRCARS + 'Player.png')
            pimage = pygame.transform.scale(pimage, (15, 38))
            px = WINDOW_WIDTH - 20 - (i * 30)
            SCREEN.blit(pimage, [px, 15])
        else:
            text_pnumber = font_01.render("+" + str(PNUMBER - 5), True, WHITE)
            text_pnumber_x = WINDOW_WIDTH - 30 - (5 * 30)
            SCREEN.blit(text_pnumber, [text_pnumber_x, 25])

def increase_score():
    global SCORE, STAGE, STAGESCORE
    SCORE += 10
    stair = STAGESTAIR if STAGE == 1 else (STAGE - 1) * STAGESTAIR
    if SCORE >= STAGESCORE + stair:
        STAGE += 1
        STAGESCORE += stair

def audio_recognition():
    # 음성 데이터 읽기 및 진폭 계산
    data = stream.read(CHUNK, exception_on_overflow=False)
    audio_data = np.frombuffer(data, dtype=np.int16)
    peak_amplitude = np.max(np.abs(audio_data))

    # 소리 구분 조건
    if peak_amplitude < 1000:
        return "배경 소음"
    elif peak_amplitude < 3000:
        return "부우웅"
    else:
        return "끼이익"

def main():
    global SCREEN, CAR_COUNT, WINDOW_WIDTH, WINDOW_HEIGHT, PNUMBER
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
    pygame.display.set_caption("Racing Car Game")
    windowicon = pygame.image.load(DIRCARS + 'icon.png').convert_alpha()
    pygame.display.set_icon(windowicon)
    clock = pygame.time.Clock()

    player = Car(round(WINDOW_WIDTH / 2), round(WINDOW_HEIGHT - 150), 0, 0)
    player.load_car("p")

    for i in range(CAR_COUNT):
        car = Car(0, 0, 0, 0)
        car.load_car()
        CARS.append(car)

    cap = cv2.VideoCapture(0)
    playing = True

    while playing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                playing = False
                pygame.quit()
                sys.exit()

        # 음성 인식 결과에 따른 조작
        sound = audio_recognition()
        if sound == "부우웅":
            player.dy = -5
        elif sound == "끼이익":
            player.dy = 5
        else:
            player.dy = 0

        # 카메라로부터 프레임 가져오기
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 영상을 가져올 수 없습니다.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            chest_center = get_chest_center(left_shoulder, right_shoulder)
            angle = abs(calculate_angle((nose.x, nose.y), chest_center))

            if angle > 100:
                player.dx = -5
            elif angle < 80:
                player.dx = 5
            else:
                player.dx = 0

        # 배경색을 회색으로
        SCREEN.fill(GRAY)

        # 플레이어 표시 및 이동 제어
        player.draw_car()
        player.move_x()
        player.move_y()
        player.check_screen()

        # 다른 자동차 이동 및 충돌 처리
        for car in CARS:
            car.draw_car()
            car.rect.y += car.dy
            if car.rect.y > WINDOW_HEIGHT:
                increase_score()
                car.load_car()

            if player.check_collision(car, 5):
                PNUMBER -= 1
                if player.rect.x > car.rect.x:
                    car.rect.x -= car.rect.width + 10
                else:
                    car.rect.x += car.rect.width + 10

        draw_score()
        pygame.display.flip()
        clock.tick(60)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
