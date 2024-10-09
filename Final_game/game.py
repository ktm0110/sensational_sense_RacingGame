import sys
import pygame
import random
import cv2
import mediapipe as mp
import math
import pyaudio
import numpy as np
import threading

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

# 카메라와 오디오 데이터를 사용할 글로벌 변수
tilt_angle = 90
sound_command = "배경 소음"
camera_running = True
audio_running = True
audio_peak = 0

# 게임 화면 크기
WINDOW_WIDTH = 550
WINDOW_HEIGHT = 800

# 색상
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# 소스 디렉토리
DIRCARS = "cars/"

# 기본 변수
STAGE = 1
CAR_COUNT = 5
SCORE = 0
STAGESCORE = 0
STAGESTAIR = 1000
LIFE_COUNT = 5
CARS = []

# 게임 상태 변수
game_state = "LOBBY"  # LOBBY, PLAYING, GAME_OVER

# 기울기 계산 함수 (가슴과 머리 사이의 각도)
def calculate_angle(point1, point2):
    angle = math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))
    return angle

# 가슴의 중앙 좌표 계산 함수
def get_chest_center(left_shoulder, right_shoulder):
    center_x = (left_shoulder.x + right_shoulder.x) / 2
    center_y = (left_shoulder.y + right_shoulder.y) / 2
    return (center_x, center_y)

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

    for i in range(LIFE_COUNT):
        if i < 5:
            pimage = pygame.image.load(DIRCARS + 'Player.png')
            pimage = pygame.transform.scale(pimage, (15, 38))
            px = WINDOW_WIDTH - 20 - (i * 30)
            SCREEN.blit(pimage, [px, 15])
        else:
            text_life_count = font_01.render("+" + str(LIFE_COUNT - 5), True, WHITE)
            text_life_count_x = WINDOW_WIDTH - 30 - (5 * 30)
            SCREEN.blit(text_life_count, [text_life_count_x, 25])

def draw_audio_level():
    # 상단에 소리의 크기를 바 형태로 표시
    bar_width = int((audio_peak / 5000) * WINDOW_WIDTH)
    bar_width = min(bar_width, WINDOW_WIDTH)  # 바의 최대 너비를 화면 너비로 제한
    pygame.draw.rect(SCREEN, BLUE, (0, 0, bar_width, 15))

    # 빨간색 선으로 배경 소음, 부우웅, 끼이익 기준 표시
    pygame.draw.line(SCREEN, RED, (int((1000 / 5000) * WINDOW_WIDTH), 0), (int((1000 / 5000) * WINDOW_WIDTH), 15), 2)
    pygame.draw.line(SCREEN, RED, (int((2000 / 5000) * WINDOW_WIDTH), 0), (int((2000 / 5000) * WINDOW_WIDTH), 15), 2)

def increase_score():
    global SCORE, STAGE, STAGESCORE
    SCORE += 10
    stair = STAGESTAIR if STAGE == 1 else (STAGE - 1) * STAGESTAIR
    if SCORE >= STAGESCORE + stair:
        STAGE += 1
        STAGESCORE += stair

def camera_thread():
    global tilt_angle, camera_running
    cap = cv2.VideoCapture(0)

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        # 좌우 반전 적용
        frame = cv2.flip(frame, 1)

        # Mediapipe 처리를 위해 RGB로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            chest_center = get_chest_center(left_shoulder, right_shoulder)
            tilt_angle = abs(calculate_angle((nose.x, nose.y), chest_center))

            chest_point = (int(chest_center[0] * frame.shape[1]), int(chest_center[1] * frame.shape[0]))
            nose_point = (int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0]))
            cv2.line(frame, chest_point, nose_point, (0, 255, 0), 2)

            # 기울기 화면에 표시
            cv2.putText(frame, f'ANGLE: {int(tilt_angle)} degrees', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 카메라 화면 출력 (BGR로 출력)
        cv2.imshow('Body Posture Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera_running = False
            break

    cap.release()
    cv2.destroyAllWindows()

def audio_thread():
    global sound_command, audio_running, audio_peak
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    while audio_running:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        audio_peak = np.max(np.abs(audio_data))

        # 소리 구분 조건
        if audio_peak < 1000:
            sound_command = "배경 소음"
        elif audio_peak < 2000:
            sound_command = "부우웅"
        else:
            sound_command = "끼이익"

    stream.stop_stream()
    stream.close()

def draw_lobby():
    SCREEN.fill(WHITE)
    lobby_image = pygame.image.load(DIRCARS + 'lobby.png')
    lobby_image = pygame.transform.scale(lobby_image, (WINDOW_WIDTH, int(WINDOW_WIDTH * (6307 / 6501))))
    SCREEN.blit(lobby_image, (0, (WINDOW_HEIGHT - lobby_image.get_height()) // 2))

    font = pygame.font.SysFont("FixedSsy", 50, True, False)
    text = font.render("Racing Car Game", True, BLACK)
    SCREEN.blit(text, [WINDOW_WIDTH // 2 - text.get_width() // 2, 50])

    font_small = pygame.font.SysFont("FixedSsy", 40, True, False)
    start_text = font_small.render("Press ENTER to Start", True, GREEN)
    SCREEN.blit(start_text, [WINDOW_WIDTH // 2 - start_text.get_width() // 2, 670])

def draw_game_over():
    SCREEN.fill(WHITE)
    end_image = pygame.image.load(DIRCARS + 'end.png')
    end_image = pygame.transform.scale(end_image, (WINDOW_WIDTH, int(WINDOW_WIDTH * (1024 / 1792))))
    SCREEN.blit(end_image, (0, (WINDOW_HEIGHT - end_image.get_height()) // 2))

    font = pygame.font.SysFont("FixedSsy", 70, True, False)
    text = font.render("Game Over", True, BLACK)
    SCREEN.blit(text, [WINDOW_WIDTH // 2 - text.get_width() // 2, 50])

    font_small = pygame.font.SysFont("FixedSsy", 40, True, False)
    restart_text = font_small.render("Press ENTER to Restart", True, GREEN)
    SCREEN.blit(restart_text, [WINDOW_WIDTH // 2 - restart_text.get_width() // 2, 650])

    # 점수 표시
    score_text = font_small.render(f'Your Score: {SCORE}', True, BLACK)
    SCREEN.blit(score_text, [WINDOW_WIDTH // 2 - score_text.get_width() // 2, 100])

def main():
    global SCREEN, CAR_COUNT, WINDOW_WIDTH, WINDOW_HEIGHT, LIFE_COUNT, camera_running, audio_running, game_state, SCORE, STAGE, STAGESCORE, LIFE_COUNT
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
    pygame.display.set_caption("Racing Car Game")
    clock = pygame.time.Clock()

    player = Car(round(WINDOW_WIDTH / 2), round(WINDOW_HEIGHT - 150), 0, 0)
    player.load_car("p")

    for i in range(CAR_COUNT):
        car = Car(0, 0, 0, 0)
        car.load_car()
        CARS.append(car)

    # 카메라 및 오디오 스레드 시작
    camera_thread_instance = threading.Thread(target=camera_thread)
    audio_thread_instance = threading.Thread(target=audio_thread)
    camera_thread_instance.start()
    audio_thread_instance.start()

    playing = True

    while playing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                playing = False
                camera_running = False
                audio_running = False
                camera_thread_instance.join()
                audio_thread_instance.join()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if game_state == "LOBBY":
                        game_state = "PLAYING"
                    elif game_state == "GAME_OVER":
                        # 게임 리셋
                        SCORE = 0
                        STAGE = 1
                        STAGESCORE = 0
                        LIFE_COUNT = 5
                        player.rect.x = round(WINDOW_WIDTH / 2)
                        player.rect.y = round(WINDOW_HEIGHT - 150)
                        for car in CARS:
                            car.load_car()
                        game_state = "PLAYING"

        if game_state == "LOBBY":
            draw_lobby()
        elif game_state == "PLAYING":
            # 음성 인식 결과에 따른 조작
            if sound_command == "부우웅":
                player.dy = -5
            elif sound_command == "끼이익":
                player.dy = 5
            else:
                player.dy = 0

            # 기울기에 따른 좌우 이동 제어
            if tilt_angle < 80:
                player.dx = -5
            elif tilt_angle > 100:
                player.dx = 5
            else:
                player.dx = 0

            # 배경색을 회색으로 설정
            SCREEN.fill(GRAY)

            # 오디오 레벨 표시
            draw_audio_level()

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
                    LIFE_COUNT -= 1
                    if LIFE_COUNT == 0:
                        game_state = "GAME_OVER"
                    if player.rect.x > car.rect.x:
                        car.rect.x -= car.rect.width + 10
                    else:
                        car.rect.x += car.rect.width + 10

            # 상대 자동차들끼리 충돌 감지, 각 자동차들을 순서대로 서로 비교
            for i in range(CAR_COUNT):
                for j in range(i + 1, CAR_COUNT):
                    # 충돌 후 서로 튕겨 나가게 함.
                    if CARS[i].check_collision(CARS[j]):
                        # 왼쪽에 있는 차는 왼쪽으로 오른쪽 차는 오른쪽으로 튕김
                        if CARS[i].rect.x > CARS[j].rect.x:
                            CARS[i].rect.x += 4
                            CARS[j].rect.x -= 4
                        else:
                            CARS[i].rect.x -= 4
                            CARS[j].rect.x += 4

                        # 위쪽 차는 위로, 아래쪽차는 아래로 튕김
                        if CARS[i].rect.y > CARS[j].rect.y:
                            CARS[i].rect.y += CARS[i].dy
                            CARS[j].rect.y -= CARS[j].dy
                        else:
                            CARS[i].rect.y -= CARS[i].dy
                            CARS[j].rect.y += CARS[j].dy

            draw_score()
        elif game_state == "GAME_OVER":
            draw_game_over()

        pygame.display.flip()
        clock.tick(60)

    camera_running = False
    audio_running = False
    camera_thread_instance.join()
    audio_thread_instance.join()

if __name__ == '__main__':
    main()