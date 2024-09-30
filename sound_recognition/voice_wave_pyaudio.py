import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time

# 설정값 정의
FORMAT = pyaudio.paInt16  # 음성 데이터 형식
CHANNELS = 1  # 모노 채널
RATE = 44100  # 샘플링 레이트
CHUNK = 1024  # 읽을 데이터 청크 크기

# PyAudio 객체 생성
audio = pyaudio.PyAudio()

# 마이크 스트림 시작
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

# 실시간 플롯 설정
plt.ion()
fig, ax = plt.subplots()
x = np.arange(0, 2 * CHUNK, 2)
line, = ax.plot(x, np.random.rand(CHUNK))

ax.set_ylim(-32768, 32768)  # 16비트 음성 범위 설정
ax.set_xlim(0, CHUNK)

print("실시간 음성 파형을 분석합니다...")

# 실시간 파형 플롯팅 루프
while True:
    try:
        # 스트림에서 데이터 읽기
        data = stream.read(CHUNK)

        # numpy 배열로 변환 (int16 포맷으로 디코딩)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # 데이터 업데이트
        line.set_ydata(audio_data)

        # 그래프 업데이트
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.01)

    except KeyboardInterrupt:
        print("종료합니다...")
        break

# 스트림 종료
stream.stop_stream()
stream.close()
audio.terminate()
