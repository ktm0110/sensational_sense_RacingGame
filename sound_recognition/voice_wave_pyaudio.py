import pyaudio
import numpy as np
import matplotlib.pyplot as plt

# 설정값 정의
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

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

# 초기 그래프 설정
line, = ax.plot(x, np.random.rand(CHUNK))
ax.set_ylim(-4000, 4000)  # Y축 범위 설정
ax.set_xlim(0, CHUNK)
plt.title("Real-time Voice Waveform")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

print("실시간 음성 파형을 분석합니다...")

# 실시간 파형 플롯팅 루프
while True:
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # 음성 데이터의 평균으로 중앙값 이동
        adjusted_audio_data = audio_data - np.mean(audio_data)

        # 데이터 업데이트
        line.set_ydata(adjusted_audio_data)

        # 그래프 업데이트
        fig.canvas.draw()
        fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("종료합니다...")
        break

# 스트림 종료
stream.stop_stream()
stream.close()
audio.terminate()
