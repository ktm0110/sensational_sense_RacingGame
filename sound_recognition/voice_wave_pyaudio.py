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
ax.set_ylim(-6000, 6000)  # Y축 범위 설정
ax.set_xlim(0, CHUNK)
plt.title("Real-time Voice Waveform")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

print("실시간 음성 파형을 분석합니다...")

# 초기 상태는 배경 소음으로 설정
sound_state = "배경 소음"
state_locked = False  # 상태가 고정되었는지 여부를 저장하는 플래그

# 실시간 파형 플롯팅 및 소리 구분 루프
while True:
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # 음성 데이터의 중앙값을 이동하여 노이즈 제거
        adjusted_audio_data = audio_data - np.mean(audio_data)

        # 최대 진폭 계산
        peak_amplitude = np.max(np.abs(adjusted_audio_data))

        # 소리 상태 결정
        if state_locked:
            # 현재 `끼이익~` 상태가 유지 중이면 소리가 멈출 때까지 상태 유지
            if peak_amplitude < 1000:
                # 소리가 충분히 작아지면 상태 해제
                state_locked = False
                sound_state = "배경 소음"
        else:
            # 상태가 고정되지 않은 경우에만 새로운 소리 감지
            if peak_amplitude < 1000:
                sound_state = "배경 소음"
            elif peak_amplitude < 3000:
                sound_state = "부우웅~ (Car Forward)"
            else:
                # 끼이익 상태로 전환하고 상태를 고정
                sound_state = "끼이익~ (Car Backward)"
                state_locked = True

        # 데이터 업데이트
        line.set_ydata(adjusted_audio_data)

        # 상태 출력
        print(f"Detected Sound: {sound_state}")

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
