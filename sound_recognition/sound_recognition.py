import pyaudio
import numpy as np
from scipy.fftpack import fft

# PyAudio로 마이크 설정
FORMAT = pyaudio.paInt16  # 16비트 오디오
CHANNELS = 1  # 모노 입력
RATE = 44100  # 샘플링 레이트
CHUNK = 1024  # 읽을 데이터 크기

# 주파수 분석 범위 설정
BUWOOONG_FREQ_RANGE = (50, 150)  # 예시: '부우웅' 소리에 해당하는 저주파
KKIIIIK_FREQ_RANGE = (3000, 6000)  # 예시: '끼이익' 소리에 해당하는 고주파

# PyAudio 초기화
p = pyaudio.PyAudio()

# 마이크 스트림 시작
stream = p.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True, frames_per_buffer=CHUNK)

def analyze_frequency(data):
    """주파수 분석 함수"""
    # 데이터 푸리에 변환
    audio_data = np.frombuffer(data, np.int16)
    fft_data = fft(audio_data)
    freqs = np.fft.fftfreq(len(fft_data)) * RATE

    # 절대값 취하여 크기 분석
    abs_fft_data = np.abs(fft_data)

    # 주파수 범위 확인
    return freqs, abs_fft_data

def detect_sound(freqs, abs_fft_data, freq_range):
    """주파수 범위에 해당하는 소리가 있는지 감지"""
    for i, freq in enumerate(freqs):
        if freq_range[0] <= abs(freq) <= freq_range[1]:
            if abs_fft_data[i] > 1000:  # 임계값 설정
                return True
    return False

print("Listening...")

try:
    while True:
        data = stream.read(CHUNK)
        freqs, abs_fft_data = analyze_frequency(data)

        # '부우웅' 소리 감지
        if detect_sound(freqs, abs_fft_data, BUWOOONG_FREQ_RANGE):
            print("부우웅 소리 감지!")

        # '끼이익' 소리 감지
        if detect_sound(freqs, abs_fft_data, KKIIIIK_FREQ_RANGE):
            print("끼이익 소리 감지!")

except KeyboardInterrupt:
    print("종료")

# 스트림 종료
stream.stop_stream()
stream.close()
p.terminate()
