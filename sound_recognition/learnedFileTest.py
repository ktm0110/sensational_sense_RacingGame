import sounddevice as sd
import numpy as np
import tensorflow as tf

# 모델을 불러오기 전 레이어 연결을 재정의
model = tf.keras.models.load_model('output_model.h5')

# 레이어 연결을 재정의
input_layer = model.layers[0].input
output_layer = model.layers[-1].output

# Functional API로 새 모델을 구성
new_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
new_model.save('corrected_model.h5')

# 오디오 설정
SAMPLE_RATE = 44100  # 오디오 샘플링 레이트
DURATION = 2  # 녹음할 길이 (초 단위)

# 예측을 위한 클래스 이름 정의 (라벨에 맞게 설정)
class_names = ['acceleration', 'deceleration', '배경 소음']


def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    """마이크로부터 오디오를 녹음하고 넘파이 배열로 반환."""
    print("녹음 시작...")
    recording = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
    sd.wait()  # 녹음이 끝날 때까지 대기
    print("녹음 완료.")
    return np.squeeze(recording)  # 차원을 맞추기 위해 squeeze


def preprocess_audio(audio, sample_rate=SAMPLE_RATE):
    """녹음된 오디오 데이터를 모델 입력에 맞게 전처리."""
    # 샘플링 레이트를 16000Hz로 조정
    if sample_rate != 16000:
        from scipy.signal import resample
        audio = resample(audio, int(16000 * len(audio) / sample_rate))

    # 모델 입력 크기에 맞추기 (43, 232, 1)로 reshape
    spectrogram = np.reshape(audio, (43, 232, 1))  # 입력에 맞는 형태로 변경
    return np.array([spectrogram])


def predict_audio(audio_data):
    """모델을 사용하여 오디오 데이터의 예측 결과를 반환."""
    # 전처리된 오디오 데이터를 모델에 전달하여 예측 수행
    processed_audio = preprocess_audio(audio_data)
    prediction = model.predict(processed_audio)
    class_index = np.argmax(prediction)
    return class_names[class_index]


# 실시간 녹음 및 예측 반복
try:
    while True:
        print("\n음성 인식 중... (Ctrl+C로 종료)")
        audio_data = record_audio()  # 오디오 녹음
        result = predict_audio(audio_data)  # 모델 예측
        print(f"예측 결과: {result}")  # 예측 결과 출력

        # 예측 결과에 따라 동작 제어
        if result == 'acceleration':
            print("앞으로 전진!")
        elif result == 'deceleration':
            print("뒤로 후진!")
        else:
            print("배경 소음 감지됨.")

except KeyboardInterrupt:
    print("음성 인식 종료.")
