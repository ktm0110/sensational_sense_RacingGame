import tensorflow as tf
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

# 모델 로드
model_path = "C:/Users/ktmth/my_models/output_model.h5"
model = tf.keras.models.load_model(model_path, compile=False)

# 마이크 설정
SAMPLE_RATE = 16000  # 샘플링 레이트 (Google Teachable Machine에 맞춤)
DURATION = 1         # 입력 오디오 길이 (초)
LABELS = ["Acceleration", "Deceleration", "Background Noise"]  # 예측할 레이블

# 실시간 오디오 입력 처리 및 예측 함수
def audio_callback(indata, frames, time, status):
    # 입력 데이터를 (1, 43, 232, 1) 크기로 변환 (Teachable Machine에서 사용한 입력 형식에 맞춤)
    input_data = np.mean(indata, axis=1)  # 여러 채널(스테레오)을 하나로 통합
    input_data = input_data[:SAMPLE_RATE]  # 첫 1초 데이터만 사용
    input_data = np.expand_dims(input_data, axis=1)  # 채널 추가
    input_data = np.reshape(input_data, (1, 43, 232, 1))

    # 모델 예측
    predictions = model.predict(input_data)
    predicted_index = np.argmax(predictions[0])  # 가장 높은 확률의 인덱스 찾기
    predicted_label = LABELS[predicted_index]

    print(f"Predicted Label: {predicted_label} - Confidence: {predictions[0][predicted_index]:.2f}")

# 마이크 입력 스트림 시작
print("Listening for audio commands...")

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
    sd.sleep(int(DURATION * 1000))  # 1초간 대기
