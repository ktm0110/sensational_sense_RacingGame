import sounddevice as sd
import numpy as np
import tensorflow as tf
import scipy.signal

# 미리 훈련된 모델 불러오기
model = tf.keras.models.load_model('output_model.h5')


# 입력된 오디오 데이터를 MFCC로 변환하는 함수
def extract_features(audio, sample_rate=16000):
    # MFCC 추출
    frequencies, times, spectrogram = scipy.signal.spectrogram(audio, sample_rate)
    return np.log(spectrogram.T + 1e-10)  # 로그 스펙트로그램 반환


# 예측 수행 함수
def predict_sound(audio, sample_rate=16000):
    features = extract_features(audio, sample_rate)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    return np.argmax(prediction)


# 마이크에서 실시간으로 오디오 데이터를 수집하여 예측
def audio_callback(indata, frames, time, status):
    # 실시간 오디오 데이터를 분석
    sample_rate = 16000  # 모델 훈련 시의 샘플 레이트
    audio_data = np.mean(indata, axis=1)  # 스테레오 -> 모노 변환
    prediction = predict_sound(audio_data, sample_rate)

    # 결과 출력
    if prediction == 0:
        print("부우웅~ (Forward)")
    elif prediction == 1:
        print("끼이익~ (Backward)")
    else:
        print("배경 소음 (Background noise)")


# 마이크 스트림 시작
with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
    print("실시간 음성 인식 중... (종료하려면 Ctrl+C)")
    sd.sleep(10000)  # 10초 동안 실행
