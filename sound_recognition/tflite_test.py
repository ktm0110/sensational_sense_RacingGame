import tensorflow as tf
import numpy as np
import sounddevice as sd

# 모델 파일 경로 설정 (Teachable Machine에서 받은 .tflite 파일)
MODEL_PATH = "soundclassifier_with_metadata.tflite"

# 사운드 입력 설정
SAMPLE_RATE = 44100  # Teachable Machine에서 사용한 샘플링 레이트와 일치시켜야 함
DURATION = 1  # 1초 간격으로 입력 받음
TARGET_SAMPLE_COUNT = 44032  # 모델이 기대하는 입력 데이터 길이

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# 입력/출력 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 데이터 크기 확인
input_shape = input_details[0]['shape']
print(f"모델의 입력 크기: {input_shape}")


# 예측 함수 정의
def predict(audio_data):
    # 오디오 데이터를 모델 입력 크기에 맞게 조정
    if len(audio_data) < TARGET_SAMPLE_COUNT:
        # 데이터 길이가 모자란 경우, 0으로 패딩하여 길이를 맞춤
        audio_data = np.pad(audio_data, (0, TARGET_SAMPLE_COUNT - len(audio_data)), mode='constant')
    else:
        # 데이터가 더 긴 경우, 필요한 만큼 자름
        audio_data = audio_data[:TARGET_SAMPLE_COUNT]

    # 입력 텐서의 형식에 맞게 변환 (1, TARGET_SAMPLE_COUNT)
    audio_data = np.expand_dims(audio_data, axis=0).astype(np.float32)

    # 모델 입력에 데이터 설정
    interpreter.set_tensor(input_details[0]['index'], audio_data)

    # 추론 실행
    interpreter.invoke()

    # 모델 출력 가져오기
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data)


# 실시간 음성 처리 함수
def audio_callback(indata, frames, time, status):
    # 오디오 데이터를 numpy 배열로 변환 후 모델에 전달
    audio_data = np.squeeze(indata)

    # 모델을 사용하여 예측 수행
    prediction = predict(audio_data)

    # 예측 결과 출력
    if prediction == 0:
        print("Acceleration ('부우웅')")
    elif prediction == 1:
        print("Deceleration ('끼이익')")
    else:
        print("Background noise")


# 마이크에서 실시간으로 데이터 수집 및 예측
with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback, blocksize=SAMPLE_RATE * DURATION):
    print("Listening... Press Ctrl+C to stop.")
    while True:
        pass
