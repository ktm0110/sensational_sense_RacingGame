import tensorflow as tf
import numpy as np

# TensorFlow Lite 모델 로드
interpreter = tf.lite.Interpreter(model_path=r"D:\SoundControl_CarGame\sound_recognition\soundclassifier_with_metadata.tflite")
interpreter.allocate_tensors()

# 입력/출력 정보 확인
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 데이터 생성 (예: 랜덤 데이터)
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

# 모델에 입력 데이터 설정
interpreter.set_tensor(input_details[0]['index'], input_data)

# 모델 실행
interpreter.invoke()

# 출력 결과 가져오기
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output:", output_data)
