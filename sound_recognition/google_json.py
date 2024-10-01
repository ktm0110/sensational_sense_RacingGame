from tensorflow.keras.models import model_from_json

# JSON 파일로부터 모델 로드
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# 모델 아키텍처 로드
loaded_model = model_from_json(loaded_model_json)

# 가중치 로드 (필요하다면)
loaded_model.load_weights('pretrained_model_weights.h5')

# 모델 컴파일
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
