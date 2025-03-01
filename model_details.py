from tensorflow.keras.models import load_model 
model = load_model('gesture_recognition_model.h5')
model.summary()