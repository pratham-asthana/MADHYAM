from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, TimeDistributed, Flatten, Dropout

# Defining the CNN feature extractor
def create_feature_extractor(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten()
    ])
    return model

input_shape = (64, 64, 3)  
sequence_length = 30  

feature_extractor = create_feature_extractor(input_shape)

model = Sequential([
    TimeDistributed(feature_extractor, input_shape=(sequence_length, *input_shape)),
    LSTM(64, return_sequences=True),
    Dropout(0.2), 
    LSTM(64),
    Dropout(0.2),  
    Dense(128, activation='relu'),
    Dense(47, activation='softmax') 
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
