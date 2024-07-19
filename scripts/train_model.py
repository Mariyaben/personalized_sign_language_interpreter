# train_model.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical
import pickle

def load_data(file_path="data/processed/data.npz"):
    data = np.load(file_path)
    return data['data'], data['labels']

def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    data, labels = load_data()
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)
    
    print("Data shape:", data.shape)  # Debugging line
    print("Labels shape:", labels_categorical.shape)  # Debugging line
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels_categorical, test_size=0.2, random_state=42)
    
    print("X_train shape before reshape:", X_train.shape)  # Debugging line
    print("X_test shape before reshape:", X_test.shape)  # Debugging line
    
    X_train = X_train.reshape((X_train.shape[0], 21, 3))
    X_test = X_test.reshape((X_test.shape[0], 21, 3))
    
    print("X_train shape after reshape:", X_train.shape)  # Debugging line
    print("X_test shape after reshape:", X_test.shape)  # Debugging line
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(label_encoder.classes_)
    
    model = create_model(input_shape, num_classes)
    model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
    
    model.save("models/sign_language_model.h5")
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
