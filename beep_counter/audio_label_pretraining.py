import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from skimage.transform import resize

def convert_to_spectrogram(audio_path, n_fft=2048):
    y, sr = librosa.load(audio_path, sr=22050)
    
    # If the audio is shorter than n_fft, pad it with zeros
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), constant_values=(0, 0))
    
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spectrogram = librosa.power_to_db(spectrogram)
    return spectrogram

# Prepare data
spectrograms = [convert_to_spectrogram(os.path.join('/home/torque/github/main/Torque-AI/beep_counter/data/segments', f)) for f in os.listdir('/home/torque/github/main/Torque-AI/beep_counter/data/segments')]
spectrograms_resized = [resize(s, (128, 128, 1)) for s in spectrograms]
X = np.array(spectrograms_resized)
y = np.array([0 if 'beep' in f else 1 for f in os.listdir('/home/torque/github/main/Torque-AI/beep_counter/data/segments')])  # Encode labels (0: beep, 1: non-beep)

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Save the model
# Save the model in SavedModel format
model.save('trained_model')


