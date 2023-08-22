# import numpy as np
# import librosa
# from skimage.transform import resize
# from tensorflow.keras.models import load_model
# import os
# import time

# def convert_to_spectrogram(audio_path, n_fft=2048):
#     y, sr = librosa.load(audio_path, sr=22050)
#     if len(y) < n_fft:
#         y = np.pad(y, (0, n_fft - len(y)), constant_values=(0, 0))
#     spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
#     spectrogram = librosa.power_to_db(spectrogram)
#     return spectrogram

# # Load the saved model
# model = load_model('trained_model')

# # Initialize the global beep count
# beep_count = 0

# # Define the prefix for the audio files and the maximum count of files
# audio_file_prefix = 'output'
# max_file_count = 100

# # Process the audio files one by one
# for i in range(1, max_file_count + 1):
#     audio_file = f'{audio_file_prefix}{i}.mp3'
    
#     # Wait for the file to exist and its size to be stable before trying to load it
#     old_file_size = -1
#     while not os.path.exists(audio_file) or old_file_size != os.path.getsize(audio_file):
#         print(f"Waiting for file {audio_file} to exist and be completely written...")
#         time.sleep(5)  # Wait for 5 seconds before checking again
#         if os.path.exists(audio_file):
#             old_file_size = os.path.getsize(audio_file)

#     time.sleep(5)  # Allow some additional time for the file to become accessible

#     # Convert the audio to a spectrogram
#     spectrogram = convert_to_spectrogram(audio_file)
    
#     # Segment the spectrogram into chunks of approximately 1034 frames (approx 6 seconds)
#     segments = [spectrogram[:, i:i+1034] for i in range(0, spectrogram.shape[1], 1034)]
    
#     # Initialize the beep count for the current audio file
#     beep_count_current = 0

#     # Initialize a variable to keep track of segments to skip
#     skip_segments = 0

#     # Predict each segment and update the beep counts
#     for segment in segments:
#         # Skip this segment if a beep was detected in the last few segments
#         if skip_segments > 0:
#             skip_segments -= 1
#             continue

#         if segment.shape[1] == 1034:  # Only consider segments of the correct size
#             segment_resized = resize(segment, (128, 128))
#             segment_reshaped = segment_resized.reshape(1, 128, 128, 1)
#             prediction = model.predict(segment_reshaped)
#             predicted_class = np.argmax(prediction)
#             if predicted_class == 0:  # 0 is the class for "beep"
#                 beep_count += 1
#                 beep_count_current += 1
#                 skip_segments = 3  # Skip the next 3 segments (about 18 seconds)

#     # Print the beep count for the current audio file
#     print(f"Number of beep sounds in {audio_file}: {beep_count_current}")
#     print(f"Total number of beep sounds: {beep_count}")

#     time.sleep(1)  # Add delay to reduce CPU usage

# # Print the total beep count
# print(f"Total number of beep sounds: {beep_count}")
import numpy as np
import librosa
from skimage.transform import resize
from tensorflow.keras.models import load_model
import os
import time
import requests
from datetime import datetime,timedelta
import csv
def convert_to_spectrogram(audio_path, n_fft=2048):
    y, sr = librosa.load(audio_path, sr=22050)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), constant_values=(0, 0))
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spectrogram = librosa.power_to_db(spectrogram)
    return spectrogram

# Load the saved model
model = load_model('beep_counter/trained_model')

# Initialize the global beep count
beep_count = 0

# Define the prefix for the audio files and the maximum count of files
audio_file_prefix = 'output'
max_file_count = 100000
csv_filename = 'beep_count.csv'
csv_header = ['Timestamp', 'AudioFile', 'BeepCount', 'TotalBeepCount']

    # Create or open the CSV file and write the header
with open(csv_filename, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header only if the file is empty
    if csv_file.tell() == 0:
        csv_writer.writerow(csv_header)
interval_start_time = datetime.now()
interval_duration = timedelta(minutes=15)
# Process the audio files one by one
for i in range(1, max_file_count + 1):
    audio_file = f'{audio_file_prefix}{i}.mp3'
    
    # Wait for the file to exist and its size to be stable before trying to load it
    old_file_size = -1
    while not os.path.exists(audio_file) or old_file_size != os.path.getsize(audio_file):
        print(f"Waiting for file {audio_file} to exist and be completely written...")
        time.sleep(5)  # Wait for 5 seconds before checking again
        if os.path.exists(audio_file):
            old_file_size = os.path.getsize(audio_file)

    time.sleep(5)  # Allow some additional time for the file to become accessible

    # Convert the audio to a spectrogram
    spectrogram = convert_to_spectrogram(audio_file)
    
    # Segment the spectrogram into chunks of approximately 1034 frames (approx 6 seconds)
    segments = [spectrogram[:, i:i+1034] for i in range(0, spectrogram.shape[1], 1034)]
    
    # Initialize the beep count for the current audio file
    beep_count_current = 0

    # Initialize a variable to keep track of segments to skip
    skip_segments = 0

    # Predict each segment and update the beep counts
    for segment in segments:
        # Skip this segment if a beep was detected in the last few segments
        if skip_segments > 0:
            skip_segments -= 1
            continue

        if segment.shape[1] == 1034:  # Only consider segments of the correct size
            segment_resized = resize(segment, (128, 128))
            segment_reshaped = segment_resized.reshape(1, 128, 128, 1)
            prediction = model.predict(segment_reshaped)
            predicted_class = np.argmax(prediction)
            if predicted_class == 0:  # 0 is the class for "beep"
                beep_count += 1
                beep_count_current += 1
                skip_segments = 3  # Skip the next 3 segments (about 18 seconds)
             # After processing an audio file, append the data to the CSV file
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                audio_filename = f'{audio_file_prefix}{i}.mp3'
                data_to_append = [timestamp, audio_filename, beep_count_current, beep_count]

                with open(csv_filename, mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(data_to_append)
    # Print the beep count for the current audio file
    print(f"Number of beep sounds in {audio_file}: {beep_count_current}")
    print(f"Total number of beep sounds: {beep_count}")
    current_time = datetime.now()
    if current_time - interval_start_time >= interval_duration:
        # Save the total beep count for this interval
        interval_start_time = current_time  # Reset the interval start time
        total_beep_count = beep_count
        # Append the total beep count to the CSV file for this interval
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        data_to_append = [timestamp, '', '', total_beep_count]
        with open(csv_filename, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(data_to_append)

    # Print the total beep count
    print(f"Total number of beep sounds: {beep_count}")
    # Send the beep_count_current with a POST request
   

# Print the total beep count
print(f"Total number of beep sounds: {beep_count}")
