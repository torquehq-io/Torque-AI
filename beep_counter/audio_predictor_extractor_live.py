import subprocess

clip_duration = 90  # Duration of each audio clip in seconds
output_file_prefix = 'output'  # Prefix for output filenames
output_file_counter = 1  # Counter for incremental filenames

while True:
    output_file = f'{output_file_prefix}{output_file_counter}.mp3'
    command = f'ffmpeg -i https://t1.arcischain.io:8443/live/5/index.m3u8 -t {clip_duration} -c:a libmp3lame -q:a 2 {output_file}'

    subprocess.call(command, shell=True)
    
    output_file_counter += 1
              