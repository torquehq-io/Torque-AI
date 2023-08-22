import ffmpeg

input_url = "https://t1.arcischain.io:8443/live/5/index.m3u8"
out_audio = "/home/torque/github/main/Torque-AI/beep_counter/out_audio.aac"

try:
    # Copy audio stream to a new file
    (
        ffmpeg
        .input(input_url)
        .output(out_audio, acodec='aac', ab='16k')  # convert audio stream to AAC with a lower bit rate
        .run(capture_stdout=True, capture_stderr=True)
    )

except ffmpeg.Error as e:
    print('stdout:', e.stdout.decode('utf8'))
    print('stderr:', e.stderr.decode('utf8'))
    raise e
