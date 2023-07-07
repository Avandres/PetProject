import math
import struct
from collections import deque

import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECOND = 1
THRESHOLD = 0.02
WAVE_OUTPUT_FILENAME = 'archive/Audio Commands/22/0.wav'

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

start_stop_flag = False
print('Start...')

def get_rms(data):
    count = len(data) / 2
    format = "%dh" % (count)
    shorts = struct.unpack(format, data)

    sum_squares = 0.0
    for sample in shorts:
        n = sample / 32768.0
        sum_squares += n * n

    return math.sqrt(sum_squares / count)

frames = []
last_data = deque(maxlen=20)
timer = 0
while True:
    data = stream.read(CHUNK)
    last_data.append(data)
    if get_rms(data) > THRESHOLD:
        if not start_stop_flag:
            frames += list(last_data)
        start_stop_flag = True
        timer = 0
    if start_stop_flag:
        if timer >= RATE // CHUNK * RECORD_SECOND:
            break
        timer += 1
        frames.append(data)

print('...End.')

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

