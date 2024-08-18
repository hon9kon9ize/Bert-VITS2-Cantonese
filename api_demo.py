"""
This is demo of synthesis through api
"""
import os.path
import struct

import requests
import numpy as np
from scipy.io import wavfile

host = '127.0.0.1'
port = 5000  # 18100 for docker
api = 'api/v1/synthesize'


def synthesis_request(params: dict):
    url = f'http://{host}:{port}/{api}'
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=params, headers=headers)
        if response.status_code == 200:
            return response.content
        else:
            print(f"Failed to get response: {response.status_code}")
    except Exception as err:
        print(err)
        return None


if __name__ == "__main__":
    import pygame

    # Initialize the pygame mixer
    pygame.mixer.init()

    # Example usage
    params_template = {
        'emotion': 'Happy',
        'language': 'YUE',
        'length_scale': 1,
        'noise_scale': 0.6,
        'noise_scale_w': 0.9,
        'prompt_mode': 'Text prompt',
        'reference_audio': None,
        'sdp_ratio': 0.5,
        'speaker': 'YUE_SPK500',
        'style_text': '',
        'style_weight': 0.7,
        'text': ''
    }

    if not os.path.exists('demo_output'):
        os.mkdir('demo_output')

    while True:
        text = input(">>")
        p = params_template.copy()
        p['text'] = text
        data = synthesis_request(p)

        if data:
            try:
                # Unpack the first 4 bytes to get the sample rate
                sample_rate = struct.unpack('i', data[:4])[0]
                # The rest is the audio data
                audio_data = struct.unpack(f'{(len(data) - 4) // 2}h', data[4:])
                audio_array = np.array(audio_data, dtype=np.int16)
                # save as wav
                file_path = 'demo_output/output_audio.wav'
                wavfile.write(file_path, sample_rate, audio_array)

                # playback
                sound = pygame.mixer.Sound(file_path)
                sound.play()
                pygame.time.wait(int(sound.get_length() * 1000))
            except Exception as err:
                print(err)

