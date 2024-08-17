"""
This is demo of synthesis through api
"""
import os.path

import requests
import numpy as np
from scipy.io import wavfile

host = '127.0.0.1'
port = 18100  # 18100 for docker
api = 'api/v1/synthesize'


def synthesis_request(params: dict):
    url = f'http://{host}:{port}/{api}'
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data
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
        result = synthesis_request(p)

        if result:
            audio_data = np.array(result['audio_data'], dtype=np.int16)
            sample_rate = result['sample_rate']

            # save as wav
            file_path = 'demo_output/output_audio.wav'
            wavfile.write(file_path, sample_rate, audio_data)

            # playback
            sound = pygame.mixer.Sound(file_path)
            sound.play()
            pygame.time.wait(int(sound.get_length() * 1000))
