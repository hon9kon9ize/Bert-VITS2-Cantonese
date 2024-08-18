"""
Serve API
"""
import struct

import torch
import utils
from config import config
import webui
from flask import Flask, request, jsonify, Response

print("init")
from infer import get_net_g

host = '0.0.0.0'
port = config.server_config.port
app = Flask(__name__)

print('torch: ', torch.__version__)

syn_default_param = params = {
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
    'text': '從前有個小朋友唔聽話'
}


def warmup():
    synthesize(syn_default_param.copy())


def synthesize(params):
    p = syn_default_param.copy()
    p.update(params)
    _, data = webui.tts_fn(**p)
    return data[0], data[1]


if __name__ == "__main__":

    device = config.webui_config.device
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    webui.net_g = get_net_g(
        model_path=config.webui_config.model, device=device, hps=hps
    )
    webui.device = device
    webui.hps = hps

    warmup()

    print("init complete")


    @app.route('/api/v1/synthesize', methods=['POST'])
    def api_v1_synthesize():

        # Placeholder for TTS synthesis (use an actual TTS library like pyttsx3, gTTS, or similar)
        try:
            # int, nparray, dtype=np.int16
            sample_rate, audio_concat = synthesize(request.json)

            # Convert audio data to a list for JSON serialization
            # convert nparray to binaries, also pack with sample rate
            audio_data_binary = struct.pack(f'i{len(audio_concat)}h', sample_rate, *audio_concat)
            response = Response(audio_data_binary, mimetype='application/octet-stream')

            return response, 200
        except Exception as err:
            print(err)
            return jsonify({'message': 'err'}), 500


    if __name__ == '__main__':
        app.run(host=host, port=port, debug=False)
