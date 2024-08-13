from onnx_modules.V230_OnnxInference import OnnxInferenceSession
import soundfile as sf
import commons
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
import numpy as np

Session = OnnxInferenceSession(
    {
        "enc": "onnx/BertVits2.2PT/BertVits2.2PT_enc_p.onnx",
        "emb_g": "onnx/BertVits2.2PT/BertVits2.2PT_emb.onnx",
        "dp": "onnx/BertVits2.2PT/BertVits2.2PT_dp.onnx",
        "sdp": "onnx/BertVits2.2PT/BertVits2.2PT_sdp.onnx",
        "flow": "onnx/BertVits2.2PT/BertVits2.2PT_flow.onnx",
        "dec": "onnx/BertVits2.2PT/BertVits2.2PT_dec.onnx",
    },
    Providers=["CPUExecutionProvider"],
)


def get_text(text, language_str, style_text=None, style_weight=0.5):
    style_text = None if style_text == "" else style_text
    # 在此处实现当前版本的get_text
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # add blank
    phone = commons.intersperse(phone, 0)
    tone = commons.intersperse(tone, 0)
    language = commons.intersperse(language, 0)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1

    bert_ori = get_bert(
        norm_text, word2ph, language_str, "cpu", style_text, style_weight
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if language_str == "EN":
        en_bert = bert_ori
        yue_bert = np.random.randn(1024, len(phone))
    elif language_str == "YUE":
        en_bert = np.random.randn(1024, len(phone))
        yue_bert = bert_ori
    else:
        raise ValueError("language_str should be EN or YUE")

    assert yue_bert.shape[-1] == len(
        phone
    ), f"Bert seq len {yue_bert.shape[-1]} != {len(phone)}"

    phone = np.asarray(phone)
    tone = np.asarray(tone)
    language = np.asarray(language)
    en_bert = np.asarray(en_bert.T)
    yue_bert = np.asarray(yue_bert.T)

    return en_bert, yue_bert, phone, tone, language


en_bert, yue_bert, x, tone, language = get_text("本身我就係一個言出必達嘅人", "YUE")
sid = np.array([0])

print(x, tone, language)

audio = Session(x, tone, language, en_bert, yue_bert, sid)

# export audio
sf.write("output.wav", audio[0][0], 44100)
