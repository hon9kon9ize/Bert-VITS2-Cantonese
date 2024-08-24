from onnx_modules import export_onnx
import os

if __name__ == "__main__":
    export_path = "BertVits2.2PT"
    model_path = "/notebooks/bert-vits2/dataset/zoengjyutgaai_saamgwokjinji/models/G_12956.pth"
    config_path = "/notebooks/bert-vits2/dataset/zoengjyutgaai_saamgwokjinji/configs/config.json"
    novq = False
    dev = False
    Extra = "chinese"  # japanese or chinese
    if not os.path.exists("onnx"):
        os.makedirs("onnx")
    if not os.path.exists(f"onnx/{export_path}"):
        os.makedirs(f"onnx/{export_path}")
    export_onnx(export_path, model_path, config_path, novq, dev, Extra)
