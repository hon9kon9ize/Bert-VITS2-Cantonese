# Hugging face spaces (CPU) でエディタ (server_editor.py) のデプロイ用

# See https://huggingface.co/docs/hub/spaces-sdks-docker-first-demo

FROM python:3.10

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

RUN pip install --no-cache-dir --upgrade pip

COPY --chown=user . $HOME/app

# RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir

RUN pip install --no-cache-dir -r $HOME/app/requirements.txt

# 必要に応じて制限を変更してください
CMD ["python", "serve.py"]