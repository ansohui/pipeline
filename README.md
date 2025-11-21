yamnet 다운

mkdir -p ~/yamnet_local
wget -O ~/yamnet_local/yamnet.tar.gz https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed
tar -xvf ~/yamnet_local/yamnet.tar.gz -C ~/yamnet_local

파이썬 환경 세팅

# 파이썬 3.10 기반 새 환경
conda create -n siren-all python=3.10
conda activate siren-all

conda install -c conda-forge \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    scipy \
    librosa \
    soundfile \
    joblib

# Mac / CPU 기준 (GPU 안 씀)
conda install -c pytorch pytorch torchaudio

conda activate siren-all

pip install tensorflow-macos tensorflow-metal tensorflow-hub tensorflow-io

# python 셸에서 한 번 테스트
python - << 'EOF'
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
print("matplotlib backend:", matplotlib.get_backend())
EOF


python - << 'EOF'
import tensorflow as tf
import tensorflow_hub as hub

print("TF version:", tf.__version__)
model = hub.load("https://tfhub.dev/google/yamnet/1")
print("YAMNet loaded OK")
EOF


conda activate siren-all
