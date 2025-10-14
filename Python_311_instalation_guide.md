# Setup - Alternative Python 3.11 installation guide 


Clone the repository
```
git clone https://github.com/juliamachnio/PALM
cd PALM
```

Create an environment using
```
conda create -n palm python=3.11 -y
conda activate palm

pip install --upgrade pip
pip install \
  numpy==1.26.4 \
  pandas==1.5.3 \
  scipy==1.13.1 \
  scikit-learn==1.5.2 \
  matplotlib==3.9.2 \
  pillow==11.0.0 \
  tqdm==4.66.5 \
  pyyaml==6.0.2 \
  yacs==0.1.8 \
  simplejson==3.19.2 \
  termcolor==2.5.0 \
  easydict==1.13 \
  seaborn==0.13.2

pip install \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121

conda install -y -c pytorch faiss-gpu=1.9.0
```
From now on you can get back to original [TypiClust usage guide](typiclust_original_instructions/USAGE_Typiclust.md) starting from Section Representation Learning.  