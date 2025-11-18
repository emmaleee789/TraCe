# [NeurIPS 2025 🎉] Official repo for "Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation"

## [Paper📖](https://openreview.net/pdf/cbd774fb221ea45dd918c48bbc278a4612176548.pdf) | [Video🎬](https://drive.google.com/drive/folders/1O6KeJ9fI31ZPH6EkO9kTF67BkV4D244S?usp=sharing)

## 🚀 Get Started
### **Installation**
Install [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Shap-E](https://github.com/openai/shap-e#usage) as fellow:

```
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
pip install ninja
pip install -r requirements.txt

pip install ./gaussiansplatting/submodules/diff-gaussian-rasterization
pip install ./gaussiansplatting/submodules/simple-knn

git clone https://github.com/openai/shap-e.git
cd shap-e
pip install -e .
```
Download [finetuned Shap-E](https://huggingface.co/datasets/tiange/Cap3D/resolve/9bfbfe7910ece635e8e3077bed6adaf45186ab48/our_finetuned_models/shapE_finetuned_with_330kdata.pth) by Cap3D, and put it in `./load`

## Citation

``` bibtex
@inproceedings{liwalking,
  title={Walking the Schr{\"o}dinger Bridge: A Direct Trajectory for Text-to-3D Generation},
  author={Li, Ziying and Lu, Xuequan and Zhao, Xinkui and Cheng, Guanjie and Deng, Shuiguang and Yin, Jianwei},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```
