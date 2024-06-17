## Emotional Recognition

This project aims to develop an emotional recognition system using the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). The RAVDESS dataset contains 24 professional actors (12 male, 12 female)



## Model

### U-vector attention: [Attention Based Fully Convolutional Network forSpeech Emotion Recognition](https://arxiv.org/abs/1806.01506)
- Used *0.3* lambda value
- Used Xavier Uniform Initialization

### CNN14 : [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/pdf/1912.10211.pdf)
- Used Average Pooling instead of Max Pooling
- Trained from various sources

### CNNX : [Shallow over Deep Neural Networks: A Empirical Analysis for Human Emotion Classification Using Audio Data](https://link.springer.com/chapter/10.1007/978-3-030-76736-5_13)
- Used a shallow net instead of deep net
- Used AVerage Pooling instead of Max Pooling

### 1DCNNLSTM : [A Hybrid CNN–LSTM Network for the Classification of Human Activities Based on Micro-Doppler Radar](https://ieeexplore.ieee.org/document/8978926)
|          Layer           | Width |
| :----------------------: | :---: |
| 1D Conv (ReLU + Maxpool) |  64   |
| 1D Conv (ReLU + Maxpool) |  128  |
| 1D Conv (ReLU + Maxpool) |  256  |
|           LSTM           |  256  |
|            256 FC            |   |

### VGG19
### Alexnet


## Results

*Training from Scratch*
|               | Accuracy | Precision | Recall | F1-Score |
| ------------- | -------- | --------- | ------ | -------- |
| CNN-14        | 60.00%   | 66.28%    | 60.48% | 59.30%   |
| CNN-14 + Attn | 60.68%   | 66.05%    | 61.14% | 60.20%   |
| CNN-X         | 54.93%   | 57.43%    | 54.43% | 54.24%   |
| 1DCNNLSTM     | 56.81%   | 56.98%    | 55.73% | 54.91%   |



*Transfer Learning*
|                | Accuracy | Precision | Recall | F1-Score |
| -------------- | -------- | --------- | ------ | -------- |
| VGG19 + Attn   | 65.28%   | 66.42%    | 65.10% | 64.43%   |
| ResNet         | 57.01%   | 57.02%    | 56.32% | 55.89%   |
| AlexNet + Attn | 55.62%   | 55.24%    | 54.69% | 53.81%   |
