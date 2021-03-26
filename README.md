# VGGNet 구현해보기

- [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556.pdf)

## 개요

이 논문은 대규모 이미지 인식 설정에서 정확도에 대한 convolutional network의 깊이의 영향에 대해 조사한 논문이다. 매우 작은 (3x3) convolution filter를 적용한 구조에서 깊이가 증가함에 따라 네트워크의 변화를 평가한다.

## 모델의 구조

Input : 224 x 224 RGB image

이미지 전처리 : train set의 RGB 평균값을 각 픽셀로 부터 계산하여 subtracting

filter : 3 x 3

stride = 1

padding = same

max-pooling : 2 x 2, stride = 2

FC : 4096 → 1000(class, softmax)

activation : ReLU

total 144M(VGG19) parameter
