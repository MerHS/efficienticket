# EfficienTicket
The Lottery Ticket Hypothesis - PyTorch demo for ResNet/EfficientNet

## Prerequisite
 - pytorch >= 1.3.0
 - tqdm
 - numpy

## Training scheme
 * 원 논문 Section.4의 VGG/ResNet training 방법론을 사용
    * CIFAR10 training with global iterative pruning
 * 논문의 표현에 따르면 ResNet등의 큰 네트워크는 LR과 entangled 되어있어 적절한 LR 선택이 중요
 * ResNet으로 재현을 해본 다음, EfficientNet의 training은 ResNet과 같은 hyperparameter와 EfficientNet의 hyperparameter를 둘 다 사용해봄.
 * Early-stopping iteration은 논문과 같이 따로 측정하지 않음 
 * 기존 ticketing 방법과 Stabilizing the Lottery Ticket Hypothesis 후속논문을 비교 (500 iter)
 * CIFAR-10 사용, 75 epoch training
