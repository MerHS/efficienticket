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
 * rand-reinit과 ticketing을 비교

### EfficientNet training
 * 가능한 한 논문의 방법론을 따름
 * Swish activation, stochastic depth with drop connect ratio 0.2 
 * dropout 0.2 to 0.5
   * Ticket 논문에도 dropout 넣을 시 좋아지는 현상 있다고 함
 * No fixed AutoAugment (구현 힘듬)
 * RMSProp w/ decay 0.9 momentum 0.9, bn momentum 0.99, weight decay 1e-5
 * lr_scheduler.OneCycleLR 사용 (from Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates)
   * 120 epoch, 
   * EfficientNet 논문에서는 init lr 0.256 decays by 0.97 every 2.4 epochs을 사용했으나 batch size가 명확하지 않고 ImageNet 대신 CIFAR10을 사용하므로 다르게 해야한다
   * Ticket 논문에서는 LR 선정에 학습이 크게 좌우되는 경향을 보였으므로 좀 더 나은 LR strategy 주 하나인 SGDR을 사용.
    
    