# Deep Learning Project 1
In this project we trained a constrained model ( <= 5Million ) parameters on a variant of CIFAR-10 dataset and achieved ~86% test set accuracy. We used ImprovedResNet architecture as baseline for our implementation and use extensive data augmentation techniques to increase generalization. Also we used post-processing techniques like SWA ( Stochastic Weight averaging ) and TTA ( Test time Augmentation ) to improve test results.

## ImprovedResNet

ImprovedResNet offers a careful balance between model capacity, computational efficiency, and regularization techniques. This implementation is specifically optimized for classification tasks with a parameter budget under 5M, making it suitable for resource-constrained environments while maintaining competitive accuracy.

## Key Features

- **Squeeze-and-Excitation Attention**: Channel-wise feature recalibration for dynamic emphasis of important features
- **Strategic Dropout Placement**: Gradually increasing dropout rates between network stages to reduce overfitting
- **Parameter-Efficient Design**: Carefully balanced channel growth to maintain high performance under 5M parameters
- **Optimized Initial Layers**: 3×3 convolution with stride 1 preserves spatial information for smaller input images
- **Kaiming Weight Initialization**: Ensures stable gradient flow during training

## Architecture Details

ImprovedResNet builds upon the traditional ResNet architecture with several enhancements:

- **Bottleneck Blocks**: Each residual block follows the bottleneck design (1×1 → 3×3 → 1×1 convolutions) with integrated Squeeze-and-Excitation modules
- **Progressive Channel Growth**: Feature channels expand from 32 → 64 → 128 → 238 across network stages
- **Regularization Strategy**: Dropout layers between stages with rates progressing from 0.1 to 0.3

## Usage

```python
import torch
from model import ImprovedResNet, BottleneckBlock

# Create model with default parameters
model = ImprovedResNet(
    block=BottleneckBlock,
    layers=[2, 2, 2, 2],  # ResNet-18 configuration
    num_classes=10,       # CIFAR-10 or similar
    dropout_rate=0.2      # Base dropout rate (automatically scaled at different stages)
)

# Forward pass
input_tensor = torch.randn(1, 3, 32, 32)  # Example for CIFAR-10
output = model(input_tensor)
```

## Recommended Training Approach

For optimal results, consider implementing:

- **Learning Rate Scheduling**: Cosine annealing or step decay with periodic warm restarts
- **Data Augmentation**: Random crops, horizontal flips, and color jittering
- **Stochastic Weight Averaging (SWA)**: SWA is an advanced optimization technique that enhances neural network generalization by computing the average of weights traversed by stochastic gradient descent with a cyclical or constant learning rate. This method effectively explores a broader region of the loss landscape, finding flatter minima that demonstrate superior robustness to test data variations.

## Requirements

- PyTorch 1.7.0 or higher
- torchvision (for datasets and transforms)
- numpy


## License

MIT
