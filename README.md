# Digit Recognition in C#
A simple Neural network for digit recognition, implemented from scratch in C#.

Uses the [MNIST Database of Handwritten Digits](https://github.com/cvdfoundation/mnist) as training/testing data.  
The network learns through gradient descent backpropagation and uses Sigmoid for its activation function.  

Currently achieves pretty stable accuracy on test data, although definitely improvable:  
| Training Size | Epochs | Test Accuracy |
| -----------: | -----------: | -----------: |
| 2000 | 20 | 89.0% |
| 2000 | 10 | 89.5% |
| 1000 | 20 | 88.0% |
| 1000 | 10 | 86.5% |

---
TODO:
 - Add Data Augmentation
 - Implement "better" activation functions
 - Add window for drawing your own digits to test the network
