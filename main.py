from nnfs.datasets import spiral_data

import activation
import layer
import loss

X, y = spiral_data(samples=100, classes=3)

layer1 = layer.Layer(2, 8)
layer2 = layer.Layer(8, 16)
layer3 = layer.Layer(16, 16)
layer4 = layer.Layer(16, 3)
ReLU = activation.Activation_ReLU()
Softmax = activation.Activation_Softmax()

layer1.forward(X)
ReLU.forward(layer1.output)
layer2.forward(ReLU.output)
ReLU.forward(layer2.output)
layer3.forward(ReLU.output)
ReLU.forward(layer3.output)
layer4.forward(ReLU.output)
Softmax.forward(layer4.output)
loss_function = loss.Loss_CategoricalCrossentropy()
loss = loss_function.calculate(Softmax.output, y)

print(Softmax.output[:5])
print("loss:", loss)
