from nnfs.datasets import spiral_data

import activation
import layer
import loss

X, y = spiral_data(samples=100, classes=3)

layer1 = layer.Layer(2, 8)
layer2 = layer.Layer(8, 16)
layer3 = layer.Layer(16, 16)
layer4 = layer.Layer(16, 3)
activation1 = activation.Activation_ReLU()
activation2 = activation.Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation1.forward(layer2.output)
layer3.forward(activation1.output)
activation1.forward(layer3.output)
layer4.forward(activation1.output)
activation2.forward(layer4.output)
loss_function = loss.Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print(activation2.output)
print("loss", loss)
