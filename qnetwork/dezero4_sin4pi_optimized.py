import numpy as np
import matplotlib.pyplot as plt
from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F

class ThreeLayerNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(100)
        self.l2 = L.Linear(100)
        self.l3 = L.Linear(1)

    def forward(self, x):
        y = F.tanh(self.l1(x))
        y = F.tanh(self.l2(y))
        y = self.l3(y)
        return y

x = np.linspace(0, 1, 200).reshape(-1, 1)
y = np.sin(4 * np.pi * x)

model = ThreeLayerNet()
optimizer = optimizers.Adam(lr=0.001)
optimizer.setup(model)

iters = 50000
loss_history = []

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()
    optimizer.update()

    if i % 1000 == 0:
        print(f'iter {i}, loss: {loss.data}')
    loss_history.append(float(loss.data))

plt.figure(figsize=(15, 5))

plt.subplot(121)
plt.plot(loss_history)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Loss History')

plt.subplot(122)
t = np.arange(0, 1, .01)[:, np.newaxis]
y_true = np.sin(4 * np.pi * t)
plt.plot(t, y_true, 'k--', label='True')
y_pred = model(t)
plt.plot(t, y_pred.data, 'r-', label='Prediction')
plt.scatter(x[::10], y[::10], c='b', s=20, label='Training data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin(4πx) Approximation')
plt.legend()

plt.tight_layout()
plt.show()

print(f'Final loss: {loss_history[-1]}') 