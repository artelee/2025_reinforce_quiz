import numpy as np
import matplotlib.pyplot as plt
from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F

def train_and_evaluate(x_data, hidden_size, activation, lr, optimizer_type):
    y_data = np.sin(4 * np.pi * x_data)
    
    class Net(Model):
        def __init__(self, hidden_size, out_size):
            super().__init__()
            self.l1 = L.Linear(hidden_size)
            self.l2 = L.Linear(out_size)
            self.activation = activation

        def forward(self, x):
            y = self.activation(self.l1(x))
            y = self.l2(y)
            return y

    model = Net(hidden_size, 1)
    
    if optimizer_type == 'Adam':
        optimizer = optimizers.Adam(lr)
    else:
        optimizer = optimizers.SGD(lr)
    
    optimizer.setup(model)

    loss_history = []
    iters = 20000
    
    for i in range(iters):
        y_pred = model(x_data)
        loss = F.mean_squared_error(y_data, y_pred)

        model.cleargrads()
        loss.backward()
        optimizer.update()
        
        if i % 1000 == 0:
            print(f'iter {i}, loss: {loss.data}')
        loss_history.append(float(loss.data))
    
    return loss_history, model

# 실험 설정
experiments = [
    {
        'name': 'Random + Adam + Sigmoid',
        'x_data': np.random.rand(100, 1),
        'hidden_size': 20,
        'activation': F.sigmoid,
        'lr': 0.01,
        'optimizer': 'Adam'
    },
    {
        'name': 'Linspace + Adam + Sigmoid',
        'x_data': np.linspace(0, 1, 100).reshape(-1, 1),
        'hidden_size': 20,
        'activation': F.sigmoid,
        'lr': 0.01,
        'optimizer': 'Adam'
    },
    {
        'name': 'Random + Adam + ReLU',
        'x_data': np.random.rand(100, 1),
        'hidden_size': 20,
        'activation': F.relu,
        'lr': 0.01,
        'optimizer': 'Adam'
    },
    {
        'name': 'Random + SGD + Sigmoid',
        'x_data': np.random.rand(100, 1),
        'hidden_size': 20,
        'activation': F.sigmoid,
        'lr': 0.1,
        'optimizer': 'SGD'
    }
]

plt.figure(figsize=(15, 10))

# Loss 비교
plt.subplot(221)
for exp in experiments:
    loss_history, model = train_and_evaluate(
        exp['x_data'], 
        exp['hidden_size'],
        exp['activation'],
        exp['lr'],
        exp['optimizer']
    )
    plt.plot(loss_history, label=exp['name'])
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Loss History Comparison')
plt.legend()

# 예측 결과 비교
plt.subplot(222)
t = np.arange(0, 1, .01)[:, np.newaxis]
y_true = np.sin(4 * np.pi * t)
plt.plot(t, y_true, 'k--', label='True function')

for exp in experiments:
    _, model = train_and_evaluate(
        exp['x_data'],
        exp['hidden_size'],
        exp['activation'],
        exp['lr'],
        exp['optimizer']
    )
    y_pred = model(t)
    plt.plot(t, y_pred.data, label=exp['name'])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Prediction Comparison')
plt.legend()

plt.tight_layout()
plt.show() 