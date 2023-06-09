#pytorch consists of the following pipeline

#1) Design Model
#2) Construct loss and optimizer
#3) Training Loop
# - forward pass : compute prediction and loss
# - backward pass: gradients
# - update weights


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets




X_numpy , y_numpy = datasets.make_regression(n_samples=100, n_features=1,noise=20,random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0],1)

n_samples , n_features = X.shape


#model

input_size = n_features
output_size = 1

model = nn.Linear(input_size,output_size)

learning_rate =0.01

#loss and optimizer

critereon = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

#training loop
num_epoch = 10

for epoch in range(num_epoch):

    #forward pass
    y_predicted = model(X)

    #find the loss
    loss = critereon(y_predicted,y)

    loss.backward() #backprop

    optimizer.step() #update the weights

    #before next iterate zero the gradient
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

#plot
predicted = model(X).detach().numpy()

plt.plot(X_numpy,y_numpy,'ro')
plt.plot(X_numpy,predicted,'b')
plt.show()