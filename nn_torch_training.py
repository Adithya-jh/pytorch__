import torch

import torch.nn as nn

# Compute every step manually

# Linear regression
# f = w * x 

# here : f = 2 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

# w = torch.tensor(0.0, dtype=torch.float32,requires_grad=True)

# model output - MANUAL FORWARD
# def forward(x):
#     return w * x

#change - use pytorch nn module

n_samples , n_features = X.shape
input_size = n_features
output_size = n_features

model = nn.Linear(input_size,output_size)



# loss = MSE
# def loss(y, y_pred):
#     return ((y_pred - y)**2).mean()   

# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)
# def gradient(x, y, y_pred):
#     return np.mean(2*x*(y_pred - y))

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

#chnge - add loss and optimizer 

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# Training
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    # predict = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)
    
    # calculate gradients = backward pass
    # dw = gradient(X, Y, y_pred)
    l.backward() #dl/dw

    # update weights
    # with torch.no_grad():
    #     w -= learning_rate * w.grad

    #change - here we use pytorch module instead of updating weights manually
    optimizer.step()

    # zero_gradients
    # w.grad.zero_()

    optimizer.zero_grad()



    if epoch % 2 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
     
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')