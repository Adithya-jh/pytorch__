#gradient calculation using autograd
import torch

x= torch.randn(3,requires_grad=True) #requires_grad = True is for telling pytorch to compute gradient of some lossFunction w.r.t the var

print(x)

y= x+2
print(y)

z= y*y*2
# print(z)

z = z.mean()
print(z)

z.backward() #backpropagation
print(x.grad)



#output
# tensor([-0.2013, -0.2013, -0.2013], requires_grad=True)
# tensor([1.7987, 1.7987, 1.7987], grad_fn=<AddBackward0>)
# t
# tensor([1.7987, 1.7987, 1.7987], grad_fn=<AddBackward0>)
# tensor([11.9870, 11.9870, 11.9870], grad_fn=<MulBackward0>)
# tensor(11.9870, grad_fn=<MeanBackward0>)
# tensor([3.4977, 3.4977, 3.4977])
