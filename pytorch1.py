import torch 

x=torch.empty((1))

y = torch.rand(2,3)
e = torch.rand(2,3)

z = torch.zeros(2,2,2)

t = torch.tensor([2.3,1.9])
# print(y)
# print(z)
print(t)

#addition
add1 = y+e
#subraction
sub1 = y-e
#multiplication
mul1 = y*e
#division
div1 = y/e

#is cuda available

if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x+y
    print(z)
    print(z.to("cpu", torch.double))

#/Users/jayahariaditiya/pytorch_bruh