import math
import Value
import torch

def forward(x1, x2, w1, w2, b):
    n = x1 * w1 + x2 * w2 + b
    e = math.exp(2*n)
    o = (e-1)/(e+1)
    return o


def gradients(x1, x2, w1, w2, b): #uses numerical derviative to calculate gradient descent
    h = 0.0001
    o = forward(x1, x2, w1, w2, b)
    x1_grad = (forward(x1+h, x2, w1, w2, b) - o) / h
    x2_grad = (forward(x1, x2+h, w1, w2, b) - o) / h
    w1_grad = (forward(x1, x2, w1+h, w2, b) - o) / h
    w2_grad = (forward(x1, x2, w1, w2+h, b) - o) / h
    b_grad = (forward(x1, x2, w1, w2, b+h) - o) / h
    return x1_grad, x2_grad, w1_grad, w2_grad, b_grad

def main():
    #first part
    """"
    a = Value.Value(4.0, label='a')
    b = Value.Value(-2.0, label='b')
    c = Value.Value(3.0, label='c')
    f = Value.Value(-1.0, label='f')

    e = a * b
    e.label = 'e'
    d = e + c
    d.label = 'd'
    L = d * f
    L.label = 'L'

    L.grad = 1.0
    f.grad = d.data * L.grad
    d.grad = f.data * L.grad
    c.grad = 1.0 * d.grad
    e.grad = 1.0 * d.grad
    a.grad = b.data * e.grad
    b.grad = a.data * e.grad

    print(f"a: {a}")
    print(f"b: {b}")
    print(f"c: {c}")
    print(f"e: {e}")
    print(f"d: {d}")
    print(f"f: {f}")
    print(f"L: {L}")


    dot = Value.draw_dot(L)
    dot.render('computation_graph', format='svg', cleanup=True)
    """
    #numerical derviative implementation

    print("numerical derviative implementation")
    print(f"x1: {gradients(3.0, 0.0, -2.0, 4.0, 5.0)[0]}")
    print(f"x2: {gradients(3.0, 0.0, -2.0, 4.0, 5.0)[1]}")
    print(f"w1: {gradients(3.0, 0.0, -2.0, 4.0, 5.0)[2]}")
    print(f"w2: {gradients(3.0, 0.0, -2.0, 4.0, 5.0)[3]}")
    print(f"b: {gradients(3.0, 0.0, -2.0, 4.0, 5.0)[4]}")      
                           

    #one neuron case
    x1 = Value.Value(3.0, label='x1')
    x2 = Value.Value(0.0, label='x2')
    w1 = Value.Value(-2.0, label='w1')
    w2 = Value.Value(4.0, label='w2')
    b = Value.Value(5, label='b')

    x1w1 = x1 * w1; x1w1.label = 'x1w1'
    x2w2 = x2 * w2; x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
    n = x1w1x2w2 + b; n.label = 'n'
    #o = n.tanh(); o.label = 'o'
    e=(2*n).exp()
    o= (e-1)/(e+1); o.label = 'o'
    o.backward()
    print("micrograd implementation")
    print(f"x1: {x1.grad}")
    print(f"x2: {x2.grad}")
    print(f"w1: {w1.grad}")
    print(f"w2: {w2.grad}")
    print(f"b: {b.grad}")

    """ bakward pass
    o.grad= 1.0
    n.grad = (1 - o.data ** 2) 
    b.grad =n.grad
    x1w1x2w2.grad = n.grad
    x1w1.grad =x1w1x2w2.grad
    x2w2.grad = x1w1x2w2.grad
    x1.grad = w1.data * x1w1.grad
    w1.grad = x1.data * x1w1.grad
    x2.grad = w2.data * x2w2.grad
    w2.grad = x2.data * x2w2.grad

    o.backward()
    """
    """
    dot = Value.draw_dot(o)
    dot.render('computation_graph_open_tanh', format='svg', cleanup=True)

    """
    x1= torch.Tensor([3.0]).double() ; x1.requires_grad=True
    x2= torch.Tensor([0.0]).double() ; x2.requires_grad=True
    w1= torch.Tensor([-2.0]).double() ; w1.requires_grad=True
    w2= torch.Tensor([4.0]).double() ; w2.requires_grad=True
    b= torch.Tensor([5.0]).double() ; b.requires_grad=True

    n= x1*w1 + x2*w2 + b
    o= torch.tanh(n)
    print("pytorch implementation")
    print (o.data.item())
    o.backward()
    print('......')
    print(f"x1: {x1.grad.item()}")
    print(f"x2: {x2.grad.item()}")
    print(f"w1: {w1.grad.item()}")
    print(f"w2: {w2.grad.item()}")
    print(f"b: {b.grad.item()}")


if __name__ == "__main__":
    main()


    """ 
    numerical derviative implementation
    x1: -0.8398207515170064
    x2: 1.6804091927624665
    w1: 1.2602109178172505
    w2: 0.0
    b: 0.4200063276482702
    micrograd implementation
    x1: -0.839948683228052
    x2: 1.679897366456104
    w1: 1.259923024842078
    w2: 0.0
    b: 0.419974341614026
    pytorch implementation
    -0.7615941559557649
    ......
    x1: -0.8399486832280523
    x2: 1.6798973664561045
    w1: 1.2599230248420783
    w2: 0.0
    b: 0.41997434161402614
    """