
import MLP
import Value

def main():
    x=[2.0,3.0,-1.0]
    n = MLP.MLP(3, [4, 4, 1])
    dot = Value.draw_dot(n(x))
    dot.render('computation_graph_MLP', format='svg', cleanup=True)

    xs=[[2.0,3.0,-1.0],
        [3.0,-1.0,0.5],
        [0.5,1.0,1.0],
        [1.0,1.0,-1.0]]
    ys=[1.0,-1.0,-1.0,1.0]
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    loss.backward()
    dot = Value.draw_dot(loss)
    dot.render('computation_graph_MLP_2', format='svg', cleanup=True)
    
    for k in range(50):
        ypred= [n(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        for p in n.parameters():
            p.data += -0.05 * p.grad

        print(f"step {k}, loss: {loss.data}")

    print(f"Final predictions: {ypred}")

if __name__ == "__main__":
    main()

