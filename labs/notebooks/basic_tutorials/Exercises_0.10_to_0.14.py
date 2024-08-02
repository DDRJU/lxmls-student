from lxmls.readers import galton
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
galton_data = galton.load()


def exercise_0_10():
    print("Exercise 0.10")
    print(galton_data.mean(0))
    print(galton_data.std(0))
    plt.hist(galton_data)
    plt.savefig("galton_histogram.png")
    plt.close()

    plt.plot(galton_data[:,0], galton_data[:,1], '.')
    plt.savefig("galton_scatter.png")
    plt.close()

    galton_data_randn = galton_data + 0.5*np.random.randn(len(galton_data), 2)
    plt.plot(galton_data_randn[:,0], galton_data_randn[:,1], '.')
    plt.savefig("galton_scatter_randn.png")
    plt.close()

def exercise_0_11():
    print("Exercise 0.11")

    a = np.arange(-5,5,0.01)
    f_x = np.power(a,2)
    plt.plot(a,f_x)
    plt.xlim(-5,5)
    plt.ylim(-5,15)
    k = np.array([-2,0,2])
    plt.plot(k,k**2,"bo")
    plt.savefig("diogoisnotcool.png")
    for i in k:
        plt.plot(a, (2*i)*a - (i**2))
        plt.savefig("diogoisnotcool.png")


def exercise_0_12():
    print("Exercise 0.12")
    def get_y(x):
        return (x+2)**2 - 16*np.exp(-((x-2)**2))

    x = np.arange(-8, 8, 0.001)
    y = get_y(x)
    plt.plot(x, y)
    plt.savefig("a_picture.png")

    def get_grad(x):
        return (2*x+4)-16*(-2*x + 4)*np.exp(-((x-2)**2))

    def grad_descent(start_x: float, func: callable, grad: callable, learning_rate: float):
        THRESHOLD = 1e-4
        point_x = start_x
        max_iterations = 1000
        current_iter = 0
        # upgrade formula is x becomes x - learning_rate * grad(x)
        pbar = tqdm(total=max_iterations)
    
        current_grad = grad(point_x)
        while current_grad > THRESHOLD and current_iter < max_iterations:
            current_grad = grad(point_x)
            grad_change = learning_rate * current_grad
            point_x = point_x - grad_change
            current_iter += 1
            pbar.update(1)

        pbar.close()
        
        return point_x
    
    start_x = 0
    learning_rate = 0.01
    final_x = grad_descent(start_x, get_y, get_grad, learning_rate)
    print(f"Final x is {final_x}")
        
def exercise_0_13():
    print("Exercise 0.13")

    use_bias = True

    w = np.random.randn(galton_data.shape[1]) # (2,)
    x = galton_data[:, 0] # (900, )
    y = galton_data[:, 1] # (900, )

    if use_bias:
        x = np.vstack([x, np.ones(x.shape[0])]) # (2, 900)
    else:
        x = np.vstack([x, np.zeros(x.shape[0])]) # (2, 900)


    def get_error(w, x, y):
        # (2, ) @ (2, 900) - (900, ) -> (900,)
        return (w @ x - y)**2

    def get_error_grad(w, x, y):
        # (2, 900) @ (900, ) -> (2, )
        return 2 * x @ (w @ x - y) / x.shape[1]

    def grad_descent(w, x, y, learning_rate):
        current_w = w
        learning_rate = 0.00001
        max_iterations = 1000
        THRESHOLD = 1e-4

        pbar = tqdm(total=max_iterations)

        res = []
        rmses = []
        for _ in range(max_iterations):
            current_error = np.mean(get_error(current_w, x, y))
            rmses.append(current_error)


            current_grad = get_error_grad(current_w, x, y)
            old_w = current_w
            current_w = current_w - learning_rate * current_grad
            res.append(current_w)
            pbar.update(1)
            if (abs(old_w- current_w)).sum() < THRESHOLD:
                break

        pbar.close()
        return current_w, rmses, res
    
    final_w, rmses, res = grad_descent(w, x, y, 0.001)
    print(f"Final w is {final_w}")
    print(f"RMSEs: {rmses}")
    #print(f"Res: {res}")


if __name__ == "__main__":
    # exercise_0_10()
    # exercise_0_11()
    # exercise_0_12()
    exercise_0_13()
