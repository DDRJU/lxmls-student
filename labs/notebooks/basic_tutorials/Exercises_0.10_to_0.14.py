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
    
    def error(x: np.array, y: np.array, w: np.array):
        return (x.T @ w - y)**2
        
    def error_grad(x: np.array, y: np.array, w: np.array):
        return 2 * x @ (x.T @ w - y)

if __name__ == "__main__":
    # exercise_0_10()
    # exercise_0_11()
    
    exercise_0_12()
