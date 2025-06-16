import numpy as np
import matplotlib.pyplot as plt


def cosine_schedule(t: float, max: float, min: float = 0.0, scale=1.0):
    t_scaled = t * (np.pi / scale)
    t_scaled = np.where(t_scaled >= np.pi, np.pi, t_scaled)
    y = np.cos(t_scaled)
    y = (y + 1) / 2 # normalize to [0, 1]
    y = y * (max - min) + min # shift/scale to [min, max]
    return y 


def interpolate_iteratively(start: float, target: float, etas: np.ndarray[float], lambdas: np.ndarray[float]):
    interpolation = np.zeros_like(etas)
    interpolation[0] = start
    for i in range(1, interpolation.shape[0]):
        a = interpolation[i - 1]**(etas[i - 1] / (1 + etas[i - 1] + lambdas[i - 1]))
        b = target**(1 / (1 + etas[i - 1] + lambdas[i - 1]))
        interpolation[i] = a * b
    return interpolation 


def gamma(a, b, betas: np.ndarray[float]):
    value = 1.
    for i in range(a, b + 1):
        value *= betas[i] / (1 + betas[i])
    return value 


def target_prod(i: int, target: float, lambdas: np.ndarray[float], betas: np.ndarray[float]):
    prod = 1.
    for k in range(i + 1):
        prod *= (target**(1 / (1 + lambdas[k])))**((1 - (betas[k] / (1 + betas[k]))) * gamma(k + 1, i, betas))
    return prod 


def interpolate_closed_form(start: float, target: float, etas: np.ndarray[float], lambdas: np.ndarray[float]):
    betas = etas / (1 + lambdas)
    interpolation = np.zeros_like(etas)
    
    for i in range(interpolation.shape[0]):
        a = start**gamma(0, i - 1, betas)
        b = target_prod(i - 1, target, lambdas, betas)
        interpolation[i] = a * b
    return interpolation


def interpolate_closed_form_2(start: float, target: float, etas: np.ndarray[float], lambdas: np.ndarray[float]):
    def beta(a: int, b: int):
        prod = 1.
        for j in range(a, b + 1):
            prod *= (etas[j] / (1 + etas[j] + lambdas[j]))
        return prod
    
    def alpha(i: int):
        sum = 0.
        for k in range(i + 1):
            sum += ((lambdas[k] / (1 + etas[k] + lambdas[k]))) * beta(k + 1, i)
        
        alpha = 1 - sum / (1 - beta(0, i))
        return alpha 

    interpolation = np.zeros_like(etas)
    interpolation[0] = start
    for i in range(1, interpolation.shape[0]):
        beta_ = beta(0, i - 1)
        a = start**beta_
        b = (target**alpha(i - 1))**(1 - beta_)
        interpolation[i] = a * b
    return interpolation


def interpolate_wrong(start: float, target: float, etas: np.ndarray[float], lambdas: np.ndarray[float]):
    betas = etas / (1 + lambdas)
    interpolation = np.zeros_like(etas)

    for i in range(0, interpolation.shape[0]):
        a = start**gamma(0, i - 1, betas)
        b = target**(1 - ((etas[0] + lambdas[0]) / etas[0]) * gamma(0, i - 1, betas))
        interpolation[i] = a * b 
    return interpolation 


def run(args):
    max_i = 20
    ts = np.arange(max_i) / max_i

    etas = cosine_schedule(ts, 3.0, scale=0.9)
    lambdas = cosine_schedule(ts, 2.0, scale=0.5)

    start = 0.1
    target = 0.7

    qs_iterative = interpolate_iteratively(start, target, etas, lambdas)
    qs_closed_form = interpolate_closed_form(start, target, etas, lambdas)
    qs_closed_form2 = interpolate_closed_form_2(start, target, etas, lambdas)
    qs_wrong = interpolate_wrong(start, target, etas, lambdas)

    plt.scatter(ts, qs_iterative, label="iterative", marker='o')
    # plt.scatter(ts, qs_closed_form, label="closed-form", marker='x')
    plt.scatter(ts, qs_closed_form2, label="closed-form-2", marker='x')
    # plt.scatter(ts, qs_wrong, label="wrong", marker='.')

    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()