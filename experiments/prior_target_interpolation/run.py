from typing import Callable, List, Literal, Optional, Tuple
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import math 


class Gaussian:
    def __init__(self, mean: float, var: float):
        self.mean = mean
        self.var = var # =sigma^2
        self._pdf = norm(self.mean, math.sqrt(self.var))
    
    def log_prob(self, x: np.ndarray[float]):
        #normalization = -0.5 * math.log(2 * np.pi * self.var)
        #return normalization -(x - self.mean)**2 / (2 * self.var)
        return self._pdf.logpdf(x)
    
    def sample(self, n: int):
        # return np.random.normal(self.mean, math.sqrt(self.var), (n,))
        return self._pdf.rvs((n,))
    
    def logZ(self):
        """
        log normalization constant of an unnormalized Gaussian with variance 'var'.
        """
        return 0.5 * math.log(2 * np.pi * self.var)
    
    def entropy(self):
        return 0.5 * math.log(2 * np.pi * self.var) + 0.5


def exp_normalized(g: Gaussian, a: float, eps: float = 1e-30):
    assert(a >= 0)
    return Gaussian(g.mean, (g.var / (a + eps)))

def mul_normalized(a: Gaussian, b: Gaussian):
    new_var = 1 / (1 / a.var + 1 / b.var)
    new_mean = new_var * (a.mean / a.var + b.mean / b.var)
    return Gaussian(new_mean, new_var)





def dual_impl(eta: float, eps_tr: float, lambda_: float, eps_ent: float, q_bar: Gaussian, target: Gaussian, entropy_contraint: str):
    """
    2D dual function with the kl trust-region constraint and relative entropy constraint
    """
    a = exp_normalized(q_bar, eta / (1 + eta + lambda_))
    b = exp_normalized(target, 1 / (1 + eta + lambda_))
    q_star = mul_normalized(a, b)

    q_bar_logZ = q_bar.logZ() * (eta / (1 + eta + lambda_))
    target_logZ = target.logZ() / (1 + eta + lambda_)
    logZ_closed_form = q_star.logZ() - (q_bar_logZ + target_logZ)

    samples = q_star.sample(10000)
    Z_estimate = np.mean(np.exp(q_bar.log_prob(samples) * (eta / (1 + eta + lambda_)) + target.log_prob(samples) / (1 + eta + lambda_) - q_star.log_prob(samples)))
    # print(Z_estimate)
    logZ = np.log(Z_estimate)
    # print(logZ, logZ_closed_form)
    
    
    # print(q_star.mean, q_star.var, q_star.logZ(), logZ)
    if eps_ent is not None: 
        if entropy_contraint == "relative":
            entropy_extension = lambda_ * (q_bar.entropy() - eps_ent)
        elif entropy_contraint == "absolute":
            entropy_extension = lambda_ * eps_ent
        elif entropy_contraint == "exponential":
            entropy_extension = lambda_ * (eps_ent * q_bar.entropy() )
        else:
            raise ValueError(f"Entropy constraint '{entropy_contraint}' not supported")
    else:
        entropy_extension = 0.
    
    dual = -(1 + eta + lambda_) * logZ - eta * eps_tr + entropy_extension
    return dual


def plot_1D_dual(f: Callable[[np.ndarray[float]], np.ndarray[float]], eta_star: float):
    x = np.linspace(0, 200, 1000)
    # Compute heights
    Z = f(x)

    fig, ax = plt.subplots()
    ax.plot(x, Z)
    # c = ax.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='terrain')
    # plt.colorbar(c, ax=ax, label='Height')
    # ax.set_title("Negative 1D dual")
    ax.scatter(eta_star, f(np.array([eta_star])), marker="x", c="red")
    plt.show()



def plot_2D_dual(f: Callable[[np.ndarray[float]], np.ndarray[float]], point: Tuple[float]):
    x = np.linspace(0, 50, 50)
    y = np.linspace(0, 50, 50)
    X, Y = np.meshgrid(x, y)
    xy_pairs = np.stack([X.ravel(), Y.ravel()], axis=-1)  # shape: (10000, 2)
    
    # Compute heights
    Z = f(xy_pairs).reshape(X.shape)

    fig, ax = plt.subplots()
    c = ax.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='terrain')
    ax.scatter(point[0], point[1], c="red")
    plt.colorbar(c, ax=ax)
    # ax.set_title("2D Height Map")
    plt.show()
    
    

def find_best_multipliers(eps_tr: float, q_bar: Gaussian, target: Gaussian, entropy_constraint: str, eps_ent: Optional[float] = None, max_eta = 1e10, max_lambda = 1e10) -> Tuple[float, float]:
    def neg_dual(multipliers: np.ndarray[float]):
        eta = multipliers[0]
        if eps_ent is not None:
            lambda_ = multipliers[1]
        else:
            lambda_ = 0.
        return -dual_impl(eta, eps_tr, lambda_, eps_ent, q_bar, target, entropy_contraint=entropy_constraint)

    def vectorized_dual(batched_multipliers: np.ndarray[float]):
        if len(batched_multipliers.shape) == 1:
            batched_multipliers = np.expand_dims(batched_multipliers, -1)
        batch = batched_multipliers.shape[0]
        output = np.zeros((batch,))
        for i in range(batch):
            output[i] = neg_dual(batched_multipliers[i])
        return output
    
    bounds = [(0.0, max_eta), (0.0, max_lambda)]
    initial_guess = np.zeros(2) + 10
    result = minimize(fun=neg_dual, x0=initial_guess, bounds=bounds)
    eta_star: float = result.x[0]
    
    if eps_ent is None:
        # plot_1D_dual(vectorized_dual, eta_star)
        return eta_star, 0.0
    else:
        lambda_star: float = result.x[1]
        # plot_2D_dual(vectorized_dual, (eta_star, lambda_star))
        return eta_star, lambda_star

def run(args=None):
    # Define x-axis
    x = np.linspace(-6, 6, 200)

    # Define two-mode target distribution p(x)
    prior = Gaussian(-1, 3)
    target = Gaussian(3, 0.1)

    print(f"prior entropy: {prior.entropy():.2f}, target entropy: {target.entropy():.2f}")

    entropy_constraint: Literal["relative", "absolute", "exponential"] = "relative" 
    eps_tr = 0.2
    eps_ent = None # 0.1

    fontsize = 25
    file_type = None # "pgf"
    pretty = True

    # Set up horizontal subplots
    fig, ax = plt.subplots()

    target_y = np.exp(target.log_prob(x))
    ax.plot(x, target_y, label="target", c="green")
    ax.fill_between(x, target_y, color="green", alpha=0.2)
    
    prior_y = np.exp(prior.log_prob(x))
    ax.plot(x, prior_y, label="prior", c="blue")
    ax.fill_between(x, prior_y, color='cornflowerblue', alpha=0.4)

    if pretty:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

        ax.text(3.7, 0.9, "$p^*$", fontsize=fontsize, color="green")
        ax.text(-2.5, 0.29, "$q$", fontsize=fontsize, color="blue")
    else:
        plt.title(f"eps_tr: {eps_tr} eps_ent: {eps_ent}")
    
    q_bar = prior
    while True:
        eta_star, lambda_star = find_best_multipliers(eps_tr, q_bar, target, eps_ent=eps_ent, entropy_constraint=entropy_constraint)
        print(f"eta*: {eta_star:.2f}, lambda*: {lambda_star:.2f}")
        a = exp_normalized(q_bar, eta_star / (1 + eta_star + lambda_star))
        b = exp_normalized(target, 1 / (1 + eta_star + lambda_star))
        q_star = mul_normalized(a, b)
        kl_diff = 0.5 * ((q_star.mean - q_bar.mean)**2 / q_bar.var + q_star.var / q_bar.var - (math.log(q_star.var) - math.log(q_bar.var)) - 1)
        # kl_diff = 0.5 * (math.log(q_bar.var / q_star.var) + (q_star.var + (q_star.mean - q_bar.mean)**2) / q_bar.var - 1)
        print(f"q* entropy: {q_star.entropy():.2f} kl-diff: {kl_diff:.2f}")

        q_star_y = np.exp(q_star.log_prob(x))
        ax.plot(x, q_star_y, linestyle="--", color="gray")
        q_bar = q_star

        if eta_star < 1e-3 and lambda_star < 1e-3:
            break

        print()
    
    # Layout and display
    fig.tight_layout()

    if file_type is not None:
        name = f"eps_tr_{eps_tr}_eps_ent_{eps_ent}".replace(".", "_")
        fig.savefig(f"outputs/experiments/prior_target_interpolation/{name}.{file_type}", format=file_type, bbox_inches='tight')
    else:
        plt.show()

    plt.close()




if __name__ == "__main__":
    run()