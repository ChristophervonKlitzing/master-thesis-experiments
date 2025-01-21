import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Define the functions
def p(x):
    return 1/(x**4 + 1) + jnp.exp(-5*jnp.square(x-7))

def q(x):
    # return jnp.clip(1/(x**2 + 1), min=0.4)
    return 1/(x**2 + 1)

def residual(x, a):
    q_x = q(x)
    p_x = p(x)

    # q_x = jnp.where(q_x<=p_x, q_x, p_x)

    log_q_x_a = jnp.log(q_x + a)
    log_p_x_a = jnp.log(p_x + a)
    log_q_x = jnp.log(q_x)
    log_p_x = jnp.log(p_x)
    # return jnp.minimum(jnp.exp(log_p_x - log_q_x), jnp.exp(log_p_x_a - log_q_x_a))
    return jnp.exp(log_p_x_a - log_q_x_a)


def plot(output_dir: str):
    # Set the range of x-values (avoiding division by zero)
    x = jnp.linspace(-5, 15, 500)

    # Define the parameter 'a'
    a = 0.01

    # Compute the functions
    y1 = p(x)
    y2 = q(x)
    y3 = residual(x, a)

    # Plot the curves
    fig = plt.figure(figsize=(10, 6))

    log = True
    if log:
        y1 = jnp.log(y1)
        y2 = jnp.log(y2)
        y3 = jnp.log(y3)
    else:
        plt.ylim(-0.1, 1.5)
    
    def make_label(f_name: str):
        if log:
            return f"log{{{f_name}}}"
        else:
            return f_name
    plt.plot(x, y1, label=f"{make_label('f(x)')}", color="blue")
    plt.plot(x, y2, label=f"{make_label('q(x)')}", color="orange")
    plt.plot(x, y3, label=f"{make_label('(p(x) + a) / (q(x) + a)')}, a={a}", color="green")

    # Add labels, legend, and grid
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

    fig.savefig(os.path.join(output_dir, "experiments/residual_gradient.pdf"), bbox_inches='tight')
