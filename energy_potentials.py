import jax.numpy as jnp
import numpy as np



def w1(z):
    return jnp.sin(2 * np.pi * z[:, 0] / 4)


def w2(z):
    return 3 * jnp.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def w3(z):
    return 3 * sigmoid((z[:, 0] - 1) / 0.3)


def U1(z, exp=False):
    z_norm = jnp.linalg.norm(z, 2, 1)
    add1 = 0.5 * ((z_norm - 2) / 0.4) ** 2
    add2 = -jnp.log(
        jnp.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2)
        + jnp.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2)
        + 1e-9
    )
    f = -(add1 + add2)
    if exp:
        return jnp.exp(f)
    else:
        return f


def U2(z, exp=False):
    f = -0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2
    if exp:
        return jnp.exp(f)
    else:
        return f


def U3(z, exp=False):
    in1 = jnp.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2)
    in2 = jnp.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2)
    f = jnp.log(in1 + in2 + 1e-9)
    if exp:
        return jnp.exp(f)
    else:
        return f


def U4(z, exp=False):
    in1 = jnp.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
    in2 = jnp.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2)
    f = jnp.log(in1 + in2 + 1e-9)
    if exp:
        return jnp.exp(f)
    else:
        return f


def U5(z, exp=False):
    z_norm = jnp.norm(z, 2, 1)
    add1 = 0.5 * ((z_norm - 2) / 0.4) ** 2
    add2 = -jnp.log(
        jnp.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2)
        + jnp.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2)
        + jnp.exp(-0.5 * ((z[:, 1] + 2) / 0.6) ** 2)
        + jnp.exp(-0.5 * ((z[:, 1] - 2) / 0.6) ** 2)
        + 1e-9
    )
    f = -(add1 + add2)
    if exp:
        return jnp.exp(f)
    else:
        return f


def U6(z, exp=False):
    z1 = z[:, 0] + 10
    z2 = z[:, 1] - 1.5
    z = jnp.hstack([z1.reshape(-1, 1), z2.reshape(-1, 1)])
    in1 = jnp.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
    in2 = jnp.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2)
    f = jnp.log(in1 + in2 + 1e-9)
    if exp:
        print(jnp.exp(f))
        return jnp.exp(f)
    else:
        return f

