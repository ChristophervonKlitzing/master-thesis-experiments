import torch
import numpy as np


def w1(z):
    return torch.sin(2 * np.pi * z[:, 0] / 4)


def w2(z):
    return 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)


def w3(z):
    return 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)


def U1(z, exp=False):
    z_norm = torch.norm(z, 2, 1)
    add1 = 0.5 * ((z_norm - 2) / 0.4) ** 2
    add2 = -torch.log(
        torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2)
        + torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2)
        + 1e-9
    )
    f = -(add1 + add2)
    if exp:
        return torch.exp(f)
    else:
        return f


def U2(z, exp=False):
    f = -0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2
    if exp:
        return torch.exp(f)
    else:
        return f


def U3(z, exp=False):
    in1 = torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2)
    in2 = torch.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2)
    f = torch.log(in1 + in2 + 1e-9)
    if exp:
        return torch.exp(f)
    else:
        return f


def U4(z, exp=False):
    in1 = torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
    in2 = torch.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2)
    f = torch.log(in1 + in2 + 1e-9)
    if exp:
        return torch.exp(f)
    else:
        return f


def U5(z, exp=False):
    z_norm = torch.norm(z, 2, 1)
    add1 = 0.5 * ((z_norm - 2) / 0.4) ** 2
    add2 = -torch.log(
        torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2)
        + torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2)
        + torch.exp(-0.5 * ((z[:, 1] + 2) / 0.6) ** 2)
        + torch.exp(-0.5 * ((z[:, 1] - 2) / 0.6) ** 2)
        + 1e-9
    )
    f = -(add1 + add2)
    if exp:
        return torch.exp(f)
    else:
        return f


def U6(z, exp=False):
    z1 = z[:, 0] + 10
    z2 = z[:, 1] - 1.5
    z = torch.hstack([z1.reshape(-1, 1), z2.reshape(-1, 1)])
    in1 = torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
    in2 = torch.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2)
    f = torch.log(in1 + in2 + 1e-9)
    if exp:
        print(torch.exp(f))
        return torch.exp(f)
    else:
        return f

