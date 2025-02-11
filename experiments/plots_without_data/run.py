import os
import matplotlib.pyplot as plt
import numpy as np

def plot_energy(output_dir: str):
    def potential(x):
        x = x - 0.2
        return np.where(x<=1.5, (x-1)**2, -np.exp(-(x-1.5)) + 1 + 0.5**2)
    
    plt.figure(figsize=(6, 4))
    plt.xlabel("Molecule State")
    plt.ylabel("Energy u(x)")
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False) 
    plt.gca().spines["bottom"].set_visible(False)

    x = np.arange(0.1, 5, 0.1)
    y = np.vectorize(potential)(x)
    y2 = np.exp(-y)
    
    plt.plot(x, y)
    plt.arrow(0, 0, 5, 0, head_width=0.05, head_length=0.1, fc='black', ec='black')
    plt.arrow(0, 0, 0, 1.5, head_width=0.1, head_length=0.05, fc='black', ec='black')

    plt.plot(x, np.zeros_like(x) + 1.25, linestyle='dashed')

    plt.savefig(os.path.join(output_dir, "energy_potential.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, "energy_potential.svg"), bbox_inches='tight')
    plt.show()

    # =========================
    
    plt.xlabel("Molecule State")
    plt.ylabel("p*(x)")
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False) 
    plt.gca().spines["bottom"].set_visible(False)
    plt.arrow(0, 0.28, 5, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    plt.arrow(0, 0.28, 0, 0.9, head_width=0.1, head_length=0.05, fc='black', ec='black')

    plt.plot(x, y2)

    plt.savefig(os.path.join(output_dir, "probability_from_energy.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, "probability_from_energy.svg"), bbox_inches='tight')

    plt.show()


def plot_gmm(output_dir: str):
    def t1(x):
        return 0.7 * np.exp(-0.3*(x-6)**2)
    def t2(x):
        return np.exp(-(x-2.5)**2)

    plt.figure(figsize=(6, 4))
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False) 
    plt.gca().spines["bottom"].set_visible(False)

    x = np.arange(0, 10, 0.05)
    y1 = np.vectorize(t1)(x)
    y2 = np.vectorize(t2)(x)
    y3 = y1 + y2 + 0.01

    plt.plot(x, y3, color="orange")
    plt.fill_between(x, y3, color="orange", alpha=0.2)
    plt.text(3.8, 0.5, "p*(x)", color="orange")

    plt.plot(x, y1, color="green")
    plt.fill_between(x, y1, color="green", alpha=0.2)
    plt.text(5.3, 0.75, "q(x|o2)", color="green")

    plt.plot(x, y2, color="blue")
    plt.fill_between(x, y2, color="blue", alpha=0.2)
    plt.text(1.1, 1, "q(x|o1)", color="blue")
    
    plt.plot(x, np.zeros_like(x), color="black", linewidth=0.5)

    plt.savefig(os.path.join(output_dir, "gmm_example.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, "gmm_example.svg"), bbox_inches='tight')

    plt.show()


def plot_symmetries_illustration(output_dir: str):
    def target(x):
        return 0.7 * np.exp(-0.3*(x-6)**2) + np.exp(-(x-2.5)**2)

    plt.figure(figsize=(6, 4))
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False) 
    plt.gca().spines["bottom"].set_visible(False)

    plt.xlabel("Molecule State")
    plt.ylabel("p*(x)")

    plt.arrow(0, 0, 10.5, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    plt.arrow(0, 0, 0, 6, head_width=0.1, head_length=0.05, fc='black', ec='black')

    x = np.arange(0, 10, 0.05)
    y = np.vectorize(target)(x)

    offsets = [0.5, 2.5, 4.5]

    for o in offsets:
        plt.plot(x, y + o, color="orange")
        plt.fill_between(x, y + o, np.zeros_like(x) + o, color="orange", alpha=0.2)
        plt.plot(x, np.zeros_like(x) + o, linestyle="dashed", color="black")

    plt.savefig(os.path.join(output_dir, "symmetries_illustration.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, "symmetries_illustration.svg"), bbox_inches='tight')

    plt.show()

def run(args):
    output_dir = os.path.join(args.output_dir, "experiments")
    plot_symmetries_illustration(output_dir)