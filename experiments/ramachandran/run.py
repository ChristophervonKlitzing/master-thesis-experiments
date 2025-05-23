from typing import Any, Dict
import yaml
from .generate_figures import generate_ram_figure
from auto_paper.figure_generation import FigureGenerator
from auto_paper.mpl.mpl_figure import MPLFigure
import matplotlib.pyplot as plt
import matplotlib
import os 

"""
This script is based on https://github.com/henrik-schopmans/annealed_bg_paper/blob/master/figures_and_tables.ipynb
"""


def run(args):
    """
    Before running this script, run:

    sudo sshfs -o allow_other,default_permissions <user>@haicore.scc.kit.edu:AnnealedBG /mnt/haicore

    Cleanup:
    sudo umount /mnt/haicore

    to mount the wandb
    """
    config_name: str = args.extra_1
    matplotlib.use('Agg')

    plt.style.use(["science", "nature", "bright"])
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    wandb_main_path = "/mnt/haicore/annealed_bg/wandb"
    
    file_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(file_dir, "configs", f"{config_name}.yaml")

    print(f"Use config '{config_path}'")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    columns: Dict[str, Dict[str, Any]] = config["columns"]

    for rows in columns.values():
        for row_key in rows.keys():
            row_data = rows[row_key]
            rows[row_key] = (
                row_data["run"],
                row_data["ground_truth"],
                row_data["path"],
            )
    
    if config_name == "demo":
        generate_ram_figure(
            output_file=f"./outputs/ramachandran/{config_name}.pdf",
            columns=columns,
            wandb_main_path=wandb_main_path,
            base_path_pdf=None,
            figure_width=4.0,
        )
    elif config_name == "diff_algorithms_and_systems":
        generate_ram_figure(
            output_file=f"./outputs/ramachandran/{config_name}.pdf",
            columns=columns,
            wandb_main_path=wandb_main_path,
            base_path_pdf=f"./outputs/ramachandran/{config_name}_base.pdf",
        )
    