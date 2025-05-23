"""
This is code copied (with his permission) from Henrik Schopmans' repository https://github.com/henrik-schopmans/annealed_bg_paper.
"""

from auto_paper.figure_generation import FigureGenerator
from auto_paper.mpl.mpl_figure import MPLFigure
from auto_paper.utils.wandb_utils import wandb_get_pickle_file_path
import matplotlib.pyplot as plt


def generate_marginal_figure(
    output_file,
    exp_id="3y1u2ooq",
    figure_pkl_filename="300.0K_torsions_F_marginals_169999.pkl",
    wandb_main_path="/home/henrik/Dokumente/Promotion/AnnealedBG/annealed_bg/wandb/",
    n_columns=7,
    n_rows=9,
    n_total=59,
):

    legend_height = 0.2
    figure_width = 6.663
    height_per_row = 8.0 / 9.0

    ####################

    marginals_height = height_per_row * n_rows

    fg = FigureGenerator(figure_width, marginals_height + legend_height)

    pickle_path = wandb_get_pickle_file_path(
        figure_pkl_filename, wandb_main_path, exp_id
    )
    figure = MPLFigure(pickle_path=pickle_path)

    legend_box = fg.main_box.add_box(
        top=0.0,
        height=legend_height * 72.0,
        left=0.0,
        right=0.0,
        width=None,
        bottom=None,
    )
    marginal_box = fg.main_box.add_box(
        top=legend_height * 72.0,
        height=marginals_height * 72.0,
        left=0.0,
        right=0.0,
        width=None,
    )

    figure.change_size(figure_width, marginals_height)

    legend_figure = figure.create_separate_figure_from_legend(
        axis_index=0, figsize=(5, 0.5), ncol=3
    )
    legend_figure = MPLFigure(fig=legend_figure)
    legend_box.add_figure(legend_figure, align="centered", top_offset=5.0)

    figure.remove_all_legends()

    figure.set_x_range_to_max()
    # figure.set_y_range_to_max()

    y_label_text = figure.fig.get_axes()[0].get_ylabel()
    figure.fig.text(
        0.0,
        0.5,
        y_label_text,
        va="center",
        rotation="vertical",
        fontsize=8,
    )

    for i, ax in enumerate(figure.fig.axes):
        ax.set_xticks([0.0, 0.5, 1.0])

        if i % n_columns != 0:
            ax.set_xticklabels(["", "0.5", "1.0"])
        else:
            ax.set_xticklabels(["0.0", "0.5", "1.0"])

    figure.adjust_axes(
        n_columns=n_columns,
        n_rows=n_rows,
        n_total=n_total,
        remove_x_labels=False,
        remove_y_labels=True,
        also_remove_outer_x_labels=False,
        also_remove_outer_y_labels=True,
        remove_x_tick_labels=True,
        remove_y_tick_labels=True,
        also_remove_outer_y_tick_labels=False,
        also_remove_outer_x_tick_labels=False,
    )

    # Shift the x axis labels a bit higher:
    for ax in figure.fig.axes:
        ax.xaxis.labelpad = -0.5

    figure.activate_share_x(n_columns=n_columns)
    # figure.activate_share_y(n_columns=n_columns)

    figure.fig.tight_layout()
    figure.fig.subplots_adjust(wspace=0.1, hspace=0.2)

    marginal_box.add_figure(figure, align="centered")

    fg.save_pdf(output_file=output_file)


def ram_figure_prepare_subfigure(
    wandb_id: str,
    wandb_main_path: str,
    pickle_filename_str: str,
    figure_width: float,
    figure_height: float,
    select_ground_truth: bool = False,
    select_colorbar: bool = False,
    hide_x_label=False,
    hide_y_label=False,
    hide_x_tick_labels=False,
    hide_y_tick_labels=False,
    shift_x_label_down_by: float = 0.0,
    remove_x_tick_labels=False,
    remove_y_tick_labels=False,
):
    assert not (select_ground_truth and select_colorbar)

    pickle_path = wandb_get_pickle_file_path(
        pickle_filename_str, wandb_main_path, wandb_id
    )

    original_figure = MPLFigure(pickle_path=pickle_path)

    new_fig = plt.figure(figsize=(figure_width / 72.0, figure_height / 72.0))
    original_axis = original_figure.fig.axes[0 if not select_ground_truth else 1]

    if select_colorbar:  # Create the figure just including the colorbar

        new_axis = new_fig.gca()

        heatmap = original_axis.images[0]
        sm = plt.cm.ScalarMappable(cmap=heatmap.get_cmap(), norm=heatmap.norm)
        cbar = new_fig.colorbar(sm, cax=new_axis, orientation="vertical")
        new_axis.set_ylabel(r"free energy / $k_\text{B} T$")

        cbar.set_ticks([0.0, 5.0, 10.0])

        # Move axis a bit to the left:
        new_axis.yaxis.labelpad -= 3.0

        # new_fig.savefig(
        #     f"test_{wandb_id}.pdf",
        #     dpi=600,
        # )

        return MPLFigure(fig=new_fig)

    new_axis = new_fig.add_subplot(111)

    # Extract the heatmap (imshow) data from the original axis
    heatmap = original_axis.images[0]

    data = heatmap.get_array()
    extent = heatmap.get_extent()
    cmap = heatmap.get_cmap()
    norm = heatmap.norm

    cmap.set_bad(color="white")

    # Recreate the heatmap on the new axis
    new_axis.imshow(
        data,
        extent=extent,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="none",
        origin="lower",
    )

    new_axis.set_xticks([0.0, 0.5, 1.0])
    new_axis.set_yticks([0.0, 0.5, 1.0])

    orig_x_label = original_axis.get_xlabel()
    if "dim" in orig_x_label:
        orig_x_label = orig_x_label.split("(")[0].strip()
    new_axis.set_xlabel(orig_x_label, fontsize=7)

    if hide_x_label:
        new_axis.get_xaxis().label.set_color("none")
    if shift_x_label_down_by != 0.0:
        new_axis.xaxis.labelpad += shift_x_label_down_by

    orig_y_label = original_axis.get_ylabel()
    if "dim" in orig_y_label:
        orig_y_label = orig_y_label.split("(")[0].strip()
    new_axis.set_ylabel(orig_y_label, fontsize=7)
    if hide_y_label:
        new_axis.get_yaxis().label.set_color("none")

    new_axis.xaxis.labelpad -= 3.0
    new_axis.yaxis.labelpad -= 3.0

    # Equal aspect ratio:
    new_axis.set_aspect("equal")

    # Set background color to transparent to avoid overlaps:
    new_axis.patch.set_alpha(0.0)

    # Remove tick labels:
    if hide_x_tick_labels:
        minor_labels = new_axis.get_xticklabels(minor=True)
        major_labels = new_axis.get_xticklabels(minor=False)
        for label in minor_labels + major_labels:
            label.set_color("none")

    if remove_x_tick_labels:
        new_axis.set_xticklabels([])

    if hide_y_tick_labels:
        minor_labels = new_axis.get_yticklabels(minor=True)
        major_labels = new_axis.get_yticklabels(minor=False)
        for label in minor_labels + major_labels:
            label.set_color("none")

    if remove_y_tick_labels:
        new_axis.set_yticklabels([])

    new_fig = MPLFigure(fig=new_fig)

    return new_fig


def generate_ram_figure(
    output_file,
    columns,
    wandb_main_path,
    top_labels_rightshift=5.0,
    left_labels_bottomshift=-5.0,
    hide_left_labels: bool = False,
    remove_all_tick_labels=True,
    height_per_row=1.12,
    colorbar_column_width=35.0,
    colorbar_height_offset=5.0,
    colorbar_y_offset=-3.0,
    shift_x_labels_and_cb_down=False,
    base_path_pdf=None,
    figure_width=6.663,  # in inches
):

    top_padding_width = 18.0
    left_padding_width = 13.0

    bottom_padding_width = 20.0

    figure_height = (
        top_padding_width / 72.0
        + height_per_row * len(columns[list(columns.keys())[0]].keys())
        + bottom_padding_width / 72.0
    )

    if base_path_pdf is not None:
        fg = FigureGenerator(base_pdf_path=base_path_pdf)
    else:
        fg = FigureGenerator(figure_width, figure_height)

    # Define top and left padding
    top_padding = fg.main_box.add_box(
        left=left_padding_width,
        top=0.0,
        width=None,
        height=top_padding_width,
        right=colorbar_column_width,
        bottom=None,
    )

    left_padding = fg.main_box.add_box(
        left=0.0,
        top=top_padding_width,
        width=left_padding_width,
        height=None,
        right=None,
        bottom=0.0,
    )

    # Add column headers based on keys in `experiments`:
    column_keys = list(columns.keys())
    column_width = top_padding.width / len(column_keys)

    for idx, column_key in enumerate(column_keys):
        top_padding.add_box(
            left=idx * column_width,
            right=None,
            width=column_width,
            top=5.0,
            bottom=0.0,
            height=None,
        ).add_latex(
            f"\\textbf{{{column_key}}}",
            align="centered",
            left_offset=top_labels_rightshift,
        )

    # Add row labels dynamically based on system names in the first column
    system_names = list(columns[column_keys[0]].keys())
    row_height_percentage = 100.0 / len(system_names)

    for idx, system_name in enumerate(system_names):
        current_box = left_padding.add_box(
            left=0.0,
            right=0.0,
            width=None,
            top=f"{idx * row_height_percentage}%",
            height=f"{row_height_percentage}%",
            bottom=None,
        )

        if not hide_left_labels:
            current_box.add_latex(
                f"\\textbf{{{system_name.capitalize()}}}",
                align="centered",
                rotation=90.0,
                top_offset=left_labels_bottomshift,
            )

    # Create main figure area
    figures_main_box = fg.main_box.add_box(
        left=left_padding_width,
        top=top_padding_width,
        width=None,
        height=None,
        right=colorbar_column_width,
        bottom=0.0,
    )
    colorbar_box = fg.main_box.add_box(
        left=None,
        top=top_padding_width,
        width=colorbar_column_width,
        height=None,
        right=0.0,
        bottom=0.0,
    )

    subfigures_width = figures_main_box.width / len(column_keys) + 7.0
    subfigures_height = figures_main_box.height / len(system_names) + 7.0

    for col_idx, column_key in enumerate(column_keys):
        column_box = figures_main_box.add_box(
            left=col_idx * column_width,
            top=0.0,
            width=column_width,
            height=None,
            bottom=0.0,
            right=None,
        )

        for row_idx, system_name in enumerate(system_names):
            subfigure_box = column_box.add_box(
                left=0.0,
                top=f"{row_idx * row_height_percentage}%",
                width=None,
                height=f"{row_height_percentage}%",
                bottom=None,
                right=0.0,
            )

            wandb_id, select_ground_truth, pickle_filename = columns[column_key][
                system_name
            ]
            figure = ram_figure_prepare_subfigure(
                wandb_id=wandb_id,
                wandb_main_path=wandb_main_path,
                pickle_filename_str=pickle_filename,
                figure_width=subfigures_width,
                figure_height=subfigures_height,
                select_ground_truth=select_ground_truth,
                hide_x_label=False,
                hide_y_label=(col_idx != 0),
                hide_x_tick_labels=(row_idx != len(system_names) - 1),
                hide_y_tick_labels=(col_idx != 0),
                shift_x_label_down_by=(
                    (0.0 if row_idx == len(system_names) - 1 else -7.0)
                    if shift_x_labels_and_cb_down
                    else 0.0
                ),
                remove_x_tick_labels=remove_all_tick_labels,
                remove_y_tick_labels=remove_all_tick_labels,
            )
            subfigure_box.add_figure(figure, align="centered")

            if col_idx == 0:
                colorbar_subbox = colorbar_box.add_box(
                    left=0.0,
                    top=f"{row_idx * row_height_percentage}%",
                    width=None,
                    height=f"{row_height_percentage}%",
                    bottom=None,
                    right=0.0,
                )
                colorbar_figure = ram_figure_prepare_subfigure(
                    wandb_id=wandb_id,
                    wandb_main_path=wandb_main_path,
                    pickle_filename_str=pickle_filename,
                    figure_width=7.0,
                    figure_height=subfigures_height - 5.0 + colorbar_height_offset,
                    select_colorbar=True,
                )
                colorbar_subbox.add_figure(
                    colorbar_figure,
                    align="centered",
                    left_offset=2.0,
                    top_offset=(
                        (-5.5 if (row_idx == len(system_names) - 1) else -2.0)
                        if shift_x_labels_and_cb_down
                        else 0.0
                    )
                    + colorbar_y_offset,
                )

    # fg.draw_bounding_boxes()

    fg.save_pdf(output_file=output_file)


if __name__ == "__main__":
    experiments = {
        "Ground truth": {
            "Dipeptide": (
                "8drj0amh",
                True,
                "base_300.0K_torsions_F_ramachandran_4_11_<i>.pkl",
            ),
            "Tetrapeptide": (
                "79ngf1l1",
                True,
                "base_300.0K_torsions_F_ramachandran_0_1_<i>.pkl",
            ),
            "Hexapeptide": (
                "n9ssk55r",
                True,
                "base_300.0K_torsions_F_ramachandran_6_7_<i>.pkl",
            ),
        },
        "FAB": {
            "Dipeptide": (
                "8drj0amh",
                False,
                "base_torsions_F_ramachandran_4_11_<i>.pkl",
            ),
            "Tetrapeptide": (
                "79ngf1l1",
                False,
                "base_torsions_F_ramachandran_0_1_<i>.pkl",
            ),
            "Hexapeptide": (
                "n9ssk55r",
                False,
                "base_torsions_F_ramachandran_6_7_<i>.pkl",
            ),
        },
        "PlaceHolder1": {
            "Dipeptide": (
                "8drj0amh",
                False,
                "base_torsions_F_ramachandran_4_11_<i>.pkl",
            ),
            "Tetrapeptide": (
                "79ngf1l1",
                False,
                "base_torsions_F_ramachandran_0_1_<i>.pkl",
            ),
            "Hexapeptide": (
                "n9ssk55r",
                False,
                "base_torsions_F_ramachandran_6_7_<i>.pkl",
            ),
        },
        "PlaceHolder2": {
            "Dipeptide": (
                "8drj0amh",
                False,
                "base_torsions_F_ramachandran_4_11_<i>.pkl",
            ),
            "Tetrapeptide": (
                "79ngf1l1",
                False,
                "base_torsions_F_ramachandran_0_1_<i>.pkl",
            ),
            "Hexapeptide": (
                "n9ssk55r",
                False,
                "base_torsions_F_ramachandran_6_7_<i>.pkl",
            ),
        },
        "PlaceHolder3": {
            "Dipeptide": (
                "8drj0amh",
                False,
                "base_torsions_F_ramachandran_4_11_<i>.pkl",
            ),
            "Tetrapeptide": (
                "79ngf1l1",
                False,
                "base_torsions_F_ramachandran_0_1_<i>.pkl",
            ),
            "Hexapeptide": (
                "n9ssk55r",
                False,
                "base_torsions_F_ramachandran_6_7_<i>.pkl",
            ),
        },
    }

    wandb_main_path = "/home/henrik/Dokumente/Promotion/AnnealedBG/annealed_bg/wandb/"

    generate_ram_figure(
        output_file="./figures/figure_2/main.pdf",
        columns=experiments,
        wandb_main_path=wandb_main_path,
    )