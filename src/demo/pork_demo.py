import logging
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import numpy.typing as npt
from matplotlib import colors, colormaps

from data_loader import load_pork_data  # type: ignore
from mvg_variational_bayes import MultiVariateGaussVariationalBayes, data_dtype
from mvg_variational_bayes.helper_scripts import plot_result

IMAGE_OUTPUT_DIR = Path(__file__).parent / "img"


def run_with_intermediate_plots(
    model: MultiVariateGaussVariationalBayes, flattened_data: npt.NDArray[data_dtype], colors: npt.NDArray[np.float64]
) -> None:
    iteration = 0
    picture_id = 0
    while not model.converged:
        iteration += 1
        logging.info(f"Iteration {iteration}: model complexity: {model.num_components}")
        model.update_model(0.019)
        if iteration == int(1.5 ** (picture_id + 1)):
            logging.info(f"Creating plot {picture_id} at iteration {iteration}")
            plot_result(
                model,
                flattened_data,
                colors[model.components_alive],
                iteration=iteration,
                save_path=IMAGE_OUTPUT_DIR / f"pork_timelaps_{picture_id:03d}.png",
            )
            picture_id += 1
    plot_result(
        model,
        flattened_data,
        colors[model.components_alive],
        iteration=iteration,
        save_path=IMAGE_OUTPUT_DIR / f"pork_timelaps_{picture_id:03d}.png",
    )


def plot_pork_image(
    model: MultiVariateGaussVariationalBayes, data: npt.NDArray[data_dtype], component_colors: npt.NDArray[np.float64]
) -> None:
    cmap_array = [colors.ListedColormap([color]) for color in component_colors]
    for component_id in range(model.num_components):
        plt.clf()
        plt.imshow(data, cmap="gray")
        test = model.q_z.prob_matrix[:, component_id].reshape(data.shape)
        plt.imshow(np.ones(data.shape), cmap=cmap_array[component_id], alpha=test)
        plt.savefig(IMAGE_OUTPUT_DIR / f"pork_comp_{component_id}.png")


def create_timelaps_gif() -> None:
    timelaps_images = IMAGE_OUTPUT_DIR.glob("pork_timelaps_*.png")
    frames = np.stack([iio.imread(file_path) for file_path in sorted(timelaps_images)], axis=0)
    iio.imwrite(IMAGE_OUTPUT_DIR / "pork_timelaps.gif", frames, duration=500, loop=20)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    data_flatten, data_raw = load_pork_data()
    initial_num_components = 20
    comp_colors = np.array([colormaps["viridis"](i / initial_num_components) for i in range(initial_num_components)])

    mvgvb = MultiVariateGaussVariationalBayes(initial_num_components, data_flatten)
    mvgvb.set_hyperparameter_according_to_variance_range(0.5, 20.0)
    mvgvb.tolerance = 0.001

    run_with_intermediate_plots(mvgvb, data_flatten, comp_colors)

    plot_pork_image(mvgvb, data_raw, comp_colors[mvgvb.components_alive])
    plot_result(mvgvb, data_flatten, comp_colors[mvgvb.components_alive], save_path=IMAGE_OUTPUT_DIR / "pork_final.png")
    create_timelaps_gif()
