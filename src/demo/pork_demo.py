import logging
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import numpy.typing as npt
from matplotlib import colors, colormaps

from data_loader import load_pork_data  # type: ignore
from mvg_variational_bayes import MultiVariateGaussVariationalBayes, data_dtype
from mvg_variational_bayes.helper_scripts import plot_result

IMAGE_OUTPUT_DIR = Path(__file__).parent / "img"


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    data_flatten, data_raw = load_pork_data()
    initial_num_components = 20
    comp_colors = np.array([colormaps["viridis"](i / initial_num_components) for i in range(initial_num_components)])

    mvgvb = MultiVariateGaussVariationalBayes(initial_num_components, data_flatten)
    mvgvb.set_hyperparameter_according_to_variance_range(0.5, 20.0)
    mvgvb.tolerance = 0.001

    mvgvb.run(0.019)
    plot_pork_image(mvgvb, data_raw, comp_colors[mvgvb.components_alive])
    plot_result(mvgvb, data_flatten, comp_colors[mvgvb.components_alive], save_path=IMAGE_OUTPUT_DIR / "pork_final.png")
