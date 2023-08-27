from pathlib import Path
from typing import Optional

import matplotlib.cm as cm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import numpy.typing as npt
from scipy.stats import norm  # type: ignore

from mvg_variational_bayes import MultiVariateGaussVariationalBayes, data_dtype


def plot_result(
    final_gaussian: MultiVariateGaussVariationalBayes,
    flattened_data: npt.NDArray[data_dtype],
    colors: npt.NDArray[np.float64],
    iteration: Optional[int] = None,
    save_path: Optional[Path] = None,
) -> None:
    bins = 256
    histogram = np.histogram(flattened_data, density=True, bins=bins)
    x = histogram[1][:-1]
    kernel_density = histogram[0]
    pdf_x_s = [
        final_gaussian.weights[i] * norm.pdf(x, final_gaussian.means[i], final_gaussian.std[i])
        for i in range(final_gaussian.num_components)
    ]
    pdf_x = np.sum(pdf_x_s, axis=0)

    fig = plt.figure()
    ax = fig.gca()
    ax.cla()
    ax.fill_between(x, kernel_density, facecolor="gray", alpha=0.2)
    ax.plot(x, kernel_density, "-", color="black")
    ax.plot(x, pdf_x, "r-")
    for i, gauss in enumerate(pdf_x_s):
        ax.plot(x, gauss, "-", color=colors[i])
    top_left_text = f"k = {final_gaussian.num_components:2d}"
    if iteration is not None:
        top_left_text += f", step: {iteration:4d}"
    ax.text(0.05, 0.9, top_left_text, transform=ax.transAxes)
    y_max = np.max(kernel_density[10:-10])  # avoid possible extremes on the edges of the datasets kernel density
    ax.set_ylim(0.0, y_max)
    if save_path is not None:
        fig.savefig(save_path)
    else:
        fig.show()
