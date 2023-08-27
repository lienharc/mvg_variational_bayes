import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt

from mvg_variational_bayes import data_dtype

DEFAULT_PORK_PATH = Path(__file__).parent / "data" / "porkScan_porkScan.csv"


def load_pork_data(file: Path = DEFAULT_PORK_PATH) -> Tuple[npt.NDArray[data_dtype], npt.NDArray[data_dtype]]:
    """
    :param file: The path to the csv file containing the CT scan of a pork.
    :return:
    """
    with file.open("r") as f:
        reader = csv.reader(f)
        result_list = list(reader)
    data_raw = np.array(result_list[1:], dtype=data_dtype)[:, 1:]  # type: npt.NDArray[data_dtype]
    data_flatten = data_raw.flatten()
    return data_flatten, data_raw
