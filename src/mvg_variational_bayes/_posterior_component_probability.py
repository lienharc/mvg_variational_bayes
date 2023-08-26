from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np
from numpy import typing as npt

from mvg_variational_bayes._data_dtype import data_dtype


@dataclass
class PosteriorComponentProbability:
    """A dataclass representing the posterior probability that an observation i belongs to a component j.

    Wrapper around a matrix of dimension data_size x num_components"""

    _prob_matrix: npt.NDArray[data_dtype]
    _importance: Optional[npt.NDArray[data_dtype]] = field(init=False, default=None)

    @classmethod
    def random_init(cls, data_size: int, num_components: int) -> "PosteriorComponentProbability":
        """
        randomly selects a component for each measurement i which is 3x as probable as all other components
        """
        prob_matrix = np.zeros((data_size, num_components), dtype=data_dtype)  # type: npt.NDArray[data_dtype]
        data_indices = np.arange(data_size)
        # assign each data point to a component at random
        component_selection = np.random.randint(0, num_components, data_size)
        prob_matrix[data_indices, component_selection] = 1.0
        return cls(prob_matrix)

    @property
    def prob_matrix(self) -> npt.NDArray[data_dtype]:
        return self._prob_matrix.copy()

    @property
    def importance(self) -> npt.NDArray[data_dtype]:
        """Returns a 1D array of size "num_components" describing the "importance" of each component.
        It is a measure of how many observations are likely to belong to a component
        (summing the probability of all observations)."""
        if self._importance is None:
            self._importance = np.array(np.sum(self._prob_matrix, axis=0), dtype=data_dtype)
        return self._importance

    @property
    def shape(self) -> Tuple[int, int]:
        return self._prob_matrix.shape

    def reduce_components(self, keep_mask: npt.NDArray[np.bool_]) -> "PosteriorComponentProbability":
        """Creates a new PosteriorComponentProbability class but only with the components specified by keep_mask.

        :param keep_mask: A 1D array containing a boolean value for each component stating whether to keep or remove it.
        """
        if len(keep_mask.shape) != 1 or keep_mask.shape[0] != self.prob_matrix.shape[1]:
            raise AttributeError(
                f"Unexpected shape of keep_mask: {keep_mask.shape}. Expected: ({self.prob_matrix.shape[1]},)"
            )

        new_prob_matrix = self.prob_matrix.transpose()[keep_mask].transpose()
        return PosteriorComponentProbability(new_prob_matrix)

    def __sub__(
        self, other: Union["PosteriorComponentProbability", npt.NDArray[data_dtype]]
    ) -> npt.NDArray[data_dtype]:
        if isinstance(other, PosteriorComponentProbability):
            return np.array(self._prob_matrix - other.prob_matrix, dtype=data_dtype)
        if isinstance(other, np.ndarray):
            return np.array(self._prob_matrix - other, dtype=data_dtype)
        raise TypeError("Unsupported operand type for subtracting from PosteriorComponentProbability")
