import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.special import digamma  # type: ignore

from mvg_variational_bayes._posterior_component_probability import PosteriorComponentProbability
from mvg_variational_bayes._data_dtype import data_dtype


class MultiVariateGaussVariationalBayes:
    """
    A class handling the variational approach algorithm for a multivariate gaussian mixture model.
    """

    def __init__(
        self,
        k: int,
        data: npt.NDArray[data_dtype],
        beta: Optional[float] = None,
        prior_mean: Optional[float] = None,
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        k: number of initial components
        data: data array
        beta: scale parameter for the variance of the means' prior normal distributions.
            Default: 1.0
        prior_mean: mean parameter for the mean value of the means' prior normal distributions.
            Default: The middle of the data's value range
        gamma: 0.5 * gamma is the shape parameter for the variances' prior inverse gamma distributions
            shifted by one, i.e. gamma = alpha + 1, where alpha is the shape parameter.
            Default: Selected such that the variance's prior mean resolves to half the data's value range.
        delta: 0.5 * delta is the scale parameter for the variances' prior inverse gamma distributions.
            Default: 2.0

        Raises
        ------
        AttributeError
            If the data array is not one-dimensional
        AttributeError
            If any of the parameters is out of range
        """
        self._k = k
        self._components_alive = np.full((k,), True, dtype=np.bool_)  # type: npt.NDArray[np.bool_]
        if len(data.shape) != 1:
            raise AttributeError(
                f"Data array has shape {data.shape} but needs to be one-dimensional. "
                f"When in doubt, flatten it before using it in this class."
            )
        self._data = data.copy()
        self._data_size = data.shape[0]
        data_min = float(np.min(data))
        data_max = float(np.max(data))

        self._alpha_initial = 1.0e-10
        self._alpha = np.full(self._k, self._alpha_initial, dtype=self._data.dtype)
        self._beta_initial = self._parse_initial_beta(beta)
        self._beta = np.full(self._k, self._beta_initial, dtype=self._data.dtype)
        self._hyper_mean_initial = self._parse_initial_prior_mean(prior_mean, data_min, data_max)
        self._hyper_mean = np.full(self._k, self._hyper_mean_initial, dtype=self._data.dtype)
        self._gamma_initial = self._parse_initial_gamma(gamma)
        self._gamma = np.full(self._k, self._gamma_initial, dtype=self._data.dtype)
        self._delta_initial = self._parse_initial_delta(delta, data_min, data_max)
        self._delta = np.full(self._k, self._delta_initial, dtype=self._data.dtype)

        self._q_z = PosteriorComponentProbability.random_init(self._data_size, k)

        self.tolerance = 1.0e-7
        self.conv_meas = 1.0
        self._converged = False

    @property
    def num_components(self) -> int:
        """The number of components (still) available in this multivariate gaussian approximation"""
        return self._k

    @property
    def components_alive(self) -> npt.NDArray[np.bool_]:
        """An array specifying which of the initial components are still alive."""
        return self._components_alive.copy()

    @property
    def weights(self) -> npt.NDArray[data_dtype]:
        """The weights for the components in the total a-posterior approximation."""
        return np.array(self._alpha / np.sum(self._alpha))

    @property
    def means(self) -> npt.NDArray[data_dtype]:
        """The mean values of the components' normal distributions.

        To be precise: The value m of the a-posterior approximation of the prior normal distribution of the mean value.
        """
        return self._hyper_mean

    @property
    def std(self) -> npt.NDArray[data_dtype]:
        """The standard deviation of the components' normal distributions

        To be precise: The mean value of the a-posterior approximation of the variance's inverse gamma prior.
        """
        # technically this is only correct if gamma > 2
        if np.any(self._gamma <= 2.0):
            raise ValueError(
                f"Cannot determine std since the value of gamma ({self._gamma}) is smaller than 2. for at "
                f"least one component."
            )
        return np.array(np.sqrt(self._delta / (self._gamma - 2)), dtype=data_dtype)

    @property
    def q_z(self) -> PosteriorComponentProbability:
        return self._q_z

    def set_hyperparameter_according_to_variance_range(self, mode: float, mean: float) -> None:
        """
        This function sets the gamma and delta hyperparameters for the variance prior inverse gamma distribution such
        that the distribution's mode and mean resemble the given parameters.
        The idea behind this is that by looking at the dataset one can infer the range of "widths" of the different
        components.
        In this notion the lower bound of component width can be set by an appropriate mode, the upper bound is adjusted
        by setting an appropriate mean value
        (which should be in the same order of magnitude as the upper bound but still much lower).
        Remember that the variance is the standard deviation squared.

        Furthermore, sets the beta hyperparameter (scaling the variance of the mean's prior normal distribution)
        such that it is half the data value range.
        """
        if mode >= mean:
            raise AttributeError(f"Mode (value: {mode}) has to be strictly smaller than mean (value: {mean}).")
        self._gamma_initial = 2.0 * (mode + mean) / (mean - mode)
        self._delta_initial = (self._gamma_initial - 2) * mean
        data_min = float(np.min(self._data))
        data_max = float(np.max(self._data))
        target_std = (data_max - data_min) / 2
        self._beta_initial = mean / target_std**2

    def _update_model_parameters(self) -> None:
        self._alpha = self._alpha_initial + self._q_z.importance
        self._beta = self._beta_initial + self._q_z.importance
        self._gamma = self._gamma_initial + self._q_z.importance
        self._hyper_mean = (
            self._beta_initial * self._hyper_mean_initial + np.sum(self._q_z.prob_matrix * self._data[:, None], axis=0)
        ) / self._beta
        self._delta = (
            self._delta_initial
            + np.sum(self._q_z.prob_matrix * (self._data * self._data)[:, None], axis=0)
            + self._beta_initial * self._hyper_mean_initial * self._hyper_mean_initial
            - self._beta * self._hyper_mean * self._hyper_mean
        )

    def _update_q_z(self) -> None:
        """
        updates prob distr :math:'q_{ij}' for hidden variables :math:'\{z_i\}'.
        Also calculates a "convergence measure" based on how much the distribution has changed since the last update.
        """
        log_q_z_part1 = (
            digamma(self._alpha)
            - digamma(np.sum(self._alpha))
            + 0.5 * (digamma(0.5 * self._gamma) - np.log(0.5 * self._delta) - 1 / self._beta)
        )
        log_q_z_part2 = (
            -0.5 * (self._gamma / self._delta) * (self._data[:, None] - self._hyper_mean[None, :]) ** 2
        )  # type: npt.NDArray[data_dtype]
        q_z = np.exp(log_q_z_part1[None, :] + log_q_z_part2)  # type: npt.NDArray[data_dtype]
        normal_const = np.sum(q_z, axis=1)
        non_zero_mask = normal_const != 0.0
        zero_mask = normal_const == 0.0
        q_z[non_zero_mask] = q_z[non_zero_mask] / (normal_const[:, None])[non_zero_mask]
        q_z[zero_mask] = np.zeros((np.sum(zero_mask), self._k), dtype=float)
        new_q_z = PosteriorComponentProbability(q_z)
        self.conv_meas = (np.abs(new_q_z - self._q_z)).max()
        self._q_z = new_q_z
        self._converged = self.conv_meas < self.tolerance

    def _reduce_complexity(self, epsilon: float = 1.0) -> npt.NDArray[np.bool_]:
        """
        remove all components i for which :math:'sum_i q_{ij} / sum_{ij} q_{ij} <' tolerance

        Return
        ------
        keep_mask : np.ndarray of bool
            a mask of booleans with size old_k and a True value for each component which stays and a False value for
            each component that has to go, such that e.g. a[keep_mask] has size new_k
        """

        measure = self._q_z.importance / np.sum(self._q_z.importance)
        keep_mask = measure > epsilon  # type: npt.NDArray[np.bool_]

        self._alpha = self._alpha[keep_mask]
        self._beta = self._beta[keep_mask]
        self._gamma = self._gamma[keep_mask]
        self._delta = self._delta[keep_mask]
        self._hyper_mean = self._hyper_mean[keep_mask]

        self._q_z = self._q_z.reduce_components(keep_mask)
        self._k = int(np.sum(keep_mask))

        self._components_alive[self._components_alive] = keep_mask
        return keep_mask

    def run(self, epsilon: float = 1.0) -> None:
        """Run the VB approximation.
        Afterward, the resulting values for the hyperparameters are accessible via
        the attributes a, b, c, d and m.
        Algorithm stops either by reaching the convergence tolerance or by issuing a KeyboardInterrupt.

        Parameters
        ----------
        epsilon : float
            Threshold for which components are eliminated.
            The threshold is compared to the value sum_i q_ij / sum_ij q_ij for each component j."""
        if self._converged:
            raise RuntimeError(f"Approximation already ran. Initialize class again to run another time")

        logging.info(f"Starting variational bayes approximation with {self._k} initial components.")
        iteration = 0
        try:
            while not self._converged:
                iteration += 1
                logging.info(
                    f"Iteration {iteration}: model complexity: {self._k}, convergence measure: {self.conv_meas}"
                )
                self.update_model(epsilon)
        except KeyboardInterrupt:
            logging.warning(f"Model optimization was interrupted by user.")
        else:
            logging.info(f"Optimization finished after {iteration} iterations. There are {self._k} components left.")

    def update_model(self, epsilon: float = 1.0) -> None:
        self._update_model_parameters()
        self._update_q_z()
        self._reduce_complexity(epsilon=epsilon)

    @staticmethod
    def _parse_initial_beta(beta: Optional[float]) -> float:
        if beta is None:
            return 1.0
        if beta <= 0:
            raise AttributeError(f"The beta parameter needs to be greater than 0. but is {beta}.")
        return beta

    @staticmethod
    def _parse_initial_prior_mean(prior_mean: Optional[float], data_min: float, data_max: float) -> float:
        if prior_mean is None:
            return data_min + (data_max - data_min) / 2
        return prior_mean

    @staticmethod
    def _parse_initial_gamma(gamma: Optional[float]) -> float:
        """
        mean = delta / (gamma - 2)
        mode = delta / (gamma + 2)
        """
        if gamma is None:
            return 3.0
        if gamma is None:
            return 1.0
        if gamma <= 2.0:
            raise AttributeError(f"The gamma hyperparameter needs to be greater than 2. but is {gamma}.")
        return gamma

    @staticmethod
    def _parse_initial_delta(delta: Optional[float], data_min: float, data_max: float) -> float:
        if delta is None:
            target_variance = ((data_max - data_min) / 2) ** 2
            # the motivation behind setting delta to the target variance of half the data's value range is that
            # together with gamma being 3.0 everywhere, this sets the mean of the prior inverse gamma distribution for
            # the variance to half the data range.
            return target_variance
        if delta <= 0.0:
            raise AttributeError(f"The delta hyperparameter needs to be greater than 0. but is {delta}.")
        return delta
