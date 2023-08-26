import numpy as np
import pytest

from mvg_variational_bayes import MultiVariateGaussVariationalBayes
from mvg_variational_bayes._data_dtype import data_dtype


class TestConstructor:
    TEST_COMPONENTS = 7

    def test_raises_if_data_array_not_flattened(self) -> None:
        with pytest.raises(AttributeError) as exc_info:
            MultiVariateGaussVariationalBayes(self.TEST_COMPONENTS, np.array([[1, 2], [2, 1]]))

        assert "one-dimensional" in str(exc_info.value)

    def test_passes_beta_parameter(self) -> None:
        mvgvb = MultiVariateGaussVariationalBayes(self.TEST_COMPONENTS, np.array([1, 2, 2]), beta=3.0)

        assert mvgvb._beta_initial == 3.0
        assert np.all(np.all(mvgvb._beta == np.full(self.TEST_COMPONENTS, 3.0, dtype=data_dtype)))

    def test_raises_if_beta_is_not_greater_than_zero(self) -> None:
        with pytest.raises(AttributeError) as exc_info:
            MultiVariateGaussVariationalBayes(self.TEST_COMPONENTS, np.array([1, 2, 2]), beta=-1)

        assert "beta" in str(exc_info.value)
        assert "greater than 0" in str(exc_info.value)

    def test_passes_gamma_parameter(self) -> None:
        mvgvb = MultiVariateGaussVariationalBayes(self.TEST_COMPONENTS, np.array([1, 2, 2]), gamma=3.0)

        assert mvgvb._gamma_initial == 3.0
        assert np.all(mvgvb._gamma == np.full(self.TEST_COMPONENTS, 3.0, dtype=data_dtype))

    def test_raises_if_gamma_is_not_greater_than_zero(self) -> None:
        with pytest.raises(AttributeError) as exc_info:
            MultiVariateGaussVariationalBayes(self.TEST_COMPONENTS, np.array([1, 2, 2]), gamma=1.5)

        assert "gamma" in str(exc_info.value)
        assert "greater than 2." in str(exc_info.value)

    def test_passes_delta_parameter(self) -> None:
        mvgvb = MultiVariateGaussVariationalBayes(self.TEST_COMPONENTS, np.array([1, 2, 2]), delta=3.0)

        assert mvgvb._delta_initial == 3.0
        assert np.all(mvgvb._delta == np.full(self.TEST_COMPONENTS, 3.0, dtype=data_dtype))

    def test_raises_if_delta_is_not_greater_than_two(self) -> None:
        with pytest.raises(AttributeError) as exc_info:
            MultiVariateGaussVariationalBayes(self.TEST_COMPONENTS, np.array([1, 2, 2]), delta=-1.5)

        assert "delta" in str(exc_info.value)
        assert "greater than 0" in str(exc_info.value)

    def test_passes_prior_mean(self) -> None:
        mvgvb = MultiVariateGaussVariationalBayes(self.TEST_COMPONENTS, np.array([1, 2, 2]), prior_mean=-3.0)

        assert mvgvb._hyper_mean_initial == -3.0
        assert np.all(mvgvb._hyper_mean == np.full(self.TEST_COMPONENTS, -3.0, dtype=data_dtype))

    def test_default_hyperparamter_values(self) -> None:
        var_bayes = MultiVariateGaussVariationalBayes(1, np.array([10.0, 20.0]))

        # prior mean should be in the middle of the data range
        assert var_bayes.means == 15.0
        # prior variance should be half the data range
        assert var_bayes.std == 5.0
        # (shape - 1) of inverse gamma distribution for variance defaults to 1.
        assert var_bayes._gamma_initial == 3.0
        assert var_bayes._beta_initial == 1.0
        # default is such that the mean value for the variance's inverse gamma prior defaults to half the data range
        assert var_bayes._delta_initial == 25.0


def test_set_hyperparameter_according_to_variance_range() -> None:
    var_bayes = MultiVariateGaussVariationalBayes(1, np.array([10.0, 20.0]))
    var_bayes.set_hyperparameter_according_to_variance_range(1.0, 5.0)

    # prior mean should be in the middle of the data range
    assert var_bayes.means[0] == 15.0
    # prior variance should be half the data range
    assert var_bayes.std[0] == 5.0
    # (shape - 1) of inverse gamma distribution for variance defaults to 1.
    assert var_bayes._gamma_initial == 3.0
    assert var_bayes._beta_initial == 0.2
    # scale of inverse gamma distribution for variance defaults to 1/25 such that the prior variance for the mean
    # value is also half the data range.
    assert var_bayes._delta_initial == 5.0


def test_simple_approximation() -> None:
    input_array = np.array([0.5, 2.0, 2.0, 2.0, 3.5, 10.0, 10.0, 10.0, 10.0])
    var_bayes = MultiVariateGaussVariationalBayes(5, input_array)
    var_bayes.set_hyperparameter_according_to_variance_range(0.001, 2.0)
    var_bayes.run(epsilon=1e-10)

    assert var_bayes.num_components == 2
    sorted_means = np.sort(var_bayes.means)
    assert sorted_means[0] == pytest.approx(2.0, abs=0.5)
    assert sorted_means[1] == pytest.approx(10.0, abs=0.5)
