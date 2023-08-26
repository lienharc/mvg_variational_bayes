import numpy as np

from mvg_variational_bayes import PosteriorComponentProbability


def test_random_init() -> None:
    pcp = PosteriorComponentProbability.random_init(5, 3)

    assert pcp.shape == (5, 3)
    prob_sum_per_observation = np.sum(pcp.prob_matrix, axis=1)
    assert np.all(
        prob_sum_per_observation == 1.0
    ), "The probabilities to belong to a component should be sum to 1. for each 'observation'"
    number_of_ones_per_observation = np.sum(pcp.prob_matrix == 1.0, axis=1)
    assert np.all(
        number_of_ones_per_observation == 1.0
    ), "After random_init each observation should be 'attached' to exactly one component"


def test_reduce_components() -> None:
    input_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    expected_matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )
    pcp = PosteriorComponentProbability(input_matrix)

    new_pcp = pcp.reduce_components(keep_mask=np.array([True, True, False]))

    assert new_pcp.prob_matrix == expected_matrix
