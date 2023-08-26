# Variational Bayesian Inference - A case study with multivariate Gaussian approximation

This project was part of a seminar during my Master studies in 
[Theoretical and Mathematical Physics](https://www.theorie.physik.uni-muenchen.de/TMP/) in 2018.
It's a small python project to classify an unknown amount of components in e.g. a CT scan.

The whole project is largely based on a case-study by Clare Anne McGrory documented in her PhD Thesis 
"Variational Approximations in Bayesian Model Selection".


## Working with the source code

Install dependencies including the dev dependencies:

    pip install -e .[dev]

## Theory overview

*For a more detailed description see the complementary [presentation](presentation.pdf).*

The project approximates the posterior of a multivariate gaussian bayesian hierarchical model.
It assumes that a dataset $`y`$ is distributed according to an unknown number of gaussian distributed components:
```math
p(y_i | \mathbf \lambda, \mathbf \mu, \mathbf \sigma ) = \sum_{j=1}^N \lambda_j \mathcal N (y_i; \mu_j, \sigma_j)
```

Full model boils down to the following system of parameters:
![full hierarchical mixture model](./img/mixture_model_hierachal.svg)

The following priors are used in the hierarchical model:
* $`\mu`$: Normal distributed with mean $`m`$ and variance $`\beta^{-1}\sigma^2`$
* $`\sigma^2`$: Inverse gamma distributed with shape $`0.5 \gamma`$ and scale $`0.5 \delta`$
* $`\lambda`$: Dirichlet distributed with concentration parameters $`\alpha`$


