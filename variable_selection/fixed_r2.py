import numpy as np
import pandas as pd
from numpy.random import multivariate_normal


def fixed_R2(n, p, sigma=1, R2=0.5, rho=0.1):
    """
    Generate synthetic data with a fixed R-squared value.

    Parameters:
    -----------
    n : int
        Number of observations.
    p : int
        Number of predictors.
    sigma : float
        Standard deviation of the error term.
    R2 : float
        Target R-squared value (between 0 and 1).
    rho : float
        Correlation coefficient for the predictors.

    Returns:
    - pandas.DataFrame: A DataFrame containing synthetic data with columns 'y' and 'X1', 'X2', ..., 'Xp'.
    - numpy.ndarray: The generated beta coefficients.
    """
    # Generate beta coefficients based on R2
    if R2:
        # Calculate the variance of beta based on R2
        var_betas = sigma**2 / p / (1 / R2 - 1)
        # Generate random beta coefficients from a normal distribution
        betas = np.random.normal(0, (var_betas) ** 0.5, p)
    else:
        betas = np.zeros(p)

    # Set mean and covariance matrix for multivariate normal distribution
    mu = np.zeros(p)
    Sigma = np.eye(p)

    # Fill off-diagonal elements of Sigma
    np.fill_diagonal(Sigma, rho)

    # Generate random predictors X from a multivariate normal distribution
    X = multivariate_normal(mu, Sigma, n)

    # Generate response variable y based on predictors X, beta, and normal error term
    y = X @ betas + np.random.normal(0, sigma, n)

    # Create a DataFrame with columns 'y' and 'X1', 'X2', ..., 'Xp'
    df = pd.DataFrame(
        np.column_stack((y, X)), columns=["y"] + [f"X{i}" for i in range(1, p + 1)]
    )

    return df, betas
