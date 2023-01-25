"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from scipy.stats import multivariate_normal


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    p_K = np.zeros((n, 1))
    Cu = []

    for i in range(n):
        Cu.append(np.nonzero(X[i]))
        for j in range(K):
            if np.array(Cu[i]).size == 0:
                post[i, j] = np.log(mixture.p[j])
            else:
                post[i, j] = np.log(
                    multivariate_normal.pdf(X[i][Cu[i]], mixture.mu[j][Cu[i]], mixture.var[j]) * mixture.p[j])
            p_K[i] = logsumexp(post[i])
    post = np.exp(post - np.multiply(p_K, np.ones((n, K))))
    cost = np.sum(np.log(np.exp(post).sum(axis=1)))
    return post, cost



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    mu = np.zeros((K, d))
    var = np.zeros(K)
    delta = np.zeros((n, d))
    den = np.zeros((n, K))
    Cu = []

    p = post.sum(axis=0) / n

    for u in range(n):
        Cu.append(np.nonzero(X[u]))
        for l in range(d):
            delta[u, l] = 1 if l in np.array(Cu[u]) else 0

    for j in range(K):
        for u in range(n):
            for l in range(d):
                if sum(np.multiply(post[:, j], delta[:, l])) >= 1:
                    mu[j, l] = sum(np.multiply(np.multiply(post[:, j], delta[:, l]), X[:, l])) / sum(
                        np.multiply(post[:, j], delta[:, l]))
                else:
                    mu[j, l] = mixture.mu[j, l]
                sse = (np.multiply((mu[j] - X), delta) ** 2).sum(axis=1) @ post[:, j]
                den[u, j] = np.array(Cu[u]).size * post[u, j]
                if sse / sum(den[:, j]) >= 0.25:
                    var[j] = sse / sum(den[:, j])
                else:
                    var[j] = 0.25
    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_cost = None
    cost = None
    while (prev_cost is None or cost - prev_cost > 1e-6 * abs(cost)):
        prev_cost = cost
        post, cost = estep(X, mixture)
        mixture = mstep(X, post, mixture)
    return mixture, post, cost


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    p_K = np.zeros((n, 1))
    Cu = []

    for i in range(n):
        Cu.append(np.nonzero(X[i]))
        for j in range(K):
            if np.array(Cu[i]).size == 0:
                post[i, j] = np.log(mixture.p[j])
            else:
                post[i, j] = np.log(
                    multivariate_normal.pdf(X[i][Cu[i]], mixture.mu[j][Cu[i]], mixture.var[j]) * mixture.p[j])
            p_K[i] = logsumexp(post[i])
    post = np.exp(post - np.multiply(p_K, np.ones((n, K))))
    mask = (X != 0)
    X_m = np.ma.array(X, mask=mask)
    X = X + np.dot(post, mixture.mu) * ~X_m.mask
    return X
