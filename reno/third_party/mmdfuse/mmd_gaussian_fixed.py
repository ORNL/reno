# =============================================================================
#   Gaussian-kernel MMD distance with fixed bandwidths.
#
#   This file modifies the MMD-FUSE implementation by removing two-sample
#   testing, permutations, softmax/logsumexp fusion, and data-dependent
#   bandwidth selection. It uses only Gaussian kernels with fixed bandwidths
#   averages the kernel matrices into a single multi-scale Gaussian kernel.
#
#   The modification allows using MMD as a metric to pairwise compare
#   multiple distributions in a consistent manner.
# =============================================================================

import numpy as np


# Fixed, data-independent Gaussian bandwidths. Ideally, features are already
# on a comparable scale, e.g. standardized/z-scored or otherwise normalized.
DEFAULT_FIXED_BANDWIDTHS = np.array([0.1, 0.3, 1.0, 3.0, 10.0], dtype=float)


def np_distances(X, Y, l="l2", max_samples=None, matrix=False):
    """
    Compute pairwise l1 or l2 distances using NumPy broadcasting.

    Parameters
    ----------
    X, Y : ndarray, shape (n_samples, n_features)
    l : {"l1", "l2"}
    max_samples : int or None
        Optional cap applied to both X and Y before computing distances.
    matrix : bool
        If True, return the full distance matrix. If False, return the
        upper-triangular entries of the distance matrix.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    Xs = X[:max_samples]
    Ys = Y[:max_samples]

    diff = Xs[:, None, :] - Ys[None, :, :]

    if l == "l1":
        output = np.sum(np.abs(diff), axis=-1)
    elif l == "l2":
        output = np.sqrt(np.sum(diff**2, axis=-1))
    else:
        raise ValueError("Value of 'l' must be either 'l1' or 'l2'.")

    if matrix:
        return output
    return output[np.triu_indices(output.shape[0])]


def gaussian_kernel_matrix(pairwise_l2_distances, bandwidth):
    """Gaussian/RBF kernel matrix from an l2 distance matrix."""
    d = pairwise_l2_distances / bandwidth
    return np.exp(-(d**2) / 2)


def average_gaussian_kernel_matrix(pairwise_l2_distances, bandwidths=DEFAULT_FIXED_BANDWIDTHS):
    """
    Build one multi-scale Gaussian kernel by averaging Gaussian kernels over
    fixed bandwidths. No softmax, no learned weights, no data-dependent tuning.
    """
    bandwidths = np.asarray(bandwidths, dtype=float)
    if bandwidths.ndim != 1 or len(bandwidths) == 0:
        raise ValueError("bandwidths must be a non-empty 1D array.")
    if np.any(bandwidths <= 0):
        raise ValueError("All bandwidths must be positive.")

    K = np.zeros_like(pairwise_l2_distances, dtype=float)
    for bandwidth in bandwidths:
        K += gaussian_kernel_matrix(pairwise_l2_distances, bandwidth)
    K /= len(bandwidths)
    return K


def mmd2_average_gaussian(
    X,
    Y,
    bandwidths=DEFAULT_FIXED_BANDWIDTHS,
    unbiased=False,
    return_kernel=False,
):
    """
    Compute MMD^2 between X and Y using a single kernel formed by averaging
    multiple fixed-bandwidth Gaussian kernels for measuring distribution
    dissimilarity.

    Parameters
    ----------
    X, Y : ndarray, shape (n_samples, n_features)
        Samples from the two windows/distributions to compare.
    bandwidths : ndarray
        Fixed Gaussian bandwidths. The default is
        DEFAULT_FIXED_BANDWIDTHS = [0.1, 0.3, 1.0, 3.0, 10.0].
    unbiased : bool
        If True, use the unbiased U-statistic estimator and exclude diagonals
        in Kxx and Kyy. This estimator can be negative for finite samples.
        If False, use the biased V-statistic estimator, which is nonnegative
        up to numerical precision and is better for a metric-like distance.
    return_kernel : bool
        If True, also return the averaged full kernel matrix and bandwidths.

    Returns
    -------
    mmd2 : float
        Estimated squared MMD. Small means similar distributions; large means
        different distributions.
    bandwidths : ndarray
        The fixed bandwidths used.
    K : ndarray, optional
        Full averaged kernel matrix over concat(X, Y), returned only when
        return_kernel=True.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must both be 2D arrays: samples x features.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features.")

    m = X.shape[0]
    n = Y.shape[0]

    if unbiased and (m < 2 or n < 2):
        raise ValueError("Unbiased MMD requires at least 2 samples in X and Y.")

    bandwidths = np.asarray(bandwidths, dtype=float)

    Z = np.concatenate((X, Y), axis=0)
    pairwise_l2 = np_distances(Z, Z, l="l2", matrix=True)
    K = average_gaussian_kernel_matrix(pairwise_l2, bandwidths)

    Kxx = K[:m, :m]
    Kyy = K[m:, m:]
    Kxy = K[:m, m:]

    if unbiased:
        mmd2 = (
            (np.sum(Kxx) - np.trace(Kxx)) / (m * (m - 1))
            + (np.sum(Kyy) - np.trace(Kyy)) / (n * (n - 1))
            - 2 * np.mean(Kxy)
        )
    else:
        mmd2 = np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)

    # Guard tiny negative values caused by floating-point roundoff.
    if not unbiased and mmd2 < 0 and np.isclose(mmd2, 0.0):
        mmd2 = 0.0

    if return_kernel:
        return float(mmd2), bandwidths.copy(), K
    return float(mmd2), bandwidths.copy()


def mmd_average_gaussian(*args, **kwargs):
    """
    Return sqrt(max(MMD^2, 0)) using fixed-bandwidth averaged Gaussian kernels.

    With the default biased estimator, this is a nonnegative empirical MMD
    distance. The kernel/bandwidth choice is fixed and not fit to the data.
    """
    result = mmd2_average_gaussian(*args, **kwargs)
    mmd2 = result[0]
    mmd = np.sqrt(max(mmd2, 0.0))
    return (mmd, *result[1:])
