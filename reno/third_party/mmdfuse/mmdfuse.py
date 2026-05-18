"""Third party tool for a value of information algorithm.

=============================================================================
mmdfuse.py - copy of the original MMD-FUSE implementation
=============================================================================
Copyright (c) 2023 Biggs, Schrab & Gretton
Licensed under the MIT License (see LICENSE file in this directory).

-------------------------------------------------------------------------
  Modifications made for the Reno repository
-------------------------------------------------------------------------
* Removed JAX dependency – all JAX‑specific imports and `jax.numpy`
calls have been replaced with NumPy equivalents.  The public API
(`mmdfuse(...)`) and the statistical behaviour remain unchanged.
* Updated type hints to use `numpy.ndarray` instead of `jax.numpy.ndarray`.
* Minor refactoring to avoid JAX random‑key handling.
* Update docstring style.

-------------------------------------------------------------------------
  Citation
-------------------------------------------------------------------------
If you use this code in a publication, please cite the original work:
    @article{biggs2023mmdfuse,
        author        = {Biggs, Felix and Schrab, Antonin and Gretton, Arthur},
        title         = {{MMD-FUSE}: {L}earning and Combining Kernels for Two-Sample Testing Without Data Splitting},
        year          = {2023},
        journal       = {Advances in Neural Information Processing Systems},
        volume        = {36}
    }

Repo link: https://github.com/antoninschrab/mmdfuse
=============================================================================
"""

import numpy as np


def _logsumexp(a, axis=None, b=1.0):
    """NumPy-only stable logsumexp.

    Equivalent to ``scipy.special.logsumexp(a, axis=axis, b=b)`` for positive scalar b.
    """
    a = np.asarray(a)
    a_max = np.max(a, axis=axis, keepdims=True)

    # Avoid invalid operations if all values are -inf
    shifted = np.exp(a - a_max)
    s = np.sum(b * shifted, axis=axis, keepdims=True)
    out = a_max + np.log(s)

    if axis is not None:
        out = np.squeeze(out, axis=axis)

    return out


def kernel_matrix(pairwise_matrix, l, kernel, bandwidth, rq_kernel_exponent=0.5):
    """Compute kernel matrix for a given kernel and bandwidth.

    Args:
        pairwise_matrix (ndarray): Matrix of pairwise distances.
        l (str): {"l1", "l2"} Distance type.
        kernel (str): Kernel name.
        bandwidth (float): Kernel bandwidth.
        rq_kernel_exponent (float): Exponent for rational quadratic kernel.

    Returns:
        Kernel matrix.
    """
    d = pairwise_matrix / bandwidth

    if kernel == "gaussian" and l == "l2":
        return np.exp(-(d**2) / 2)

    elif kernel == "laplace" and l == "l1":
        return np.exp(-d * np.sqrt(2))

    elif kernel == "rq" and l == "l2":
        return (1 + d**2 / (2 * rq_kernel_exponent)) ** (-rq_kernel_exponent)

    elif kernel == "imq" and l == "l2":
        return (1 + d**2) ** (-0.5)

    elif (kernel == "matern_0.5_l1" and l == "l1") or (
        kernel == "matern_0.5_l2" and l == "l2"
    ):
        return np.exp(-d)

    elif (kernel == "matern_1.5_l1" and l == "l1") or (
        kernel == "matern_1.5_l2" and l == "l2"
    ):
        return (1 + np.sqrt(3) * d) * np.exp(-np.sqrt(3) * d)

    elif (kernel == "matern_2.5_l1" and l == "l1") or (
        kernel == "matern_2.5_l2" and l == "l2"
    ):
        return (1 + np.sqrt(5) * d + 5 / 3 * d**2) * np.exp(-np.sqrt(5) * d)

    elif (kernel == "matern_3.5_l1" and l == "l1") or (
        kernel == "matern_3.5_l2" and l == "l2"
    ):
        return (
            1 + np.sqrt(7) * d + 2 * 7 / 5 * d**2 + 7 * np.sqrt(7) / 3 / 5 * d**3
        ) * np.exp(-np.sqrt(7) * d)

    elif (kernel == "matern_4.5_l1" and l == "l1") or (
        kernel == "matern_4.5_l2" and l == "l2"
    ):
        return (
            1
            + 3 * d
            + 3 * (6**2) / 28 * d**2
            + (6**3) / 84 * d**3
            + (6**4) / 1680 * d**4
        ) * np.exp(-3 * d)

    else:
        raise ValueError('The values of "l" and "kernel" are not valid.')


def np_distances(X, Y, l, max_samples=None, matrix=False):
    """NumPy replacement for jax_distances.

    Computes pairwise l1 or l2 distances using broadcasting.

    Args:
        X (ndarray): shape (m, d)
        Y (ndarray): shape (n, d)
        l (str): {"l1", "l2"} Distance type.
        max_samples (int): Maximum number of pairs to draw for computing distances.
        matrix (bool): Returns the full distance matrix if ``True``, otherwise just the
            upper-triangular entries.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

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
    else:
        return output[np.triu_indices(output.shape[0])]


def compute_bandwidths(X, Y, l, number_bandwidths, only_median=False):
    """NumPy replacement for the JAX/JIT compute_bandwidths function."""
    Z = np.concatenate((X, Y), axis=0)
    distances = np_distances(Z, Z, l, matrix=False)

    median = np.median(distances)

    if only_median:
        return median

    distances = distances + (distances == 0) * median
    dd = np.sort(distances)

    lambda_min = dd[int(np.floor(len(dd) * 0.05))] / 2
    lambda_max = dd[int(np.floor(len(dd) * 0.95))] * 2

    bandwidths = np.linspace(lambda_min, lambda_max, number_bandwidths)
    return bandwidths


def _make_rng(key=None):
    """Convert a key-like input into a NumPy random Generator.

    Args:
        key (int | np.random.Generator): The key or existing generator.

    Returns:
        np.random.Generator
    """
    if isinstance(key, np.random.Generator):
        return key
    return np.random.default_rng(key)


# NOTE: typehint for array-like is np.typing.ArrayLike
def mmdfuse(
    X,
    Y,
    key=None,
    alpha=0.05,
    kernels=("laplace", "gaussian"),
    lambda_multiplier=1,
    number_bandwidths=10,
    number_permutations=2000,
    return_p_val=False,
):
    """Two-Sample MMD-FUSE test, NumPy-only version.

    Args:
        X (array_like): shape (m, d)
        Y (array_like): shape (n, d)
        key (int | np.random.Generator) Random seed or NumPy Generator.
            Example: ``key=0``
        alpha (float): Test level.
        kernels (str | tuple[str] | list[str]): Kernel names.
        lambda_multiplier (float): ???
        number_bandwidths (int): ???
        number_permutations (int): ???
        return_p_val (bool): ???

    Returns:
        0 if the test fails to reject the null. 1 if the test rejects the null.
        Or, if return_p_val=True, returns a tuple of the int output, float p_val, and
        the numpy array containing all statistics.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    rng = _make_rng(key)

    # Match original behavior: ensure n <= m
    if Y.shape[0] > X.shape[0]:
        X, Y = Y, X

    m = X.shape[0]
    n = Y.shape[0]

    assert n <= m
    assert n >= 2 and m >= 2
    assert 0 < alpha < 1
    assert lambda_multiplier > 0
    assert number_bandwidths > 1 and type(number_bandwidths) is int
    assert number_permutations > 0 and type(number_permutations) is int

    if type(kernels) is str:
        kernels = (kernels,)

    valid_kernels = (
        "imq",
        "rq",
        "gaussian",
        "matern_0.5_l2",
        "matern_1.5_l2",
        "matern_2.5_l2",
        "matern_3.5_l2",
        "matern_4.5_l2",
        "laplace",
        "matern_0.5_l1",
        "matern_1.5_l1",
        "matern_2.5_l1",
        "matern_3.5_l1",
        "matern_4.5_l1",
    )

    for kernel in kernels:
        assert kernel in valid_kernels

    all_kernels_l1 = (
        "laplace",
        "matern_0.5_l1",
        "matern_1.5_l1",
        "matern_2.5_l1",
        "matern_3.5_l1",
        "matern_4.5_l1",
    )

    all_kernels_l2 = (
        "imq",
        "rq",
        "gaussian",
        "matern_0.5_l2",
        "matern_1.5_l2",
        "matern_2.5_l2",
        "matern_3.5_l2",
        "matern_4.5_l2",
    )

    number_kernels = len(kernels)
    kernels_l1 = [k for k in kernels if k in all_kernels_l1]
    kernels_l2 = [k for k in kernels if k in all_kernels_l2]

    # Setup for permutations
    B = number_permutations
    total = m + n

    # Shape: (B + 1, m + n)
    idx = np.empty((B + 1, total), dtype=int)
    for b in range(B + 1):
        idx[b] = rng.permutation(total)

    # 11
    v11 = np.concatenate((np.ones(m), -np.ones(n)))
    V11i = np.tile(v11, (B + 1, 1))
    V11 = np.take_along_axis(V11i, idx, axis=1)
    V11[B] = v11
    V11 = V11.T

    # 10
    v10 = np.concatenate((np.ones(m), np.zeros(n)))
    V10i = np.tile(v10, (B + 1, 1))
    V10 = np.take_along_axis(V10i, idx, axis=1)
    V10[B] = v10
    V10 = V10.T

    # 01
    v01 = np.concatenate((np.zeros(m), -np.ones(n)))
    V01i = np.tile(v01, (B + 1, 1))
    V01 = np.take_along_axis(V01i, idx, axis=1)
    V01[B] = v01
    V01 = V01.T

    # Compute all permuted MMD estimates
    N = number_bandwidths * number_kernels
    M = np.zeros((N, B + 1))

    kernel_count = -1

    Z = np.concatenate((X, Y), axis=0)

    for r in range(2):
        kernels_l = (kernels_l1, kernels_l2)[r]
        l = ("l1", "l2")[r]

        if len(kernels_l) > 0:
            # Pairwise distance matrix
            pairwise_matrix = np_distances(Z, Z, l, matrix=True)

            # Collection of bandwidths
            distances = pairwise_matrix[np.triu_indices(pairwise_matrix.shape[0])]

            median = np.median(distances)
            distances = distances + (distances == 0) * median

            dd = np.sort(distances)
            lambda_min = dd[int(np.floor(len(dd) * 0.05))] / 2
            lambda_max = dd[int(np.floor(len(dd) * 0.95))] * 2

            bandwidths = np.linspace(lambda_min, lambda_max, number_bandwidths)

            # Compute all permuted MMD estimates for either l1 or l2
            for kernel in kernels_l:
                kernel_count += 1

                for i in range(number_bandwidths):
                    bandwidth = bandwidths[i]

                    # Compute kernel matrix and set diagonal to zero
                    K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth)
                    np.fill_diagonal(K, 0)

                    # Compute standard deviation
                    unscaled_std = np.sqrt(np.sum(K**2))

                    # Matrix products
                    KV10 = K @ V10
                    KV01 = K @ V01
                    KV11 = K @ V11

                    values = (
                        np.sum(V10 * KV10, axis=0)
                        * (n - m + 1)
                        * (n - 1)
                        / (m * (m - 1))
                        + np.sum(V01 * KV01, axis=0) * (m - n + 1) / m
                        + np.sum(V11 * KV11, axis=0) * (n - 1) / m
                    )

                    values = values / unscaled_std * np.sqrt(n * (n - 1))

                    M[kernel_count * number_bandwidths + i] = values

    # Compute permuted and original statistics
    all_statistics = _logsumexp(lambda_multiplier * M, axis=0, b=1 / N)
    original_statistic = all_statistics[-1]

    # Compute p-value and test output
    p_val = np.mean(all_statistics >= original_statistic)
    output = int(p_val <= alpha)

    if return_p_val:
        return output, p_val, all_statistics
    else:
        return output
