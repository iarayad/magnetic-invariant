# +
from functools import reduce

import numpy as np
from scipy.optimize import minimize
import kwant
from pfapack.pfaffian import pfaffian

# +
def wilson_line_operator(vecs):
    """
    Parameters:
    -----------
    vecs: Occupied eigenvectors along a path

    Returns:
    -----------
    Wilson line operator
    """
    W = np.einsum("ijk,ihk->ijh", vecs, vecs.conj())  # projectors along line
    return vecs[-1].conj().T @ reduce(np.matmul, W[::-1]) @ vecs[0]


def dressed_wilson_line_det(S, vecs):
    """
    Parameters:
    -----------
    S: Unitary part of symmetry
    vecs: Occupied eigenvectors along a path

    Returns:
    -----------
    Gauge-fixed determinant of Wilson line
    """
    det_W = np.linalg.det(wilson_line_operator(vecs))
    w_sew_0, w_sew_1 = [
        vec.conj().T @ (S - S.T) / 2 @ vec.conj() for vec in (vecs[0], vecs[-1])
    ]
    return (1 / pfaffian(w_sew_1)) * det_W * pfaffian(w_sew_0)


# -

def ham_function(template, params):
    """
    Parameters:
    -----------
    template: Kwant Builder
    params: Dictionary with system parameters

    Returns:
    -----------
    ham: Bloch Hamiltonian at k_x, k_y
    """
    wrapped = kwant.wraparound.wraparound(template).finalized()

    def ham(k_x, k_y):
        return wrapped.hamiltonian_submatrix(
            params={**params, "k_x": k_x, "k_y": k_y}, sparse=False
        )

    return ham


def invariant_Z4(C4T_U, ham, N=200, opt_method="Powell", atol=1e-06):
    """
    C4T Invariant of two-dimensional magnetic insulators.

    Parameters:
    -----------
    C4T_U:  Unitary part of C4T
    ham: Hamiltonian function
    N: BZ discretization

    Returns:
    -----------
    nu: invariant
    """
    ks = np.linspace(-np.pi, np.pi, N + 1, endpoint=True)
    hamiltonians = np.array([[ham(kx, ky) for ky in ks] for kx in ks])
    n_occ = hamiltonians.shape[2] // 2

    # First store all eigendecompositions, as they are needed for wilson loop and for Wannier band spaces
    vals, vecs = np.linalg.eigh(hamiltonians)
    # Force edges of the BZ to the same gauge
    vecs[-1, :] = vecs[0, :]
    vecs[:, -1] = vecs[:, 0]

    gap_guess = np.argmin(vals[:, :, n_occ] - vals[:, :, n_occ - 1])
    gap = minimize(
        lambda k: np.diff(np.linalg.eigvalsh(ham(*k))[1:3]),
        np.array([ks[gap_guess // (N + 1)], ks[gap_guess % (N + 1)]]),
        method=opt_method,
    ).fun

    # Berry flux in EBZ:
    bc = berry_curvature(vecs[:-1, :-1, :, :n_occ])
    bc_eff = bc * EBZ_Z4_quarter(bc)
    berry_flux = np.sum(bc_eff)
    # Dressed wilson line determinant along Gamma->X->M:
    line = np.concatenate(
        (vecs[N // 2, N // 2 :, :, :n_occ], vecs[N // 2 + 1 :, -1, :, :n_occ])
    )
    det_W_dressed = dressed_wilson_line_det(C4T_U, line)

    # Invariant Z4 itself:
    nu = 2 * (((berry_flux + 2 * np.log(det_W_dressed).imag) / 2 / np.pi) % 2)
    if np.isclose(nu, 4, atol=atol):
        nu = 0
    return nu, gap

def berry_curvature(vectors):
    """Berry curvature of a system.

    Parameters:
    -----------
    vecs :  4D array with occupied eigenvectors in the 2D BZ grid

    Returns:
    --------
    bc : 2D array
        Berry curvature on each square in a `ks x ks` grid.
    """
    # The actual Berry curvature calculation
    vectors_x = np.roll(vectors, 1, 1)
    vectors_xy = np.roll(vectors_x, 1, 0)
    vectors_y = np.roll(vectors, 1, 0)

    shifted_vecs = [vectors, vectors_x, vectors_xy, vectors_y]

    v_shape = vectors.shape

    shifted_vecs = [i.reshape(-1, v_shape[-2], v_shape[-1]) for i in shifted_vecs]

    dets = np.ones(len(shifted_vecs[0]), dtype=complex)
    for vec, shifted in zip(shifted_vecs, np.roll(shifted_vecs, 1, 0)):
        dets *= [np.linalg.det(a.T.conj() @ b) for a, b in zip(vec, shifted)]
    bc = np.angle(dets).reshape(int(np.sqrt(len(dets))), -1)

    bc = (bc + np.pi / 2) % (np.pi) - np.pi / 2

    return bc


def EBZ_Z4_quarter(bz):
    """
    Template for IBZ over which to calculate Berry flux.
    """
    N = bz.shape[0]
    assert np.isclose(N % 2, 0)
    EBZ = np.kron(np.array([[0, 0], [1, 0]]), np.ones((N // 2, N // 2)))
    return np.roll(np.roll(EBZ, 1, 1), 1, 0)
