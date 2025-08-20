"""QITE main functions."""
from collections.abc import Sequence
from functools import partial
from typing import Any
import logging
import numpy as np
from scipy.sparse.linalg import minres
import jax
import jax.numpy as jnp

PAULI = {}
MAX_WEIGHT_CACHED = 7
TROTTERIZE_AMAT = True

LOG = logging.getLogger(__name__)


def qite(
    paulis: Sequence[str],
    coeffs: Sequence[float],
    domains: Sequence[tuple[int, ...]],
    initial_state: np.ndarray,
    delta_beta: float,
    num_steps: int,
    solver_params: dict[str, Any],
    return_energies: bool = True
) -> np.ndarray | tuple[np.ndarray, list[float]]:
    """Loop over Trotter steps and evolve the initial state."""
    if len(paulis) != len(coeffs) or len(paulis) != len(domains):
        raise ValueError('Lengths of paulis, coeffs, and domains must match')

    if not PAULI:
        fill_pauli_globals()

    for dom_size in set(len(dom) for dom in domains):
        assert_pauli_matrices(dom_size)

    positions_list = [pauli_positions(p) for p in paulis]
    if max(len(positions) for positions in positions_list) <= MAX_WEIGHT_CACHED:
        # Convert strings + coeffs -> matrices + positions
        hterms = []
        for pauli, coeff, positions in zip(paulis, coeffs, positions_list):
            ops = [pauli[::-1][pos].upper() for pos in positions]
            pmat = make_pauli_matrix(ops)
            hterms.append((pmat * coeff, positions))
    else:
        hterms = (paulis, coeffs)

    # Start imaginary time evolution
    state = initial_state
    energies = []
    for istep in range(num_steps):
        LOG.info('QITE step %d', istep)
        for hterm, domain in zip(hterms, domains):
            state = qite_step(hterm, domain, state, delta_beta, solver_params)

        if return_energies:
            energy = 0.
            for hterm in hterms:
                energy += (state.conjugate() @ apply_hterm(hterm, state)).real

            energies.append(energy)

    if return_energies:
        return state, energies
    else:
        return state


def qite_step(
    hterm: tuple[str, float] | tuple[np.ndarray, tuple[int, ...]],
    domain: tuple[int, ...],
    state: np.ndarray,
    delta_beta: float,
    solver_params: dict[str, Any]
) -> np.ndarray:
    """Single Trotter step on one Hamiltonian term."""
    if not PAULI:
        fill_pauli_globals()

    dom_size = len(domain)
    assert_pauli_matrices(dom_size)
    # Sort domain to descending order
    domain = tuple(sorted(domain, reverse=True))

    # <ψ|σ_I
    if dom_size <= MAX_WEIGHT_CACHED:
        sigma_psi_dag_arr = vapply_on_domain(PAULI['matrices'][dom_size], domain, state)
    else:
        sigma_psi_dag_arr = vcompute_sigma_psi(jnp.arange(4 ** dom_size), domain, state)

    # h|ψ>
    h_psi = apply_hterm(hterm, state)
    # √c
    sqrt_c = np.sqrt((1. - 2. * delta_beta * state.conjugate() @ h_psi
                      + 2. * (delta_beta ** 2) * h_psi.conjugate() @ h_psi).real)
    # S matrix
    prod_indices, prod_coeffs = generate_pauli_prod_table(dom_size)
    sigma_expvals = sigma_psi_dag_arr @ state
    s_matrix = sigma_expvals[prod_indices] * prod_coeffs
    # S + ST
    problem = (s_matrix + s_matrix.T).real
    # b
    b_vector = 2. * (sigma_psi_dag_arr @ h_psi).imag / sqrt_c
    # Solve the linear system of equations
    a_vec = solve_linear_system(problem, b_vector, solver_params)

    if TROTTERIZE_AMAT:
        def trotter_update(ipauli, _state):
            return update_state_by_pauli(ipauli, a_vec[ipauli], _state, domain, delta_beta)

        state = jax.lax.fori_loop(
            0, 4 ** dom_size,
            trotter_update,
            state
        )
    else:
        if dom_size <= MAX_WEIGHT_CACHED:
            a_matrix = jnp.sum(a_vec[:, None, None] * PAULI['matrices'][dom_size], axis=0)
        else:
            a_matrix = jax.lax.fori_loop(
                0, 4 ** dom_size,
                lambda ipauli, mat: mat + a_vec[ipauli] * make_pauli_matrix(ipauli, dom_size),
                jnp.zeros((2 ** dom_size,) * 2, dtype=np.complex128)
            )
        state = update_state(a_matrix, state, domain, delta_beta)

    return state


def fill_pauli_globals():
    """Helper function to avoid creating JAX arrays at the module level."""
    PAULI['idx'] = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    PAULI['basis'] = jnp.array([
        [[1., 0.], [0., 1.]],
        [[0., 1.], [1., 0.]],
        [[0., -1.j], [1.j, 0.]],
        [[1., 0.], [0., -1.]]
    ])
    PAULI['prod_idx'] = jnp.array([
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ])
    PAULI['prod_coeff'] = jnp.array([
        [1., 1., 1., 1.],
        [1., 1., 1.j, -1.j],
        [1., -1.j, 1., 1.j],
        [1., 1.j, -1.j, 1.]
    ])
    PAULI['prod_table'] = {}
    PAULI['matrices'] = {}


@partial(jax.jit, static_argnums=[1])
def unravel_pauli_indices(index: int | np.ndarray, num_qubits: int) -> np.ndarray:
    return (jnp.asarray(index)[..., None] // (4 ** jnp.arange(num_qubits)[::-1])) % 4


@jax.jit
def ravel_pauli_index(indices: np.ndarray) -> np.ndarray:
    return jnp.sum(indices * (4 ** jnp.arange(indices.shape[-1])[::-1]), axis=-1)


def pauli_positions(pauli: str):
    """Return the positions of non-identity Paulis, counting qubits from the right end."""
    return tuple(iq for iq, p in enumerate(pauli[::-1]) if p != 'I')[::-1]


@partial(jax.jit, static_argnums=[1])
def make_pauli_matrix(
    indices: int | np.ndarray | str,
    num_qubits: int = None
) -> np.ndarray:
    if isinstance(indices, int):
        indices = unravel_pauli_indices(indices, num_qubits)
    elif isinstance(indices, str):
        indices = np.ndarray([PAULI['idx'][op] for op in indices])

    num_qubits = len(indices)

    args = []
    for ip, idx in enumerate(indices):
        args += [PAULI['basis'][idx], [2 * ip, 2 * ip + 1]]
    args.append(list(range(0, 2 * num_qubits, 2)) + list(range(1, 2 * num_qubits + 1, 2)))
    return jnp.einsum(*args).reshape((2 ** num_qubits,) * 2)


vmake_pauli_matrix = jax.jit(jax.vmap(make_pauli_matrix, (0, None)), static_argnums=[1])


def assert_pauli_matrices(dom_size):
    """Expand and cache the full Pauli matrices for small domain sizes."""
    if dom_size <= MAX_WEIGHT_CACHED and dom_size not in PAULI['matrices']:
        PAULI['matrices'][dom_size] = vmake_pauli_matrix(jnp.arange(4 ** dom_size), dom_size)


@partial(jax.jit, static_argnums=[2])
def lookup_pauli_prod(idx1: int, idx2: int, num_qubits: int) -> tuple[int, complex]:
    """Compute the Pauli string corresponding to the product of two strings."""
    idx1 = unravel_pauli_indices(idx1, num_qubits)
    idx2 = unravel_pauli_indices(idx2, num_qubits)
    index = ravel_pauli_index(PAULI['prod_idx'][idx1, idx2])
    coeff = np.prod(PAULI['prod_coeff'][idx1, idx2])
    return index, coeff


_pauli_prod_table = jax.vmap(jax.vmap(lookup_pauli_prod, (None, 0, None)), (0, None, None))
_pauli_prod_table = jax.jit(_pauli_prod_table, static_argnums=[2])


def generate_pauli_prod_table(num_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a matrix of Pauli product operators and coefficients."""
    if num_qubits not in PAULI['prod_table']:
        indices = jnp.arange(4 ** num_qubits)
        PAULI['prod_table'][num_qubits] = _pauli_prod_table(indices, indices, num_qubits)
    return PAULI['prod_table'][num_qubits]


@partial(jax.jit, static_argnums=[1])
def apply_on_domain(matrix: np.ndarray, domain: tuple[int, ...], state: np.ndarray) -> np.ndarray:
    """Apply the matrix onto the state in the specified domain."""
    dom_size = len(domain)
    matrix = matrix.reshape((2,) * (2 * dom_size))
    num_qubits = np.round(np.log2(state.shape[0])).astype(int)
    state = state.reshape((2,) * num_qubits)
    targ_axes = [-1 - iq for iq in domain]
    state = jnp.tensordot(matrix, state, [list(range(dom_size, 2 * dom_size)), targ_axes])
    state = jnp.moveaxis(state, list(range(dom_size)), targ_axes)
    return state.reshape(-1)


vapply_on_domain = jax.jit(jax.vmap(apply_on_domain, (0, None, None)), static_argnums=[1])


@partial(jax.jit, static_argnums=[1])
def compute_sigma_psi(
    idx: np.ndarray,
    domain: tuple[int, ...],
    state: np.ndarray
) -> np.ndarray:
    """Compute σ_I|ψ>."""
    return apply_on_domain(make_pauli_matrix(idx, len(domain)), domain, state)


vcompute_sigma_psi = jax.jit(jax.vmap(compute_sigma_psi, in_axes=(0, None, None)),
                             static_argnums=[1])


def apply_hterm(
    hterm: tuple[str, float] | tuple[np.ndarray, tuple[int, ...]],
    state: np.ndarray
) -> np.ndarray:
    """Apply the Hamiltonian term to the state."""
    if isinstance(hterm[0], str):
        positions = pauli_positions(hterm[0])
        pauli = ''.join(hterm[0][::-1][pos] for pos in positions)
        if (weight := len(pauli)) <= MAX_WEIGHT_CACHED:
            index = ravel_pauli_index(np.array([PAULI['idx'][p] for p in pauli]))
            matrix = PAULI['matrices'][weight][index]
        else:
            matrix = make_pauli_matrix(pauli)
        matrix *= hterm[1]
    else:
        matrix, positions = hterm

    return apply_on_domain(matrix, positions, state)


def solve_linear_system(
    matrix: np.ndarray,
    vector: np.ndarray,
    solver_params: dict[str, Any]
) -> np.ndarray:
    """Solve a linear system of equations."""
    match solver_params['solver']:
        case 'minres':
            return solve_linear_system_minres(matrix, vector, solver_params)
        case _:
            raise ValueError(f'Invalid solver name {solver_params["solver"]}')


def solve_linear_system_minres(
    matrix: np.ndarray,
    vector: np.ndarray,
    solver_params: dict[str, Any]
) -> np.ndarray:
    """Solve a linear system of equations using scipy.sparse.linalg.minres."""
    solution, info = minres(np.asarray(matrix), np.asarray(vector), rtol=solver_params['rtol'])
    if info != 0:
        raise RuntimeError(f'minres return value {info}')
    return jnp.array(solution)


@partial(jax.jit, static_argnums=[2])
def update_state(hmat, state, domain, delta_t):
    """Evolve the state in real time by a hamiltonian matrix applied onto the domain."""
    evolution = jax.scipy.linalg.expm(-1.j * delta_t * hmat)
    return apply_on_domain(evolution, domain, state)


@partial(jax.jit, static_argnums=[3])
def update_state_by_pauli(
    ipauli: int,
    coeff: float,
    state: np.ndarray,
    domain: tuple[int, ...],
    delta_t: float
):
    if len(domain) <= MAX_WEIGHT_CACHED:
        pop = PAULI['matrices'][len(domain)][ipauli]
    else:
        pop = make_pauli_matrix(ipauli, len(domain))
    state = (jnp.cos(delta_t * coeff) * state
             + apply_on_domain(-1.j * jnp.sin(delta_t * coeff) * pop, domain, state))
    return state
