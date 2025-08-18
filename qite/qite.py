"""QITE main functions."""

from functools import partial
from typing import Any
import numpy as np
from scipy.sparse.linalg import minres
import jax
import jax.numpy as jnp
from qiskit.quantum_info import SparsePauliOp

PAULI_IDX = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
PAULIS = jnp.array([
    [[1., 0.], [0., 1.]],
    [[0., 1.], [1., 0.]],
    [[0., -1.j], [1.j, 0.]],
    [[1., 0.], [0., -1.]]
])
PAULI_PROD_IDX = jnp.array([
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0]
])
PAULI_PROD_COEFF = jnp.array([
    [1., 1., 1., 1.],
    [1., 1., 1.j, -1.j],
    [1., -1.j, 1., 1.j],
    [1., 1.j, -1.j, 1.]
])
PAULI_INDICES = {}
PAULI_PRODUCTS = {}
PAULISTR_MATRICES = {}
MAX_CACHED_DOMSIZE = 7

TROTTERIZE_AMAT = True


def qite(
    hterms: list[SparsePauliOp],
    domains: list[tuple[int, ...]],
    initial_state: np.ndarray,
    delta_beta: float,
    num_steps: int,
    solver_params: dict[str, Any]
) -> tuple[np.ndarray, list[float]]:
    """Loop over Trotter steps and evolve the initial state."""
    state = initial_state
    norms = []
    for _ in range(num_steps):
        for hterm, domain in zip(hterms, domains):
            state, norm = qite_step(hterm, domain, state, delta_beta, solver_params)
            norms.append(norm)

    return state, norms


def qite_step(
    hterm: SparsePauliOp,
    domain: tuple[int, ...],
    state: np.ndarray,
    delta_beta: float,
    solver_params: dict[str, Any]
) -> tuple[np.ndarray, float]:
    """Single Trotter step on one Hamiltonian term."""
    dom_size = len(domain)
    if dom_size <= MAX_CACHED_DOMSIZE and dom_size not in PAULISTR_MATRICES:
        PAULISTR_MATRICES[dom_size] = vmake_paulistr_matrix(jnp.arange(4 ** dom_size), dom_size)

    # <ψ|σ_I
    sigma_psi_dag_arr = vcompute_sigma_psi(jnp.arange(4 ** dom_size), domain, state).conjugate()
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
        def trotter_update(isig, _state):
            return update_state_paulistr(isig, a_vec[isig], _state, domain, delta_beta)

        state = jax.lax.fori_loop(
            0, 4 ** dom_size,
            trotter_update,
            state
        )
    else:
        if dom_size <= MAX_CACHED_DOMSIZE:
            a_matrix = jnp.sum(a_vec[:, None, None] * PAULISTR_MATRICES[dom_size], axis=0)
        else:
            a_matrix = jax.lax.fori_loop(
                0, 4 ** dom_size,
                lambda ipauli, _mat: _mat + a_vec[ipauli] * make_paulistr_matrix(ipauli, dom_size),
                jnp.zeros((2 ** dom_size,) * 2, dtype=np.complex128)
            )
        state = update_state(a_matrix, state, domain, delta_beta)

    return state, sqrt_c


@partial(jax.jit, static_argnums=[1])
def unravel_pauli_indices(index: int | np.ndarray, num_qubits: int) -> np.ndarray:
    return (jnp.asarray(index)[..., None] // (4 ** jnp.arange(num_qubits)[::-1])) % 4


@jax.jit
def ravel_pauli_index(indices: np.ndarray) -> np.ndarray:
    return jnp.sum(indices * (4 ** jnp.arange(indices.shape[-1])[::-1]), axis=-1)


@partial(jax.jit, static_argnums=[1])
def make_paulistr_matrix(pstr_idx: int, num_qubits: int) -> np.ndarray:
    indices = unravel_pauli_indices(pstr_idx, num_qubits)
    args = []
    for ip, idx in enumerate(indices):
        args += [PAULIS[idx], [2 * ip, 2 * ip + 1]]
    args.append(list(range(0, 2 * num_qubits, 2)) + list(range(1, 2 * num_qubits + 1, 2)))
    return jnp.einsum(*args).reshape((2 ** num_qubits,) * 2)


vmake_paulistr_matrix = jax.jit(jax.vmap(make_paulistr_matrix, (0, None)), static_argnums=[1])


@partial(jax.jit, static_argnums=[2])
def lookup_pauli_prod(idx1: int, idx2: int, num_qubits: int) -> tuple[int, complex]:
    """Compute the Pauli string corresponding to the product of two strings."""
    idx1 = unravel_pauli_indices(idx1, num_qubits)
    idx2 = unravel_pauli_indices(idx2, num_qubits)
    index = ravel_pauli_index(PAULI_PROD_IDX[idx1, idx2])
    coeff = np.prod(PAULI_PROD_COEFF[idx1, idx2])
    return index, coeff


vlookup_pauli_prod = jax.vmap(jax.vmap(lookup_pauli_prod, (None, 0, None)), (0, None, None))
vlookup_pauli_prod = jax.jit(vlookup_pauli_prod, static_argnums=[2])


def generate_pauli_prod_table(num_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a matrix of Pauli product operators and coefficients."""
    if num_qubits not in PAULI_PRODUCTS:
        indices = jnp.arange(4 ** num_qubits)
        PAULI_PRODUCTS[num_qubits] = vlookup_pauli_prod(indices, indices, num_qubits)
    return PAULI_PRODUCTS[num_qubits]


@partial(jax.jit, static_argnums=[1])
def apply_on_domain(matrix: np.ndarray, domain: tuple[int, ...], state: np.ndarray) -> np.ndarray:
    """Apply the matrix onto the state in the specified domain."""
    dom_size = len(domain)
    matrix = matrix.reshape((2,) * (2 * dom_size))
    num_qubits = np.round(np.log2(state.shape[0])).astype(int)
    state = state.reshape((2,) * num_qubits)
    targ_axes = [num_qubits - 1 - iq for iq in domain]
    state = jnp.tensordot(matrix, state, [list(range(dom_size, 2 * dom_size)), targ_axes])
    state = jnp.moveaxis(state, list(range(dom_size)), targ_axes)
    return state.reshape(-1)


@partial(jax.jit, static_argnums=[1])
def compute_sigma_psi(
    idx: np.ndarray,
    domain: tuple[int, ...],
    state: np.ndarray
) -> np.ndarray:
    """Compute σ_I|ψ>."""
    pstr_mat = make_paulistr_matrix(idx, len(domain))
    return apply_on_domain(pstr_mat, domain, state)


vcompute_sigma_psi = jax.vmap(compute_sigma_psi, in_axes=(0, None, None))
vcompute_sigma_psi = jax.jit(vcompute_sigma_psi, static_argnums=[1])


def apply_hterm(hterm: SparsePauliOp, state: np.ndarray) -> np.ndarray:
    """Apply the Hamiltonian term to the state."""
    h_psi = jnp.zeros_like(state)
    for pauli, coeff in zip(hterm.paulis, hterm.coeffs):
        pstr = pauli.to_label()
        # Counting qubits from the right
        domain = tuple(iq for iq, p in enumerate(pstr[::-1]) if p != 'I')
        dom_size = len(domain)
        args = []
        for ip, iq in enumerate(domain[::-1]):
            pidx = PAULI_IDX[pstr[hterm.num_qubits - 1 - iq]]
            args += [PAULIS[pidx], [2 * ip, 2 * ip + 1]]
        args.append(list(range(0, 2 * dom_size, 2)) + list(range(1, 2 * dom_size + 1, 2)))
        hpauli = jnp.einsum(*args).reshape((2 ** dom_size,) * 2)
        h_psi += coeff * apply_on_domain(hpauli, domain, state)
    return h_psi


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
def update_state_paulistr(
    pstr_idx: int,
    coeff: float,
    state: np.ndarray,
    domain: tuple[int, ...],
    delta_t: float
):
    if len(domain) <= MAX_CACHED_DOMSIZE:
        pop = PAULISTR_MATRICES[len(domain)][pstr_idx]
    else:
        pop = make_paulistr_matrix(pstr_idx, len(domain))
    state = (jnp.cos(delta_t * coeff) * state
             + apply_on_domain(-1.j * jnp.sin(delta_t * coeff) * pop, domain, state))
    return state
