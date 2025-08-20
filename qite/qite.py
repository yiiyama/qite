"""QITE main functions."""
from collections.abc import Sequence
from functools import partial
from typing import Any
import logging
import numpy as np
from scipy.sparse.linalg import minres
import jax
import jax.numpy as jnp
from .pauli import (PAULI, MAX_WEIGHT_CACHED, assert_pauli_matrices, fill_pauli_globals,
                    generate_pauli_prod_table, ravel_pauli_index, pauli_positions,
                    make_pauli_matrix, make_pauli_matrix_from_idx)

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
            indices = jnp.array([PAULI['idx'][op] for op in ops])
            pmat = make_pauli_matrix(indices)
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

    # Make sure Pauli matrices are generated for this weight (if small)
    if isinstance(hterm[0], str):
        weight = len(pauli_positions(hterm[0]))
    else:
        weight = len(hterm[1])
    assert_pauli_matrices(weight)

    dom_size = len(domain)
    assert_pauli_matrices(dom_size)
    # Sort domain to descending order
    domain = tuple(sorted(domain, reverse=True))

    # <ψ|σ_I
    if dom_size <= MAX_WEIGHT_CACHED:
        sigma_psi_dag_arr = vapply_on_domain(PAULI['matrices'][dom_size], domain, state).conjugate()
    else:
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
            def matrix_cumsum(ipauli, mat):
                return mat + a_vec[ipauli] * make_pauli_matrix_from_idx(ipauli, dom_size)

            a_matrix = jax.lax.fori_loop(
                0, 4 ** dom_size,
                matrix_cumsum,
                jnp.zeros((2 ** dom_size,) * 2, dtype=np.complex128)
            )
        state = update_state(a_matrix, state, domain, delta_beta)

    return state


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
    idx: int,
    domain: tuple[int, ...],
    state: np.ndarray
) -> np.ndarray:
    """Compute σ_I|ψ>."""
    return apply_on_domain(make_pauli_matrix_from_idx(idx, len(domain)), domain, state)


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
        indices = np.array([PAULI['idx'][p] for p in pauli])
        if (weight := len(pauli)) <= MAX_WEIGHT_CACHED:
            index = ravel_pauli_index(indices)
            matrix = PAULI['matrices'][weight][index]
        else:
            matrix = make_pauli_matrix(indices)
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
        pop = make_pauli_matrix_from_idx(ipauli, len(domain))
    return (jnp.cos(delta_t * coeff) * state
            + apply_on_domain(-1.j * jnp.sin(delta_t * coeff) * pop, domain, state))
