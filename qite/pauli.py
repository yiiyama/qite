"""Generic Pauli manipulation functions and globals definitions."""
from functools import partial
import jax
import jax.numpy as jnp

PAULI = {}
MAX_WEIGHT_CACHED = 7


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
def unravel_pauli_indices(index: int | jax.Array, num_qubits: int) -> jax.Array:
    return (jnp.asarray(index)[..., None] // (4 ** jnp.arange(num_qubits)[::-1])) % 4


@jax.jit
def ravel_pauli_index(indices: jax.Array) -> jax.Array:
    return jnp.sum(indices * (4 ** jnp.arange(indices.shape[-1])[::-1]), axis=-1)


def pauli_positions(pauli: str):
    """Return the positions of non-identity Paulis, counting qubits from the right end."""
    return tuple(iq for iq, p in enumerate(pauli[::-1]) if p != 'I')[::-1]


@jax.jit
def make_pauli_matrix(
    indices: int | jax.Array
) -> jax.Array:
    num_qubits = len(indices)
    args = []
    for ip, idx in enumerate(indices):
        args += [PAULI['basis'][idx], [2 * ip, 2 * ip + 1]]
    args.append(list(range(0, 2 * num_qubits, 2)) + list(range(1, 2 * num_qubits + 1, 2)))
    return jnp.einsum(*args).reshape((2 ** num_qubits,) * 2)


@partial(jax.jit, static_argnums=[1])
def make_pauli_matrix_from_idx(
    idx: int,
    num_qubits: int
) -> jax.Array:
    return make_pauli_matrix(unravel_pauli_indices(idx, num_qubits))


vmake_pauli_matrix_from_idx = jax.jit(jax.vmap(make_pauli_matrix_from_idx, (0, None)),
                                      static_argnums=[1])


def assert_pauli_matrices(dom_size):
    """Expand and cache the full Pauli matrices for small domain sizes."""
    if dom_size <= MAX_WEIGHT_CACHED and dom_size not in PAULI['matrices']:
        PAULI['matrices'][dom_size] = vmake_pauli_matrix_from_idx(jnp.arange(4 ** dom_size),
                                                                  dom_size)


@partial(jax.jit, static_argnums=[2])
def lookup_pauli_prod(idx1: int, idx2: int, num_qubits: int) -> tuple[int, complex]:
    """Compute the Pauli string corresponding to the product of two strings."""
    idx1 = unravel_pauli_indices(idx1, num_qubits)
    idx2 = unravel_pauli_indices(idx2, num_qubits)
    index = ravel_pauli_index(PAULI['prod_idx'][idx1, idx2])
    coeff = jnp.prod(PAULI['prod_coeff'][idx1, idx2])
    return index, coeff


_pauli_prod_table = jax.vmap(jax.vmap(lookup_pauli_prod, (None, 0, None)), (0, None, None))
_pauli_prod_table = jax.jit(_pauli_prod_table, static_argnums=[2])


def generate_pauli_prod_table(num_qubits: int) -> tuple[jax.Array, jax.Array]:
    """Generate a matrix of Pauli product operators and coefficients."""
    if num_qubits not in PAULI['prod_table']:
        indices = jnp.arange(4 ** num_qubits)
        PAULI['prod_table'][num_qubits] = _pauli_prod_table(indices, indices, num_qubits)
    return PAULI['prod_table'][num_qubits]
