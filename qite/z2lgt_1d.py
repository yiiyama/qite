"""1D Z2 lattice gauge theory Hamiltonian."""
from typing import Optional
import numpy as np
from qiskit.quantum_info import SparsePauliOp


def hamiltonian(
    num_sites: int,
    mass: float,
    coupling: float,
    boundary_condition='open'
) -> SparsePauliOp:
    """Z2 lattice gauge theory Hamiltonian with unit-normalized field term.

    H = sum_{i in links} X_i + mass * sum_{i in sites} (-1)^i Z_i
        + coupling / 2 * sum_{i in sites} (X Z_i X + Y Z_i Y)
    """
    if boundary_condition == 'open':
        num_links = num_sites - 1
    elif boundary_condition == 'periodic':
        num_links = num_sites
    else:
        raise ValueError(f'Invalid boundary condition "{boundary_condition}"')

    num_qubits = num_sites + num_links

    # Field term
    paulis = ['I' * (num_qubits - 2 * isite - 2) + 'X' + 'I' * (2 * isite + 1)
              for isite in range(num_links)]
    coeffs = [1.] * num_links

    # Mass term
    paulis += ['I' * (num_qubits - 2 * isite - 1) + 'Z' + 'I' * (2 * isite)
               for isite in range(num_sites)]
    coeffs += [mass * (-1)**isite for isite in range(num_sites)]

    # Hopping term
    paulis += ['I' * (num_qubits - 2 * isite - 3) + 'XZX' + 'I' * (2 * isite)
               for isite in range(num_sites - 1)]
    if boundary_condition == 'periodic':
        paulis += ['ZX' + 'I' * (num_qubits - 3) + 'X']
    paulis += ['I' * (num_qubits - 2 * isite - 3) + 'YZY' + 'I' * (2 * isite)
               for isite in range(num_sites - 1)]
    if boundary_condition == 'periodic':
        paulis += ['ZY' + 'I' * (num_qubits - 3) + 'Y']
    coeffs += [coupling / 2.] * (num_links * 2)

    return SparsePauliOp(paulis, coeffs)


def domain_of(
    hterm: SparsePauliOp,
    domain_sizes: Optional[dict[int | str, int]] = None,
    flank_size: int = 1
) -> tuple[int, ...]:
    """Return the qubit domain that covers the support of the Hamiltonian term.

    Args:
        hterm: Single-term SparsePauliOp.
        domain_sizes: Map from Pauli string or Pauli weight to domain size. Overrides flank_size.
        flank_size: Number of extra qubits on each end of the domain.

    Returns:
        A tuple of qubit indices that specify the QITE domain of the Hamiltonian term.
    """
    pstr_full = hterm.paulis[0].to_label()
    # Counting qubits from the right
    pauli_positions = [iq for iq, p in enumerate(pstr_full[::-1]) if p != 'I']
    # For now we limit to contiguous Paulis
    if not np.all(np.abs(np.diff(pauli_positions)) == 1):
        raise ValueError('Non-contiguous Pauli op passed')

    pstr = ''.join(pstr_full[::-1][pos] for pos in pauli_positions[::-1])
    try:
        dom_size = domain_sizes[pstr]
    except KeyError:
        try:
            dom_size = domain_sizes[len(pstr)]
        except KeyError:
            dom_size = (max(pauli_positions) + flank_size) - (min(pauli_positions) - flank_size) + 1
    flank_size = (dom_size - len(pstr)) // 2
    low = max(min(pauli_positions) - flank_size, 0)
    high = min(low + dom_size, hterm.num_qubits)
    low = min(high - dom_size, low)
    return tuple(range(low, high))[::-1]
