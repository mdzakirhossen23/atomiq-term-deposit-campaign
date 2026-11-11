"""Small exact statevector backend. Bit strings are printed q0 first."""
import numpy as np

def bits(n):
    return ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(int)

def rx_mixer(state, beta, n):
    indices = np.arange(1 << n)
    for q in range(n):
        state = np.cos(beta) * state - 1j * np.sin(beta) * state[indices ^ (1 << q)]
    return state

def qaoa(cost, depth, gamma, beta, n):
    state = np.full(1 << n, 2 ** (-n / 2), dtype=complex)
    for layer in range(depth):
        state = rx_mixer(state * np.exp(-1j * gamma[layer] * cost), beta[layer], n)
    return np.abs(state) ** 2

def features(x):
    """RY data encoding followed by a ring of data-dependent RZZ gates."""
    n = x.shape[1]
    signs = 1 - 2 * bits(n)
    angles = np.arctan(x)
    half = angles[:, None, :] / 2
    amplitudes = np.prod(np.where(bits(n)[None, :, :], np.sin(half), np.cos(half)), axis=2)
    parity = signs * np.roll(signs, -1, axis=1)
    phases = (angles * np.roll(angles, -1, axis=1)) @ parity.T
    return amplitudes * np.exp(-0.5j * phases)

def circuit_cost_metrics(cost, n, layers):
    """Synthesize the diagonal using Walsh parities and CX-RZ-CX ladders."""
    parity = 1 - 2 * ((bits(n) @ bits(n).T) % 2)
    coefficients = parity @ cost / len(cost)
    last_layer = np.ones(n, dtype=int)
    cx = rz = 0

    def schedule(qubits):
        last_layer[list(qubits)] = max(last_layer[list(qubits)]) + 1

    for _ in range(layers):
        for mask, coefficient in enumerate(coefficients[1:], 1):
            if abs(coefficient) < 1e-12:
                continue
            *controls, target = np.flatnonzero(bits(n)[mask])
            for q in controls:
                schedule((q, target))
            schedule((target,))
            for q in reversed(controls):
                schedule((q, target))
            cx += 2 * len(controls)
            rz += 1
        for q in range(n):
            schedule((q,))
    return dict(logical_qubits=n, hardware_qubits_used=0, logical_depth=1+2*layers,
                decomposed_depth=int(max(last_layer)), one_qubit_gates=n*(1+layers)+rz,
                two_qubit_gates=cx, swap_gates=0, mapping="Identity; ideal all-to-all simulator")

