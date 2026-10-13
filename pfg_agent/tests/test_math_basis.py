"""Тесты математического ядра: Чебышёв, сигмоида, косинус, LayerNorm."""
import numpy as np
import pytest

from pfg.math_basis import (
    chebyshev_basis, cosine_similarity, edge_function, edge_gate,
    edge_input, layer_norm, sigmoid,
)
from pfg.math_basis import _fit_polynomial_desc as fit_polynomial


def test_chebyshev_values():
    # T_0=1, T_1=x, T_2=2x^2-1, T_3=4x^3-3x
    x = np.array([0.0, 0.5, -1.0], dtype=np.float32)
    basis = chebyshev_basis(x, 3)
    assert basis.shape == (3, 4)
    assert np.allclose(basis[:, 0], 1.0)
    assert np.allclose(basis[:, 1], x)
    assert np.allclose(basis[:, 2], 2 * x**2 - 1, atol=1e-6)
    assert np.allclose(basis[:, 3], 4 * x**3 - 3 * x, atol=1e-6)


def test_edge_function_linear():
    coeffs = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # f(x)=x
    x = np.float32(0.3)
    assert abs(float(edge_function(x, coeffs)) - 0.3) < 1e-5


def test_sigmoid_stability():
    big = np.array([1000.0, -1000.0, 0.0], dtype=np.float32)
    out = sigmoid(big)
    assert np.all(np.isfinite(out))
    assert out[0] > 0.99 and out[1] < 0.01 and abs(float(out[2]) - 0.5) < 1e-6


def test_cosine_similarity():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6
    assert abs(cosine_similarity(a, c)) < 1e-6
    assert cosine_similarity(a, np.zeros(2)) == 0.0


def test_layer_norm():
    x = np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32)
    y = layer_norm(x)
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(y.std(axis=-1), 1.0, atol=1e-2)


def test_edge_input_and_gate():
    state = np.ones(8, dtype=np.float32)
    proj = np.ones(8, dtype=np.float32) * 0.1
    x = edge_input(state, proj, 0.0)
    assert -1.0 <= float(x) <= 1.0
    g = edge_gate(0.5, state, proj, 0.0, 8)
    assert 0.0 < float(g) < 1.0


def test_fit_polynomial_recovers_line():
    xs = np.linspace(-1, 1, 30)
    ys = 2.0 * xs + 1.0
    coeffs, err = fit_polynomial(xs, ys, 1)
    assert err < 1e-6
    # коэффициенты возвращаются от старшей степени к младшей: [slope, intercept]
    assert abs(coeffs[0] - 2.0) < 1e-3 and abs(coeffs[-1] - 1.0) < 1e-3


def test_chebyshev_negative_order_raises():
    with pytest.raises(ValueError):
        chebyshev_basis(np.zeros(3), -1)
