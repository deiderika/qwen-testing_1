"""Математическое ядро PFG: базис Чебышёва, гейты, нормализации.

Все функции чистые, NumPy-only (float32), без побочных эффектов —
подходят для инференса на Raspberry Pi без PyTorch.
"""
from __future__ import annotations

import numpy as np


def chebyshev_basis(x: np.ndarray, order: int) -> np.ndarray:
    """Возвращает массив [T_0(x), T_1(x), ..., T_order(x)] формой (*x.shape, order+1).

    T_0(x)=1, T_1(x)=x, T_k(x)=2x*T_{k-1}(x) - T_{k-2}(x).
    """
    x = np.asarray(x, dtype=np.float32)
    if order < 0:
        raise ValueError("order должен быть >= 0")
    out = np.empty(x.shape + (order + 1,), dtype=np.float32)
    out[..., 0] = 1.0
    if order >= 1:
        out[..., 1] = x
    for k in range(2, order + 1):
        out[..., k] = 2.0 * x * out[..., k - 1] - out[..., k - 2]
    return out


def edge_function(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Вычисляет f_e(x) = sum_k c_k * T_k(x).

    x: скаляр или массив входов (ожидается в [-1, 1], обычно tanh-выход);
    coeffs: коэффициенты базиса длины K+1.
    """
    x = np.asarray(x, dtype=np.float32)
    coeffs = np.asarray(coeffs, dtype=np.float32)
    order = coeffs.shape[0] - 1
    basis = chebyshev_basis(x, order)  # (*x.shape, K+1)
    return np.einsum("...k,k->...", basis, coeffs).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Численно стабильная сигмоида (не падает на больших |x|)."""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусная близость двух векторов; 0.0 если хотя бы один нулевой."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """LayerNorm по последней оси: (x - mean) / std."""
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)).astype(np.float32)


def edge_input(state: np.ndarray, projection: np.ndarray, bias: float) -> np.ndarray:
    """x_e = tanh(a_e^T h_j + b_e) — скалярный вход ребра из состояния источника."""
    state = np.asarray(state, dtype=np.float32)
    projection = np.asarray(projection, dtype=np.float32)
    z = float(np.dot(projection, state)) + float(bias)
    return np.float32(np.tanh(z))


def edge_gate(salience: float, query: np.ndarray, gate_key: np.ndarray,
              gate_bias: float, dim: int) -> np.ndarray:
    """g_e = sigmoid(S_e + q_i^T k_e / sqrt(d) + b_gate)."""
    query = np.asarray(query, dtype=np.float32)
    gate_key = np.asarray(gate_key, dtype=np.float32)
    s = float(salience) + float(np.dot(query, gate_key)) / float(np.sqrt(max(dim, 1))) + float(gate_bias)
    return sigmoid(np.float32(s))


def _fit_polynomial_desc(xs: np.ndarray, ys: np.ndarray, degree: int) -> tuple[np.ndarray, float]:
    """Полиномиальная аппроксимация. Возвращает (коэффициенты от старшей степени к младшей, средняя abs-ошибка).

    Используется при компиляции навыков и символической выжимке рёбер.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size == 0:
        return np.zeros(degree + 1), float("inf")
    deg = min(degree, max(0, xs.size - 1))
    coeffs = np.polyfit(xs, ys, deg)  # старшая степень первой
    pred = np.polyval(coeffs, xs)
    err = float(np.mean(np.abs(pred - ys))) if xs.size else float("inf")
    return coeffs.astype(np.float64), err
