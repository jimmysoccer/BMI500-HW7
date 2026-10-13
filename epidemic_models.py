"""
epidemic_models.py
==================
BMI 500, Lecture 7 (Fall 2026) -- Homework 1
SIR and SEIR compartmental models for pandemic spread.

This module contains:
  * the two right-hand-side (RHS) functions of the SIR and SEIR models,
  * two numerical integrators written from scratch (forward Euler and the
    classical 4th-order Runge-Kutta method).  Per the assignment, NO built-in
    ODE solver is used to produce results; ``scipy.integrate.solve_ivp`` is
    imported only in partA for an *independent verification* of our solver.

Author: Heng Sun (Jimmy)
Email:  YOUR_EMORY_EMAIL@emory.edu   <- TODO: replace with your real Emory email
Question: HW 1
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sir_rhs",
    "seir_rhs",
    "forward_euler",
    "rk4",
    "integrate",
    "simulate_sir",
    "simulate_seir",
]


# --------------------------------------------------------------------------
# 1. Model equations (the ODE right-hand sides)
# --------------------------------------------------------------------------
def sir_rhs(t, y, beta, gamma, N):
    """SIR model with *counts* (not fractions).

        dS/dt = -beta * S * I
        dI/dt =  beta * S * I - gamma * I
        dR/dt =  gamma * I

    Parameters
    ----------
    t : float          current time (days); the RHS is autonomous, kept for API
    y : array_like     state [S, I, R] (numbers of people)
    beta : float       transmission rate (per person per day)
    gamma : float      recovery rate (per day)
    N : float          total population (S + I + R). Unused in the RHS but kept
                       so that both models share the same calling signature.

    Returns
    -------
    numpy.ndarray of shape (3,)
    """
    S, I, R = y
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    return np.array([dS, dI, dR], dtype=float)


def seir_rhs(t, y, beta, gamma, sigma, mu, N):
    """SEIR model with births and deaths (counts).

        dS/dt = mu*N - beta*S*I - mu*S
        dE/dt = beta*S*I - (sigma + mu)*E
        dI/dt = sigma*E - (gamma + mu)*I
        dR/dt = gamma*I - mu*R

    sigma : rate of becoming infectious (1/sigma = mean latent period)
    mu    : per-capita birth rate = per-capita death rate (constant population)
    """
    S, E, I, R = y
    dS = mu * N - beta * S * I - mu * S
    dE = beta * S * I - (sigma + mu) * E
    dI = sigma * E - (gamma + mu) * I
    dR = gamma * I - mu * R
    return np.array([dS, dE, dI, dR], dtype=float)


# --------------------------------------------------------------------------
# 2. Numerical integrators (implemented from scratch)
# --------------------------------------------------------------------------
def _as_y0(y0):
    return np.asarray(y0, dtype=float).ravel()


def _time_grid(t0, t1, dt):
    """Uniform time grid from t0 to t1 inclusive (dt scaled so the end lands
    exactly on t1)."""
    n = int(round((t1 - t0) / dt))
    dt = (t1 - t0) / n
    return np.linspace(t0, t1, n + 1), dt


def forward_euler(rhs, y0, t0, t1, dt, **params):
    """Explicit (forward) Euler integration.

    y_{k+1} = y_k + dt * f(t_k, y_k)

    First-order accurate; used here mostly as the simple baseline solver.
    """
    t, dt = _time_grid(float(t0), float(t1), float(dt))
    y = _as_y0(y0)
    Y = np.empty((t.size, y.size), dtype=float)
    Y[0] = y
    for k in range(t.size - 1):
        y = y + dt * rhs(t[k], y, **params)
        Y[k + 1] = y
    return t, Y


def rk4(rhs, y0, t0, t1, dt, **params):
    """Classical 4th-order Runge-Kutta integration.

    k1 = f(t_k,       y_k)
    k2 = f(t_k + dt/2, y_k + dt/2 * k1)
    k3 = f(t_k + dt/2, y_k + dt/2 * k2)
    k4 = f(t_k + dt,   y_k + dt   * k3)
    y_{k+1} = y_k + dt/6 * (k1 + 2 k2 + 2 k3 + k4)
    """
    t, dt = _time_grid(float(t0), float(t1), float(dt))
    y = _as_y0(y0)
    Y = np.empty((t.size, y.size), dtype=float)
    Y[0] = y
    for k in range(t.size - 1):
        tk = t[k]
        k1 = rhs(tk, y, **params)
        k2 = rhs(tk + dt / 2, y + dt / 2 * k1, **params)
        k3 = rhs(tk + dt / 2, y + dt / 2 * k2, **params)
        k4 = rhs(tk + dt, y + dt * k3, **params)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[k + 1] = y
    return t, Y


def integrate(rhs, y0, t0, t1, dt, method="rk4", **params):
    """Dispatch to one of the two self-implemented integrators."""
    if method == "euler":
        return forward_euler(rhs, y0, t0, t1, dt, **params)
    if method == "rk4":
        return rk4(rhs, y0, t0, t1, dt, **params)
    raise ValueError(f"unknown method {method!r}; use 'euler' or 'rk4'")


# --------------------------------------------------------------------------
# 3. Convenience wrappers for the two models
# --------------------------------------------------------------------------
def simulate_sir(N, S0, I0, R0_, beta, gamma, t_end, dt=0.01, method="rk4"):
    """Simulate the SIR model. Returns (t, S, I, R)."""
    t, Y = integrate(
        sir_rhs, [S0, I0, R0_], 0.0, t_end, dt,
        method=method, beta=beta, gamma=gamma, N=N,
    )
    return t, Y[:, 0], Y[:, 1], Y[:, 2]


def simulate_seir(N, S0, E0, I0, R0_, beta, gamma, sigma, mu,
                  t_end, dt=0.01, method="rk4"):
    """Simulate the SEIR model with births/deaths. Returns (t, S, E, I, R)."""
    t, Y = integrate(
        seir_rhs, [S0, E0, I0, R0_], 0.0, t_end, dt,
        method=method, beta=beta, gamma=gamma, sigma=sigma, mu=mu, N=N,
    )
    return t, Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]
