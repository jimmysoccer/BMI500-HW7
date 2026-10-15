# BMI500-HW7

- **Name:** Heng Sun (Jimmy)
- **Email:** heng.sun@emory.edu
- **Question:** HW 1 — SIR and SEIR Model Implementation for Pandemic Spread

---

## 1. How to run

```bash
# 1. create an environment
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt

# 2. run the notebooks in order A -> E
jupyter notebook            # then open partA/…, partB/…, and run all cells
```

Each notebook is self-contained: it locates the repository root automatically, imports the shared
solver from `epidemic_models.py`, and writes its figure(s) into `figures/`. The notebooks are
committed **with their outputs**, so the results can also be read without re-running anything.

---

## 2. Part A — Implementation and verification

The two models and the two integrators are located in `epidemic_models.py`. The integrators are classical forward Euler and classical

4th-order Runge–Kutta.

```Python
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
```

---

## 3. Part B — SIR simulation

![Part B SIR curves](figures/partB_sir_timecourse.png)

---

## 4. Part C — Analysis and interpretation

**(i) Infection peak.** $I(t)$ reaches its maximum of about **469 people on day 42.5**. The peak is
set by two competing effects: the supply of new infections, proportional to $\beta S I$, and the
recovered people, proportional to $\gamma I$. They balance exactly when $S=\gamma/\beta$.
Because $\gamma/\beta = 800$ is far below $S(0)=1995$, the epidemic has a long runway and a large peak.
A larger $\beta$ raises the peak (and brings it earlier), while a larger $\gamma$ both lowers the peak
and shortens the outbreak. The height of the peak also depends on how long the system lingers near
$S=\gamma/\beta$, which is why $S(0)$ and $N$ matter too.

**(ii) Basic reproduction number.** $R_0=\beta N/\gamma = (0.125\times10^{-3})(2000)/0.1 = 2.5$.
This means that, in a fully susceptible population, each infectious person generates on average 2.5
secondary infections. Every primary case therefore replaces itself more than once, so the outbreak
grows. $R_0>1$ is the condition for an epidemic to take off, and control measures must push $R_0$ below 1
(by reducing $\beta$, e.g. distancing/masks, or raising $\gamma$, e.g. faster detection and treatment),
and the _higher_ $R_0$ is, the larger the fraction of the population that must be immunized,
roughly $1-1/R_0 = 60\%$ here, to reach herd immunity.

**(iii) Epidemic dynamics.** In the early phase $I$ grows nearly exponentially because $S\approx N$
so $dI/dt\approx(\beta N-\gamma)I$, i.e. growth rate $\gamma(R_0-1)$. As $S$ falls, the growth rate
decreases, and the epidemic peaks at $S=\gamma/\beta$. Afterwards $S$ keeps falling (more slowly) and
$R$ keeps rising until $I\to 0$ and the outbreak dies out, leaving a positive susceptible pool:
**about 214 people (10.7%) are never infected**, so the attack rate is about 89.3% ($R(160)=1786$).
The reason the outbreak stops before everyone is infected is that recovered people no longer transmit and the effective
reproduction number $R_{\text{eff}}=R_0 S/N$ falls below 1 once $S<N/R_0 = 800$.

---

## 5. Part D — SEIR model with births and deaths

SEIR Model Implementation

```Python
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
```

![Part D SEIR](figures/partD_seir_365_and_1500.png)

**(iii) Waves.** With births and deaths the outbreak is **not** a single event. The first wave peaks
at **day 61.6 with about 317 infectious people**. $I$ then falls to a trough of ~15 people around
day 197, and when the birth term has refilled the susceptible pool the infection breaks out again.
The next peaks of about **85 people (day 359)**, **60 people (day 648)** and **55 people (day 938)**,
separated by roughly **290 days**. So there is one large first wave followed by smaller, damped
recurrent waves. The 365-day window shows only the first wave and the start of the second, whereas
1500 days reveals the recurrent pattern.

**(iv) $R_0$ comparison.** For the SEIR model
$R_0=\frac{\beta N\sigma}{(\sigma+\mu)(\gamma+\mu)}=\frac{(0.125\times10^{-3})(2000)(0.25)}{(0.255)(0.105)}=2.33$,
slightly **below** the SIR value of 2.50. Reasons behind it is that the latent period means an exposed
person is not yet infectious (a factor $\sigma/(\sigma+\mu)$), and the death term removes people from
each compartment (the $(\gamma+\mu)$ and $(\sigma+\mu)$ factors). Numerically, the mean susceptible
level over the last 200 days of the 1500-day run is **857**, in excellent agreement with
$N/R_0=2000/2.33=857$. This number is the **endemic susceptible level**: with births continuously
adding new susceptibles, the population settles into a state where $S=N/R_0$ exactly, so that the
effective reproduction number $R_{\text{eff}}=R_0 S/N=1$ and each infection is replaced by precisely
one new infection — the infection persists indefinitely instead of dying out.

**(v) Effect of the exposed compartment and of births/deaths.** With the same transmission parameters the SEIR peak (317, day 61.6) is smaller and later than
the SIR peak (469, day 42.4). Slowing transmission through a latent stage therefore flattens the
curve, which is exactly the public-health argument for "flattening the peak." Births and deaths are
what make the waves recur. With $\mu=0$ the SEIR model gives a single wave that dies out (final
$S\approx212$ and $I\to0$), whereas $\mu=0.005$ keeps replenishing susceptibles and produces damped
oscillations forever. The chosen $\mu=0.005\,\text{day}^{-1}$ is unrealistically large, it corresponds
to an average lifespan of only 200 days and is deliberately inflated so that the slow demographic
oscillation is visible within a short simulation. With a realistic human lifespan ($\mu\sim1/70\,\text{yr}\approx4\times10^{-5}\,\text{day}^{-1}$)
the same phenomenon would appear but the inter-wave period would be of the order of the generation
time of the infection multiplied by the demographic turnover, such as many years to decades rather than
~290 days. Therefore over a realistic horizon the epidemic would look like a single outbreak rather than a
rapidly repeating series of waves.

---

## 6. Part E — Sensitivity analysis

![Part E heatmaps](figures/partE_sensitivity_heatmaps.png)

**(iii) Discussion.** $\beta$ combines contact rate and per-contact transmission, so it is the
parameter that social distancing, masking and vaccination act on. $\gamma$ is set by the biology of
the pathogen, so it can only be changed indirectly, through treatment, antivirals and faster case
detection. In practice $\beta$-type measures are the ones that can be pushed far enough to bring
$R_0$ below 1, and this matters because the transition around $R_0=1$ is steep: moving below the
line removes the wave, whereas sliding along it does not.

---

## 7. Discussion — Hypothesis revisited

1. **Peak in Part B — predicted the peak is around day 40–50 with roughly 400–500 infected.**
   _Supported._ The simulated peak is 469 people on day 42.5.
2. **Part D — predicted several waves, because births continuously resupply susceptibles.**
   _Supported._ The first wave (day 61.6, 317 people) is followed by damped waves at ~290-day intervals.
3. **Part E — predicted that $\beta$ would move the peak more than $\gamma$.**
   _Rejected._ The elasticities are $+1.58$ for $\beta$ but $-1.84$ for $\gamma$ so $\gamma$ is slightly the stronger lever. Although both β and γ affect R₀, γ also controls how quickly people recover, so the epidemic peak was more sensitive to γ than to β.

---

## 8. AI statement

> Statement: **Deepseek** was used to complete this homework.
> The full chat is available at:
> 【WorkBuddy】实现 HW1 SIR 和 SEIR 模型代码
> https://workbuddy.link/p/WBN0VPCWFdUhy7Q9xLjuic?ext2=copy_link

<!-- TODO: replace [AI-TOOL-NAME] and [PUBLIC-URL(S)] with the real values, and make sure the link
     opens for anyone after signing out. If you in fact used no generative AI, replace this whole
     block with:
     "Statement: No generative AI (in any form) has been used to complete this homework." -->
