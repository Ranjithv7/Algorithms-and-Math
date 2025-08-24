"""
A minimal example of black‑box Variational Inference
for the posterior of the mean of a 1‑D Gaussian with known variance.

Model
------
    mu       ~ Normal(0, 1)          # prior
    x_i | mu ~ Normal(mu, 1)         # likelihood, sigma=1

Exact posterior is analytic (Gaussian); we use it for sanity‑checks.

Variational family
------------------
    q_theta(mu) = Normal(m, s)       # parameters to optimise (m, s>0)

We optimise the Evidence Lower BOund (ELBO):
    ELBO(theta) = E_q[log p(mu)] + Σ_i E_q[log p(x_i|mu)] – E_q[log q(mu)]

Gradient estimates are obtained with the reparameterisation trick
(PyTorch`s `rsample`).

Run the file:  python variational_inference_gaussian.py
"""

import torch
import torch.distributions as D

# ---------- 1. synthetic data -------------------------------------------------
torch.manual_seed(0)
N = 100
sigma = 1.0
true_mu = 2.0
x = true_mu + sigma * torch.randn(N)

# ---------- 2. variational parameters ----------------------------------------
m = torch.tensor(0.0, requires_grad=True)       # mean of q
log_s = torch.tensor(0.0, requires_grad=True)   # log‑std to keep s>0

optimizer = torch.optim.Adam([m, log_s], lr=0.05)

# ---------- 3. ELBO estimator -------------------------------------------------
def elbo(m: torch.Tensor, log_s: torch.Tensor, K: int = 10) -> torch.Tensor:
    """Monte‑Carlo estimate of the ELBO with K samples."""
    s = torch.exp(log_s)
    q = D.Normal(m, s)                # variational distribution
    mu_samples = q.rsample((K,))      # reparameterised samples

    log_prior = D.Normal(0.0, 1.0).log_prob(mu_samples)      # p(mu)
    log_likelihood = D.Normal(mu_samples.unsqueeze(1), sigma).log_prob(x)  # p(x|mu)
    log_q = q.log_prob(mu_samples)     # q(mu)

    # ELBO = E_q[log p(mu) + Σ log p(x|mu) – log q(mu)]
    elbo_estimate = (log_prior + log_likelihood.sum(-1) - log_q).mean()
    return elbo_estimate

# ---------- 4. optimisation loop ---------------------------------------------
for step in range(2000):
    optimizer.zero_grad()
    loss = -elbo(m, log_s)            # maximise ELBO ↔ minimise negative ELBO
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"step {step:4d} | ELBO {(-loss).item():.3f} | "
              f"m {m.item():.3f} | s {torch.exp(log_s).item():.3f}")

# ---------- 5. analytic posterior for comparison -----------------------------
posterior_var = 1 / (1 + N / sigma**2)
posterior_mean = posterior_var * (x.sum() / sigma**2)
posterior_std = posterior_var ** 0.5

kl = D.kl_divergence(D.Normal(m.detach(), torch.exp(log_s).detach()),
                     D.Normal(posterior_mean, posterior_std))

print("\nConverged parameters")
print(f"m_hat  = {m.item():.3f}")
print(f"s_hat  = {torch.exp(log_s).item():.3f}")
print("\nExact posterior")
print(f"mu     = {posterior_mean:.3f} ± {posterior_std:.3f}")
print(f"KL(q ‖ p) ≈ {kl.item():.4f}")
