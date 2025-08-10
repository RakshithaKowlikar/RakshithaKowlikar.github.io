---
title: "DDPM: Explained"
description: "A math-intensive breakdown of 'Denoising Diffusion Probabilistic Models', showing how forward diffusion, reverse denoising, and noise-prediction integrate into a full generative process."
date: 2025-08-10
tags: [computer-vision, deep-learning, generative-models, diffusion]
---

## TL;DR

This paper proposes a generative model where images are corrupted by a fixed Gaussian process and then reconstructed by learning the reverse process.  
The model predicts the noise added in the forward process and uses it to recover the original image.

***

### Denoising Diffusion Probabilistic Models (Ho et al., 2020)

---

### Some basic notation used throughout the paper

Let:

$$
x_0 \in \mathbb{R}^d
$$

be the original clean image. The total number of timesteps is $T$.

The forward (diffusion) process produces:

$$
x_1, x_2, \dots, x_T
$$

with $x_T$ approaching isotropic Gaussian noise.

#### Variance schedule

Let:

$$
\beta_t \in (0,1), \quad \alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s
$$

---

### Forward Diffusion Process

The forward transition is:

$$
q(x_t | x_{t-1}) = \mathcal{N} \big( x_t; \sqrt{\alpha_t} \, x_{t-1}, \beta_t \mathbf{I} \big)
$$

which can be written as:

$$
x_t = \sqrt{\alpha_t} \, x_{t-1} + \sqrt{\beta_t} \, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \mathbf{I})
$$

---

### Direct Noising from $x_0$

Tt allows sampling $x_t$ directly from $x_0$ in one step, enabling efficient training.

By unrolling the Markov chain:

$$
q(x_t | x_0) = \mathcal{N} \big( x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1 - \bar{\alpha}_t) \mathbf{I} \big)
$$

Thus:

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

---

### Reverse Process

The  task is to start from $x_T \sim \mathcal{N}(0, \mathbf{I})$ and iteratively recover $x_0$.

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N} \big( x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t) \big)
$$

From Gaussian identities, the true reverse mean is:

$$
\mu_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} \left[ x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \, \epsilon \right]
$$

Since $x_0$ is unknown, we train a network $\epsilon_\theta(x_t, t)$ to predict $\epsilon$.

Predicting noise is easier than predicting $x_0$. This is because noise has a stationary Gaussian distribution across timesteps.

---

### Training Objective

From the direct noising equation:

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon
$$

we want:

$$
\epsilon_\theta(x_t, t) \approx \epsilon
$$

The **simplified loss** is:

$$
L_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$


---

### Sampling Procedure


1. Start from:
$$
x_T \sim \mathcal{N}(0, \mathbf{I})
$$

2. For $t = T, \dots, 1$:
   - Predict noise:
$$
\hat{\epsilon} = \epsilon_\theta(x_t, t)
$$
   - Compute reverse mean:
$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left[ x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \, \hat{\epsilon} \right]
$$
   - Sample:
$$
x_{t-1} \sim \mathcal{N}(\mu_\theta(x_t, t), \sigma_t^2 \mathbf{I})
$$
     where $\sigma_t^2$ is fixed (e.g., $\beta_t$) or learned.

---

### Network Architecture

Skip connections propagate the local detail across scales, this is imperative for accurate image reconstruction.

- **Backbone**: Residual U-Net  
- **Attention**: Multi-resolution self-attention  
- **Timestep encoding**: $t$ mapped to sinusoidal embeddings → MLP → injected into residual blocks

---

### Derivation of Reverse Mean Formula

Let's see exactly how the noise-prediction formulation connects the forward and reverse processes.

From the forward process:

$$
q(x_t | x_0) = \mathcal{N} \big( x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1 - \bar{\alpha}_t) \mathbf{I} \big)
$$

We also have the one-step forward:

$$
q(x_{t-1} | x_t, x_0) \propto q(x_t | x_{t-1}) \, q(x_{t-1} | x_0)
$$

Both are Gaussian, so their product is Gaussian with:

$$
q(x_{t-1} | x_t, x_0) = \mathcal{N} \big( x_{t-1}; \tilde{\mu}(x_t, x_0), \tilde{\beta}_t \mathbf{I} \big)
$$

where:

$$
\tilde{\mu}(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \, \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t
$$

Substituting $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \, \epsilon \right)$ gives:

$$
\tilde{\mu}(x_t, \epsilon) = \frac{1}{\sqrt{\alpha_t}} \left[ x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \, \epsilon \right]
$$

Replacing $\epsilon$ with $\epsilon_\theta(x_t, t)$ gives the learned reverse mean $\mu_\theta$ used in sampling.

---

### Stage Formulas

**Forward process (given)**:
$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

**Noise prediction**:
$$
\hat{\epsilon} = \epsilon_\theta(x_t, t)
$$

**Reverse mean**:
$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left[ x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \, \hat{\epsilon} \right]
$$

**Reverse sampling**:
$$
x_{t-1} \sim \mathcal{N}(\mu_\theta(x_t, t), \sigma_t^2 \mathbf{I})
$$

**Training loss**:
$$
L_{\text{simple}}(\theta) = \| \epsilon - \epsilon_\theta(x_t, t) \|^2
$$

---
