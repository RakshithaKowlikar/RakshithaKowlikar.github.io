---
title: "DeepPose: Explained"
description: "This paper proposes a DNN-based direct regression method to localize human joints from full images."
date: 2025-08-10
tags: [computer-vision, deep-learning, pose-estimation]
---

## TL;DR
This paper proposes a DNN-based direct regression method to localize human joints from full images. The model outputs 2D joint coordinates.
$$
(x_i, y_i)
$$
It then minimizes the squared Euclidean distance between predicted and true positions.

***


### DeepPose: Human Pose Estimation via Deep Neural Networks

### Some basic notation used throughout the paper
Let **x** be the input image. Then **y** is the full pose vector consisting of all k joint coordinates flattened into one long vector:
$$
y=[y_1^T,y_2^T,\dots,y_k^T]^T, y_i=(x_i,y_i)\in\mathbb{R}^2
$$

A rectangle tightly around a person is represented by:
$$
b = (b_c,b_w,b_h)
$$
Where
$$
b_c
$$
is its center that is
$$
b_c = (c_x,c_y)
$$
and
$$
b_w, b_h
$$
are its width and height respectively.

### Bounding Box Normalization
To create a consistent input reference frame for the network, we need to normalize.

The normalization operator $N(\cdot; b)$ centers the joint or image by subtracting $b_c$ and scales it by dividing by $b_w$ in $x$ and $b_h$ in $y$.
So,
$$
N(y_i;b)=\begin{pmatrix}1/b_w & 0\\ 0 & 1/b_h\end{pmatrix}(y_i-b_c)
$$

and $N(x;b)$ crops out that rectangle around the person and resizes it to the network’s fixed input size (220 × 220 in DeepPose).

### Direct Regression function
We obtain the network's prediction of joint $i$ through the function
$$
\hat{y} = \psi(\tilde{x};\theta) \in\mathbb{R}^{2k}
$$
where $\psi$ is the 7-layer convolution network with $\theta$ comprising of all the learned weights, $k$ is the number of joints, and $\tilde{x} = N(x;b)$.

$$
\hat{y} = [\hat{x_1}, \hat{y_1}, \hat{x_2}, \hat{y_2}, ..., \hat{x_i}, \hat{y_i}]
$$
where each pair $(\hat{x_i},\hat{y_i})$ is the network’s prediction of joint $i$ but in the normalized coordinate frame.

The network learns locations relative to the person's body, not absolute image coordinates.

To get the original location of the predicted points in the image we invert the normalization:
$$
y^\ast = N^{-1}(\hat{y_i};b) = \begin{pmatrix}b_w & 0\\ 0 & b_h\end{pmatrix}\hat{y_i} + b_c
$$

### Network Architecture
This network draws inspiration from the AlexNet-style architecture.

| **Layer** | **Operation** | **Input Size** | **Output Size** |
|:---------------|:---------------------------------------------------------------------------|:--------------------------|:--------------------------|
| **Input** | $N(x;b)$ -- the $220 \times 220$ RGB crop                                   | $220 \times 220 \times 3$ | $220 \times 220 \times 3$ |
| **Conv1** | 96 filters of size $11 \times 11$, stride 4 → ReLU                         | $220 \times 220 \times 3$ | $55 \times 55 \times 96$  |
| **LRN1** | Local Response Normalization (encourages competition between channels)     | $55 \times 55 \times 96$  | $55 \times 55 \times 96$  |
| **Pool1** | $3 \times 3$ max-pool, stride 2 (takes max in each $3\times3$ block)         | $55 \times 55 \times 96$  | $27 \times 27 \times 96$  |
| **Conv2** | 256 filters of size $5 \times 5$, padding to preserve spatial dims → ReLU  | $27 \times 27 \times 96$  | $27 \times 27 \times 256$ |
| **LRN2** | Local Response Normalization (encourages competition between channels)     | $27 \times 27 \times 256$ | $27 \times 27 \times 256$ |
| **Pool2** | $3 \times 3$ max-pool, stride 2                                            | $27 \times 27 \times 256$ | $13 \times 13 \times 256$ |
| **Conv3** | 384 filters of size $3 \times 3$, padding → ReLU                           | $13 \times 13 \times 256$ | $13 \times 13 \times 384$ |
| **Conv4** | 384 filters of size $3 \times 3$, padding → ReLU                           | $13 \times 13 \times 384$ | $13 \times 13 \times 384$ |
| **Conv5** | 256 filters of size $3 \times 3$, padding → ReLU                           | $13 \times 13 \times 384$ | $13 \times 13 \times 256$ |
| **Pool5** | $3 \times 3$ max-pool, stride 2                                            | $13 \times 13 \times 256$ | $6 \times 6 \times 256$   |
| **FC6** | Fully connected → ReLU                                                     | $6 \times 6 \times 256 = 9216$ | 4096                   |
| **FC7** | Fully connected → ReLU                                                     | 4096                      | 4096                   |
| **Output** | Linear layer producing $2k$ values                                         | 4096                      | $2k$ (e.g. $2\times14$)   |

![Initial stage image](/assets/lib/Screenshot from 2025-06-03 11-02-11.png)

### Cascade of Pose Regressors
The single regressor captures rough pose but lacks finer details due to pooling and limited input size.

We use cascade method to iteratively refine coarse predictions. This is done by zooming in on joints for higher precision.

Stage 1 predicts an initial pose $y^{(1)}$ from the full image crop. Stages 2 to S then crop a higher‐resolution patch around the current estimate and predict a small displacement to refine it, letting later stages correct small errors.

#### Notation
Let **k** be the number of joints. True pose: $y = [y_1^T, y_2^T, \dots, y_k^T]^T$. At stage *s*, the predicted pose is $y^{(s)}$ with the $i$-th joint $y_i^{(s)}$.

#### Pose Diameter
$$
\mathrm{diam}(y) = \|y_p - y_q\|_2,
$$
where $(p,q)$ is a fixed torso-opposite pair (e.g., left shoulder vs. right hip).

To make the refinement box size proportional to the person's scale in the image, pose diameter has been used.

#### Refinement Box
Refinement box for joint *i* at stage *s−1*:
$$
b_i^{(s-1)} = \bigl(y_i^{(s-1)},\;\sigma\,\mathrm{diam}\bigl(y^{(s-1)}\bigr),\;\sigma\,\mathrm{diam}\bigl(y^{(s-1)}\bigr)\bigr),
$$
i.e., center at the current joint, with width/height = $\sigma \times \text{pose diameter}$.

### Stage Formulas
**Stage 1 (initial full-image regression)**
$$
y^{(1)} \;=\; N^{-1}\!\Bigl(\,\psi\bigl(N(x;\,b_0)\;;\,\theta_1\bigr)\;;\,b_0\Bigr)
$$

**Stage $s \ge 2$ (joint-wise refinement)**
For each joint *i*, predict a normalized displacement and add it to the previous estimate:
$$
\Delta \hat{y}_i^{(s)} = \psi_i\bigl(N(x;\,b_i^{(s-1)})\;;\,\theta_s\bigr) \quad \implies \quad y_i^{(s)} = y_i^{(s-1)} \;+\; N^{-1}\!\bigl(\,\Delta \hat{y}_i^{(s)}\;;\,b_i^{(s-1)}\bigr)
$$
In practice, $S=3$ stages suffice.

### Training the Cascade

#### Stage 1
$$
\underset{\theta_1}{\min} \sum_{(x,y)\in D_N} \sum_{i} \|\,y_i \;-\; \psi_i\bigl(N(x)\;;\,\theta_1\bigr)\|_2^2.
$$

#### Stage $s \ge 2$

We compute the mean $\mu_i^{(s-1)}$ and covariance $\Sigma_i^{(s-1)}$ of the errors from the previous stage. Then, for each training joint, we sample a fake error (displacement) $\delta$.

We sample from a Gaussian to create a diverse set of realistic incorrect predictions for the refiner to learn from.
$$
\delta \;\sim\; \mathcal{N}\bigl(\,\mu_i^{(s-1)},\,\Sigma_i^{(s-1)}\bigr)
$$
This creates a jittered box $b$ and an augmented training set $D_s^A$. The objective is to learn the parameters $\theta_s$ that minimize the L2 error on this new augmented data.
$$
\theta_s \;=\; \underset{\theta}{\arg\min}\; \sum_{(\tilde{x},\tilde{y}_i)\in D_s^A} \|\tilde{y}_i \;-\; \psi_i(\tilde{x};\,\theta)\|_2^2.
$$

Each $\theta_s$ is learned independently using the same network architecture.

![Cascaded layers image](/assets/lib/cascade.png)