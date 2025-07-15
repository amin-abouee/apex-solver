

# Lie Theory Cheat Sheet

This document provides a summary of common Lie groups and their properties, as well as key formulas and Jacobians used in robotics and state estimation.

## Lie Groups and Tangent Spaces

| Lie group M, O | size | dim | X ∈ M | Constraint | T_E M (Lie Algebra) | τ ∈ ℝⁿ |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| Vector n-D | ℝⁿ, + | n | n | **v** ∈ ℝⁿ | **v** - **v** = 0 | **v** ∈ ℝⁿ | **v** ∈ ℝⁿ |
| Unit Complex number | S¹, · | 2 | 1 | z ∈ ℂ | z*z = 1 | iθ ∈ iℝ | θ ∈ ℝ |
| 2D Rotation | SO(2), · | 4 | 1 | **R** | **R**ᵀ**R** = **I** | $$[\theta]_x = \begin{bmatrix} 0 & -\theta \\ \theta & 0 \end{bmatrix} \in so(2)$$ | θ ∈ ℝ |
| 2D Rigid Motion | SE(2), · | 9 | 3 | $$M = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$$ | **R**ᵀ**R** = **I** | $$[\rho, \theta]^\wedge \in se(2)$$ | $$[\rho, \theta]^T \in \mathbb{R}^3$$ |
| Unit Quaternion | S³, · | 4 | 3 | **q** ∈ ℍ | **q*** **q** = 1 | **θ**/2 ∈ ℍₚ | **θ** ∈ ℝ³ |
| 3D Rotation | SO(3), · | 9 | 3 | **R** | **R**ᵀ**R** = **I** | $$[\theta]_x \in so(3)$$ | **θ** ∈ ℝ³ |
| 3D Rigid Motion | SE(3), · | 16 | 6 | $$M = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$$ | **R**ᵀ**R** = **I** | $$[\rho, \theta]^\wedge \in se(3)$$ | $$[\rho, \theta]^T \in \mathbb{R}^6$$ |

## Right Jacobians

| Operation | Inverse | Compose | Exp | Log | Right-⊕ | Right-⊖ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Right Jacobians** | $$J_X^{-1} = -Ad_X$$ | $$J_{XY}^X = Ad_{Y^{-1}}$$, $$J_{XY}^Y = I$$ | $$J_{Exp(\tau)} = J_r(\tau)$$ | $$J_{Log(X)} = J_r^{-1}(\tau)$$ | $$J_{X\oplus\tau}^X = Ad_{Exp(\tau)^{-1}}$$, $$J_{X\oplus\tau}^\tau = J_r(\tau)$$ | $$J_{X\ominus Y}^X = J_r^{-1}(\tau)$$, $$J_{X\ominus Y}^Y = -J_l^{-1}(\tau)$$ |

**Note:** In accordance to `manif` implementation, all Jacobians in this document are **right Jacobians**, whose definition reads:
$$ \frac{\delta f(X)}{\delta X} = \lim_{\phi \to 0} \frac{f(X \oplus \phi) \ominus f(X)}{\phi} $$
However, notice that one can relate the left- and right- Jacobians with the Adjoint,
$$ (\frac{\partial f(X)}{\partial X^L}) Ad_X = Ad_{f(X)} (\frac{\partial f(X)}{\partial X^R}) $$
see [1] Eq. (46).

## Operations

| M, o | Identity | Inverse | Compose | Act | Exp | Log |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ℝⁿ, +** | **v** = [0] | -**v** | **v₁** + **v₂** | **v** + **p** | | **v** |
| **S¹, ·** | z = 1 + i0 | z* | z₁z₂ | z u | z = cosθ + i sinθ | θ = arctan2(Im(z), Re(z)) |
| **SO(2), ·** | **R** = **I** | **R**⁻¹ = **R**ᵀ | **R₁R₂** | **R**·**v** | $$R = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$ | θ = arctan2(r₂₁, r₁₁) |
| **SE(2), ·** | **M** = **I** | $$M^{-1} = \begin{bmatrix} R^T & -R^T t \\ 0 & 1 \end{bmatrix}$$ | $$M_1 M_2 = \begin{bmatrix} R_1 R_2 & t_1+R_1 t_2 \\ 0 & 1 \end{bmatrix}$$ | **M**·**p** = **t** + **R**·**p** | $$M = \begin{bmatrix} Exp(\theta) & V(\theta)\rho \\ 0 & 1 \end{bmatrix}^{(1)}$$ | $$\tau = \begin{bmatrix} V^{-1}(\theta)\rho \\ Log(R) \end{bmatrix}^{(1)}$$ |
| **S³, ·** | q = 1+i0+j0+k0 | q* = w - ix - jy - jz | q₁q₂ | qvq* | q = cos(θ/2) + **u** sin(θ/2) | $$\theta = 2v \frac{\arctan2(\|v\|, w)}{\|v\|}$$ |
| **SO(3), ·** | **R** = **I** | **R**⁻¹ = **R**ᵀ | **R₁R₂** | **R**·**v** | $$R = I + \frac{\sin\theta}{\theta}[u]_x + \frac{1-\cos\theta}{\theta^2}[u]_x^2$$ | $$\theta = \frac{\theta}{2\sin\theta}(R-R^T)^\vee$$ |
| **SE(3), ·** | **M** = **I** | $$M^{-1} = \begin{bmatrix} R^T & -R^T t \\ 0 & 1 \end{bmatrix}$$ | $$M_1 M_2 = \begin{bmatrix} R_1 R_2 & t_1+R_1 t_2 \\ 0 & 1 \end{bmatrix}$$ | **M**·**p** = **t** + **R**·**p** | $$M = \begin{bmatrix} Exp(\theta) & V(\theta)\rho \\ 0 & 1 \end{bmatrix}^{(2)}$$ | $$\tau = \begin{bmatrix} V^{-1}(\theta)\rho \\ Log(R) \end{bmatrix}^{(2)}$$ |

## Adjoint and Jacobians

| M, o | Ad | J | J_r | J_l | J_p' ^X (Act) | J_p' ^p (Act) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ℝⁿ, +** | **I** ∈ ℝⁿˣⁿ | **I** | **I** | **I** | **I** | **I** |
| **S¹, ·** | 1 | 1 | 1 | 1 | R[i]ₓv | R |
| **SO(2), ·** | 1 | 1 | 1 | 1 | R[1]ₓv | R |
| **SE(2), ·** | $$\begin{bmatrix} R & [t]_x R \\ 0 & R \end{bmatrix}$$ | $$\begin{bmatrix} \frac{\sin\theta}{\theta} & -\frac{1-\cos\theta}{\theta} \\ \frac{1-\cos\theta}{\theta} & \frac{\sin\theta}{\theta} \end{bmatrix}$$ | $$\begin{bmatrix} \frac{\sin\theta}{\theta} & \frac{\cos\theta-1}{\theta} & \frac{\theta\rho_1+\rho_2-\rho_2\cos\theta-\rho_1\sin\theta}{\theta^2} \\ \frac{1-\cos\theta}{\theta} & \frac{\sin\theta}{\theta} & \frac{-\rho_1+\theta\rho_2+\rho_1\cos\theta-\rho_2\sin\theta}{\theta^2} \\ 0 & 0 & 1 \end{bmatrix}$$ | $$\begin{bmatrix} \frac{\sin\theta}{\theta} & \frac{1-\cos\theta}{\theta} & \frac{\rho_2+\theta\rho_1-\rho_2\cos\theta-\rho_1\sin\theta}{\theta^2} \\ -\frac{1-\cos\theta}{\theta} & \frac{\sin\theta}{\theta} & \frac{-\rho_1+\theta\rho_2+\rho_1\cos\theta-\rho_2\sin\theta}{\theta^2} \\ 0 & 0 & 1 \end{bmatrix}$$ | $$\begin{bmatrix} R & R[p]_x \\ 0 & 1 \end{bmatrix}$$ | $$\begin{bmatrix} R & 0 \\ 0 & 1 \end{bmatrix}$$ |
| **S³, ·** | R(q) | | $$I + \frac{1-\cos\theta}{\theta^2}[\theta]_x + \frac{\theta-\sin\theta}{\theta^3}[\theta]_x^2$$ | $$I - \frac{1-\cos\theta}{\theta^2}[\theta]_x + \frac{\theta-\sin\theta}{\theta^3}[\theta]_x^2$$ | -R(q)[v]ₓ⁽³⁾ | R(q)⁽³⁾ |
| **SO(3), ·** | **R** | | $$I + \frac{1-\cos\theta}{\theta^2}[\theta]_x + \frac{\theta-\sin\theta}{\theta^3}[\theta]_x^2$$ | $$I - \frac{1-\cos\theta}{\theta^2}[\theta]_x + \frac{\theta-\sin\theta}{\theta^3}[\theta]_x^2$$ | -R[v]ₓ | R |
| **SE(3), ·** | $$\begin{bmatrix} R & [t]_x R \\ 0 & R \end{bmatrix}$$ | | $$\begin{bmatrix} J_r(\theta) & Q(-\rho, -\theta) \\ 0 & J_r(\theta) \end{bmatrix}^{(4)}$$ | $$\begin{bmatrix} J_l(\theta) & Q(\rho, \theta) \\ 0 & J_l(\theta) \end{bmatrix}^{(4)}$$ | [R, -R[p]ₓ] | R |

## Some useful identities:
$$ X \oplus \tau = Ad_X \tau \oplus X \quad | \quad Ad_{X^{-1}} = Ad_X^{-1} \quad | \quad Ad_{XY} = Ad_X Ad_Y \quad | \quad J_l(\tau) = Ad_{Exp(\tau)}J_r(\tau) \quad | \quad J_l(\tau) = J_r(-\tau) $$

---
(1) $$ V(\theta) = \frac{\sin\theta}{\theta}I + \frac{1-\cos\theta}{\theta^2}[u]_x $$
(2) $$ V(\theta) = I + \frac{1-\cos\theta}{\theta^2}[u]_x + \frac{\theta-\sin\theta}{\theta^3}[u]_x^2 $$
(3) $$ R(q) = \begin{bmatrix} w^2+x^2-y^2-z^2 & 2(xy-wz) & 2(xz+wy) \\ 2(xy+wz) & w^2-x^2+y^2-z^2 & 2(yz-wx) \\ 2(xz-wy) & 2(yz+wx) & w^2-x^2-y^2+z^2 \end{bmatrix} $$
(4) $$ Q(\rho, \theta) = \frac{1}{2}[\rho]_x + \frac{\theta - \sin\theta}{\theta^3}([\theta]_x[\rho]_x + [\rho]_x[\theta]_x + [\theta]_x[\rho]_x[\theta]_x) - \frac{1-\frac{\theta^2}{2}-\cos\theta}{\theta^4}([\theta]_x^2[\rho]_x + [\rho]_x[\theta]_x^2 - 3[\theta]_x[\rho]_x[\theta]_x) - (\frac{1}{2}(\frac{1-\frac{\theta^2}{2}-\cos\theta}{\theta^4} - \frac{3(\theta-\sin\theta-\frac{\theta^3}{6})}{\theta^5}))([\theta]_x[\rho]_x[\theta]_x^2 + [\theta]_x^2[\rho]_x[\theta]_x) $$
