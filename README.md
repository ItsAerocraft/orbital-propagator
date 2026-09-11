# Orbital Propagator

A 6 degree-of-freedom orbital decay simulator, coupling orbital and attitude dynamics for a tumbling satellite under atmospheric drag, J2/J3/J4 gravitational perturbations, and stochastic solar flux (F10.7) Monte-Carlo variation.

This project began as a rudimentary Excel spreadsheet model and has been extended into a coupled Python simulation. Since drag depends on the satellite's instantaneous cross-sectional area, and that area changes continuously as the satellite tumbles, orbital decay and attitude dynamics cannot be modeled independently.

## Features

- **13-element state vector**: position, velocity, quaternion orientation, and body-frame angular velocity, integrated together via adaptive RK45 (Dormand-Prince)
- **Coupled attitude-orbit dynamics**: tumble drives time-varying cross-sectional area, which drives drag, which drives orbital decay
- **Perturbations**: J2, J3, and J4 zonal harmonics (Earth oblateness and higher-order gravity effects)
- **Atmospheric model**: NRLMSISE-00 via [`pymsis`](https://github.com/SWxTREC/pymsis), including stochastic F10.7 solar flux variation via a Monte Carlo Ornstein-Uhlenbeck process
- **Attitude dynamics**: gravity-gradient and aerodynamic torques, quaternion kinematics (gimbal-lock free), Euler's rigid body equations

## Repository Structure

| Folder | Contents |
|---|---|
| `z_main/` | Configuration, dynamics function, and integrator |
| `forces/` | Gravity, drag, and J2/J3/J4 perturbation forces |
| `torques/` | Gravity-gradient and aerodynamic torques |
| `tumble/` | Quaternion kinematics, angular acceleration, inertia, silhouette area |
| `conversion/` | Orbital element ↔ Cartesian, inertial ↔ body frame conversions |
| `environment/` | Atmospheric density lookup, stochastic F10.7 model |
| `simulate.py` | Entry point — runs the simulation and generates plots |

## Getting Started

```bash
pip install numpy scipy matplotlib pymsis
python simulate.py
```

Model parameters (orbit, satellite geometry, integration settings) are set in `z_main/config.py`.

Graph inputs available include "orbit", "stochastic", "altitude", "silhouette", and "omega".

## Background & Theory

A full technical writeup, covering the state vector, integration method, perturbation and drag models, and attitude dynamics derivations, is available ([here](https://drive.google.com/file/d/1MmETuXVPX74q5AQh2TI0qvaK1W6nQ9bC/view?usp=sharing)).

## Status & Next Steps

The propagator is functionally complete, up to phase B. Ongoing work extends the model with an active attitude control layer, hopefully including a classical PD detumble controller, followed by a nonlinear controller based on contraction theory.

## References

Key models and constants are drawn from Vallado (2013), Schaub and Junkins (2018), and Picone et al. (2002). Please see the full technical writeup for complete citations.