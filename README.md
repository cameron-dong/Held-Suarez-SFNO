# SFNO Model Trained on Held-Suarez Simulation Data

This repository contains code used to train a Spherical Fourier Neural Operator (SFNO) deep learning model
on Held-Suarez simulation data. The Held-Suarez simulations are performed using a traditional Eulerian 
Spherical Harmonic Transform core in TensorFlow (https://github.com/cameron-dong/TensorDynamics).

These simulations are characterized by a linear relaxation towards a zonally symmetric basic state that is
also symmetric latitudinally about the equator. Eddies are generated, drawing energy from the basic state.
Statistical properties of the basic state maintenance and eddies is used for dynamical core intercomparison.

This project builds an SFNO model similar to as in Bonev et al. (2023; https://arxiv.org/abs/2306.03838), 
which can be stepped forward autoregressively. When trained for successively longer rollout lengths, the
SFNO model is stable for simulation lengths greater than 1000 days. The following figures display results
from a model with 4 SFNO blocks and an embedding dimension of size 96, trained on data from a T42 (3 degree)
resolution core.

While the traditional dynamical core mean state is characterized by near symmetry between the two hemispheres, 
the SFNO model does not maintain perfectly this characteristic. Additionally, the SFNO model is biased towards
a higher variance of high wavenumber (~5) eddies in the Northern Hemisphere. In general, however, the SFNO
model approximately reproduces the traditional dynamical core statistics.

## Example 9 day rollout
![alt text](https://github.com/user-attachments/assets/e8d945a1-6166-4e1d-839b-0454579122c0)

## Statistical Properties of the traditional dynamical core
![alt text](https://github.com/user-attachments/assets/4bdfc26f-c99a-4671-9aa6-573fa58bb639)

## Statistical Properties of SFNO 1000-day simulation
![alt text](https://github.com/user-attachments/assets/254e4f91-f405-417e-84e5-5952d380a97e)


