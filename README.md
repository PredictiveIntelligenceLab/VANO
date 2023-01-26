# VANO
## Variational Autoencoding Neural Operators

![master_figure-2](https://user-images.githubusercontent.com/3844367/214755990-dd375c41-4708-4590-a86c-14f88db07b9e.png)

This repository contains code and data accompanying the manuscript titled "Variational Autoencoding Neural Operators", authored by Jacob Seidman, Georgios Kissas, George Pappas and Paris Perdikaris.

## Abstract

Unsupervised learning with functional data is an emerging paradigm of machine learning research with applications to computer vision, climate modeling and physical systems. A natural way of modeling functional data is by learning operators between infinite dimensional spaces, leading to discretization invariant representations that scale independently of the sample grid resolution. Here we present Variational Autoencoding Neural Operators (VANO), a general strategy for making a large class of operator learning architectures act as variational autoencoders. For this purpose, we provide a novel rigorous mathematical formulation of the variational objective in function spaces for training. VANO first maps an input function to a distribution over a latent space using a parametric encoder and then decodes a sample from the latent distribution to reconstruct the input, as in classic variational autoencoders. We test VANO with different model set-ups and architecture choices for a variety of benchmarks. We start from a simple Gaussian random field where we can analytically track what the model learns and progressively transition to more challenging benchmarks including modeling phase separation in Cahn-Hilliard systems and real world satellite data for measuring Earth surface deformation.

