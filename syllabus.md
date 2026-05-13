# Generative Models for Climate: An Introduction

LEAP Summer Momentum Fellowship Bootcamp — *90-minute lecture*

---

## Motivation

The growing availability of climate model outputs and observational datasets has made data-driven methods increasingly relevant to climate research. Climate systems are complex, chaotic, and only partially observable, leading to deep structural uncertainty in any prediction. This motivates probabilistic methods, which represent predictions as probability distributions.

Generative models are a powerful family of probabilistic methods that learn to sample from complex, high-dimensional distributions. In climate science, they have found applications in statistical downscaling, ensemble generation, data assimilation, and emulation of expensive physical simulations.

---

## Overview

This lecture introduces generative models for climate, focusing on diffusion models, a class that has recently emerged as state-of-the-art across scientific domains. The lecture is organized in three parts:

1. **Context and motivation** *(~15 min)* — Generative models in modern data-driven climate approaches.
2. **Diffusion models: theory and intuition** *(~30 min)* — An intuitive presentation of diffusion models, Langevin sampling and the score function.
3. **Hands-on implementation** *(~40 min)* — We implement a diffusion model on synthetic data. We will then see an application to ocean modeling and data assimilation. The code is provided as a Jupyter notebook.

---

## Learning Objectives

By the end of this lecture, students will be able to:

- explain why probabilistic methods are needed in climate modeling,
- describe the forward and reverse diffusion process and the role of the score function,
- implement a basic diffusion model and apply it to a structured scientific dataset,
- identify climate problems where generative models offer advantages over deterministic baselines.

---

## Prerequisites

- Basic probability 
- Python 
- No prior knowledge of generative models assumed

---

## Further Reading and Resources

The lecture closes with open challenges in climate generative modeling and pointers to recent literature and datasets available through LEAP Pangeo.
