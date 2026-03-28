# Time-Reversed BSDEs for Accurate Gradient Estimation in Diffusion Models

This repository, developed by **Yuhang Mei**, contains the Python code for reproducing the experiments in our paper:

**[Time-Reversed BSDEs for Accurate Gradient Estimation in Diffusion Models](https://arxiv.org/abs/2603.20455)**

## Setup

The code requires the following dependencies:

- Python
- NumPy
- SciPy
- Matplotlib
- PyTorch

## Code Structure

- The neural network architectures are defined in `network.py`.
- Helper functions and utilities are implemented in `utils.py`.

## Running the Code and Reproducing the Figures

### 1. Linear System and Stochastic Pendulum Examples

Run the following scripts to generate and save the trained networks:

- `linear_example_compare.py`
- `solving_nonlinearbsde.py`

We also provide the trained networks and figures used in the paper.

For the **linear example**, use:

- `linear_example_inferencecompare.ipynb`

to load the saved data and reproduce the figures.

For the **stochastic pendulum example**, use:

- `optimization_compare.ipynb`
- `optimization_compare_autograd.ipynb`

to generate and save the optimization results for the initial distribution, and then use:

- `optresult_vs_heatmap.ipynb`

to reproduce the heatmap figures.

### 2. Fine-tuning the Toy Diffusion Model

Run the following scripts to generate and save the trained networks:

- `finetune_diffusion.py`
- `finetune_diffusion_ad_matching.py`

We use the high-temperature run as a warm start for the low-temperature experiments.

To reproduce the corresponding figures in the paper, use:

- `finetune_diff_result_cost_compare.ipynb`

## Notes

- Trained networks, data, and figures are provided.
