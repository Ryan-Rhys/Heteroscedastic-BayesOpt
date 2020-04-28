# Copyright Ryan-Rhys Griffiths 2020
# Author: Ryan-Rhys Griffiths
"""
This module contains objective functions for heteroscedastic Bayesian Optimisation. Train objectives represent the
noise-corrupted values that a model will observe within the BO loop. Exact objectives represent the ground truth
black-box objective being optimised.
"""

from matplotlib import pyplot as plt
import torch


def train_sin_objective(X, noise_coefficient=0.25, coefficient=0.2, fplot=False):
    """
    1D sin wave where heteroscedastic noise increases linearly in the input domain.
    Bounds for the bimodal function are [0, 10]. One maxima is higher than the other.

    :param X: input dimension
    :param noise_coefficient: noise level coefficient for linearly increasing noise
    :param coefficient: Has the effect of making the maximum with larger noise larger
    :param fplot: Boolean indicating whether to plot noisy samples of g(x)
    :return: train_obj: g(X) + noise(X)
    """

    train_obj = torch.sin(X) + coefficient*X + (noise_coefficient * torch.randn_like(X))

    if fplot:
        plt.plot(X, train_obj, '+', color='green', markersize='12', linewidth='8', label='samples with noise')
        plt.xlabel('x')
        plt.ylabel('g(x)')
        plt.title('Noisy Samples of g(x)')
        plt.legend()
        plt.ylim(-2, 2)
        plt.xlim(0, 10)
        plt.show()

    return train_obj


def exact_sin_objective(X, plot_sample, noise_coefficient=0.25, coefficient=0.2, fplot=True):
    """
    Objective function f(x) = g(x) - s(x) for the sin wave with linear noise.
    Used for monitoring the best value in the optimisation obtained so far.

    :param X: input to evaluate objective; can be an array of values
    :param plot_sample: Sample for plotting purposes (points in the input domain)
    :param noise_coefficient: noise level coefficient
    :param coefficient: Has the effect of making the maximum with larger noise larger
    :param fplot: Boolean indicating whether to plot the black-box objective and ground truth values of the acquisitions
    :return: exact_obj f(x)
             noise_obj s(x)
             optimal_value: global maximiser of f(x)
    """

    main_obj = torch.sin(X) + coefficient*X  # g(x)
    noise_obj = noise_coefficient * X  # s(x)
    exact_obj = main_obj - noise_obj  # f(x)

    optimal_value = None

    # Only if we want to plot the objective and/or compute the global maximiser

    if plot_sample is not None:

        plot_main_obj = torch.sin(plot_sample) + coefficient*plot_sample
        plot_exact_obj = plot_main_obj - noise_coefficient*plot_sample

        optimal_value = torch.max(plot_exact_obj)  # treat plot grid as input to maximise continous function

        if fplot:
            plt.scatter(X, exact_obj, color='green', marker='+', label='Acquisitions')
            plt.plot(plot_sample, plot_exact_obj, color='purple', label='f(x)')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title('Black-box Objective f(x)')
            plt.ylim(-3, 1)
            plt.xlim(0, 10)
            plt.legend()
            plt.show()

    exact_obj = exact_obj
    noise_obj = noise_obj

    return exact_obj, noise_obj, optimal_value

# Legacy Code below will need to be modified for compatibility with BoTorch


def max_one_off_sin_noise_objective(X, noise, coefficient, fplot=True):
    """
    Objective function for maximising objective + aleatoric noise (a one-off good value!) for the sin wave with linear
    noise. Used for monitoring the best value in the optimisation obtained so far.

    :param X: input to evaluate objective; can be an array of values
    :param noise: noise level coefficient
    :param coefficient: Has the effect of making the maximum with larger noise larger
    :param fplot: Boolean indicating whether to plot the black-box objective
    :return: value of the black-box objective that penalises aleatoric noise, value of the noise at X
    """

    noise_value = noise * X  # value of the heteroscedastic noise at the point(s) X
    objective_value = torch.sin(X) + coefficient*X
    composite_objective = objective_value + noise_value

    if fplot:
        plt.plot(X, composite_objective, color='purple', label='objective + aleatoric noise')
        plt.xlabel('x')
        plt.ylabel('objective(x)')
        plt.title('Black-box Objective')
        plt.ylim(-3, 1)
        plt.xlim(0, 10)
        plt.show()

    composite_objective = float(composite_objective)
    noise_value = float(noise_value)

    return composite_objective, noise_value
