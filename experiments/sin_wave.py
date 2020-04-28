# Copyright Ryan-Rhys Griffiths 2020
# Author: Ryan-Rhys Griffiths
"""
1D sin wave experiment. Random sampling, expected improvement and noisy expected improvement (Letham et al. 2017) with
homoscedastic surrogate model.
"""

import time
import warnings

from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, NoisyExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.optim import optimize_acqf
from botorch.models import FixedNoiseGP, SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
import numpy as np
import torch

from objectives import train_sin_objective, exact_sin_objective


warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

N_INIT = 3  # Number of data points to initialise with


def generate_initial_data(n=N_INIT):
    """
    Generate initialisation dataset by sampling uniformly at random within the bounds of the optimisation problem.
    :param n: number of initialisation points.
    :return: train_x: initialisation set, tensor of input coordinates
             train_obj: tensor of noisy observations of g(x) for the initialisation set
             optimal_value: global maximiser of f(x)
             best_observed_value: best value of f(x) found in the initialisation set
    """
    # samples for plotting purposes
    plot_sample = torch.linspace(0, 10, 50).reshape(-1, 1)
    # generate training data
    train_x = 10*torch.rand(N_INIT, 1, device=device, dtype=dtype)
    exact_obj, _, optimal_value = exact_sin_objective(train_x, plot_sample, fplot=False)
    train_obj = train_sin_objective(train_x, fplot=False)
    best_observed_value = exact_sin_objective(train_x, plot_sample, fplot=False)[0].max().item()
    return train_x, train_obj, optimal_value, best_observed_value


def initialize_model_ei(train_x, train_obj, state_dict=None):
    """
    Define model for the objective.
    :param train_x: input tensor
    :param train_obj: output tensor -> g(x) + s(x)
    :param state_dict: optional model parameters
    :return:
    """
    model = SingleTaskGP(train_x, train_obj).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


# Noisy Expected Improvement requires FixedNoiseGP for fantasies as explained in the analytic.py module.
NOISE_SE = 0.5
train_yvar = torch.tensor(NOISE_SE ** 2, device=device, dtype=dtype)


def initialize_model_nei(train_x, train_obj, state_dict=None):
    """
    Define model for the objective.
    :param train_x: input tensor
    :param train_obj: output tensor -> g(x) + s(x)
    :param state_dict: optional model parameters
    :return:
    """
    model = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


BATCH_SIZE = 1
bounds = torch.tensor([[0.0] * 1, [10.0] * 1], device=device, dtype=dtype)


def optimize_acqf_and_get_observation(acq_func):
    """
    Optimizes the acquisition function, and returns a new candidate and a noisy observation.

    :param acq_func: Acquisition function to use.
    :return: new_x: new candidate
             new_train_obj: noisy observation of g(x)
    """
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=512,  # used for initialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    new_x = candidates.detach()
    exact_obj, _, _ = exact_sin_objective(new_x, plot_sample=None, fplot=False)
    new_train_obj = train_sin_objective(new_x, fplot=False)
    return new_x, new_train_obj


def update_random_observations(best_random):
    """
    Simulates a random policy by taking a the current list of best values observed randomly,
    drawing a new random point, observing its value, and updating the list.
    :param best_random: list of values observed randomly
    """

    rand_x = 10*torch.rand(BATCH_SIZE, 1)
    next_random_best = exact_sin_objective(rand_x, plot_sample=None, fplot=False)[0].max().item()
    best_random.append(max(best_random[-1], next_random_best))
    return best_random


N_TRIALS = 10
N_BATCH = 20

verbose = False

if __name__ == '__main__':

    best_observed_all_ei, best_observed_all_nei, best_random_all = [], [], []

    # average over multiple trials
    for trial in range(1, N_TRIALS + 1):

        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        best_observed_ei, best_observed_nei, best_random = [], [], []

        # call helper functions to generate initial training data and initialize model
        train_x_ei, train_obj_ei, optimal_value, best_observed_value_ei = generate_initial_data(n=3)
        mll_ei, model_ei = initialize_model_ei(train_x_ei, train_obj_ei)

        train_x_nei, train_obj_nei = train_x_ei, train_obj_ei
        best_observed_value_nei = best_observed_value_ei
        mll_nei, model_nei = initialize_model_nei(train_x_nei, train_obj_nei)

        best_observed_ei.append(best_observed_value_ei)
        best_observed_nei.append(best_observed_value_nei)
        best_random.append(best_observed_value_ei)

        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):

            t0 = time.time()

            # fit the models
            fit_gpytorch_model(mll_ei)
            fit_gpytorch_model(mll_nei)

            # for best_f, we use the best observed noisy values as an approximation
            EI = ExpectedImprovement(
                model=model_ei,
                best_f=(train_obj_ei).max()
            )

            NEI = NoisyExpectedImprovement(
                model=model_nei,
                X_observed=train_x_nei
            )

            # optimize and get new observation
            new_x_ei, new_obj_ei = optimize_acqf_and_get_observation(EI)
            new_x_nei, new_obj_nei = optimize_acqf_and_get_observation(NEI)

            # update training points
            train_x_ei = torch.cat([train_x_ei, new_x_ei])
            train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])

            train_x_nei = torch.cat([train_x_nei, new_x_nei])
            train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])

            # update progress
            best_random = update_random_observations(best_random)
            best_value_ei = exact_sin_objective(train_x_ei, plot_sample=None, fplot=False)[0].max().item()
            best_value_nei = exact_sin_objective(train_x_nei, plot_sample=None, fplot=False)[0].max().item()
            best_observed_ei.append(best_value_ei)
            best_observed_nei.append(best_value_nei)

            # reinitialize the models so they are ready for fitting on next iteration
            # use the current state dict to speed up fitting
            mll_ei, model_ei = initialize_model_ei(
                train_x_ei,
                train_obj_ei,
                model_ei.state_dict(),
            )
            mll_nei, model_nei = initialize_model_nei(
                train_x_nei,
                train_obj_nei,
                model_nei.state_dict(),
            )

            t1 = time.time()

            if verbose:
                print(
                    f"\nBatch {iteration:>2}: best_value (random, qEI, qNEI) = "
                    f"({max(best_random):>4.2f}, {best_value_ei:>4.2f}, {best_value_nei:>4.2f}), "
                    f"time = {t1-t0:>4.2f}.", end=""
                )
            else:
                print(".", end="")

        best_observed_all_ei.append(best_observed_ei)
        best_observed_all_nei.append(best_observed_nei)
        best_random_all.append(best_random)


def ci(y):
    """
    Confidence interval representing the standard error of the mean estimated from N_TRIALS samples.
    :param y: Objective function value (float)
    :return: standard error of the mean objective function value.
    """
    return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)


GLOBAL_MAXIMUM = optimal_value

iters = np.arange(N_BATCH + 1) * BATCH_SIZE
y_ei = np.asarray(best_observed_all_ei)
y_nei = np.asarray(best_observed_all_nei)
y_rnd = np.asarray(best_random_all)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# plt.plot(iters, y_rnd.mean(axis=0), color='r', label='random')
# plt.plot(iters, y_ei.mean(axis=0), color='b', label='EI')
# plt.plot(iters, y_nei.mean(axis=0), color='c', label='NEI')
# plt.fill_between(iters, y_rnd.mean(axis=0) - ci(y_rnd), y_rnd.mean(axis=0) + ci(y_rnd), color='r', alpha=0.1)
# plt.fill_between(iters, y_ei.mean(axis=0) - ci(y_ei), y_ei.mean(axis=0) + ci(y_ei), color='b', alpha=0.1)
# plt.fill_between(iters, y_nei.mean(axis=0) - ci(y_nei), y_nei.mean(axis=0) + ci(y_nei), color='c', alpha=0.1)
ax.errorbar(iters, y_rnd.mean(axis=0), yerr=ci(y_rnd), label="random", linewidth=1.5)
ax.errorbar(iters, y_ei.mean(axis=0), yerr=ci(y_ei), label="EI", linewidth=1.5)
ax.errorbar(iters, y_nei.mean(axis=0), yerr=ci(y_nei), label="NEI", linewidth=1.5)
plt.plot([0, N_BATCH * BATCH_SIZE], [GLOBAL_MAXIMUM] * 2, 'k', label="true best objective", linewidth=2)
ax.set_ylim(bottom=0.15)
ax.set(xlabel='number of observations (beyond initial points)', ylabel='best objective value')
ax.legend(loc="lower right")
plt.show()
plt.savefig('Figures/sin_wave_figs/BayesOpt_{}_trials_{}_init_{}_iters'.format(N_TRIALS, N_INIT, N_BATCH))
