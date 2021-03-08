import numpy as np
import scipy.stats

from .pareto import Pareto


def score(mode, predictor_list, test, training_list, **kwargs):
    if test.X.shape[0] == 0:
        return np.zeros(0)

    if mode == "EHVI":
        pareto = kwargs["pareto"]
        fmean, fstd = _get_fmean_fstd(predictor_list, training_list, test)
        f = EHVI(fmean, fstd, pareto)
    elif mode == "HVPI":
        pareto = kwargs["pareto"]
        fmean, fstd = _get_fmean_fstd(predictor_list, training_list, test)
        f = HVPI(fmean, fstd, pareto)
    elif mode == "TS":
        alpha = kwargs.get("alpha", 1.0)
        reduced_candidate_num = kwargs["reduced_candidate_num"]
        f = TS(
            predictor_list,
            training_list,
            test,
            alpha,
            reduced_candidate_num=reduced_candidate_num,
        )
    else:
        raise NotImplementedError("mode must be EHVI, HVPI or TS.")
    return f

def HVPI(fmean, fstd, pareto):
    """
    Calculate Hypervolume-based Probability of Improvement (HVPI).

    Reference: (Couckuyt et al., 2014) Fast calculation of multiobjective probability of improvement and expected improvement criteria for Pareto optimization
    """

    N = fmean.shape[0]
    n_obj = pareto.num_objectives

    if pareto.reference_min is None:
        pareto.set_reference_min()
    reference_min = pareto.reference_min

    # Pareto front with reference points
    # shape: (front_size, n_obj)
    front = np.r_[
        np.array(reference_min).reshape((1, n_obj)),
        pareto.front,
        np.full((1, n_obj), np.inf),
    ]

    ax = np.arange(n_obj)
    n_cell = pareto.cells.lb.shape[0]

    # convert to minimization problem
    l = front[pareto.cells.ub, ax].reshape((1, n_cell, n_obj)) * -1
    u = front[pareto.cells.lb, ax].reshape((1, n_cell, n_obj)) * -1

    # convert to minimization problem
    fmean = fmean.reshape((N, 1, n_obj)) * -1
    fstd = fstd.reshape((N, 1, n_obj))

    # calculate cdf
    Phi_l = scipy.stats.norm.cdf((l - fmean) / fstd)
    Phi_u = scipy.stats.norm.cdf((u - fmean) / fstd)

    #  calculate PoI
    poi = np.sum(np.prod(Phi_u - Phi_l, axis=2), axis=1)  # shape: (N, 1)

    # calculate hypervolume contribution of fmean point
    hv_valid = np.all(fmean < u, axis=2)  # shape: (N, n_cell)
    hv = np.prod(u - np.maximum(l, fmean), axis=2)  # shape: (N, n_cell)
    hv = np.sum(hv * hv_valid, axis=1)  # shape: (N, 1)

    # HVPoI
    score = hv * poi
    return score


def EHVI(fmean, fstd, pareto):
    """
    Calculate Expected Hyper-Volume Improvement (EHVI).

    Reference: (Couckuyt et al., 2014) Fast calculation of multiobjective probability of improvement and expected improvement criteria for Pareto optimization
    """

    N = fmean.shape[0]
    n_obj = pareto.num_objectives

    if pareto.reference_min is None:
        pareto.set_reference_min()
    if pareto.reference_max is None:
        pareto.set_reference_max()
    reference_min = pareto.reference_min
    reference_max = pareto.reference_max

    # Pareto front with reference points
    # shape: (front_size, n_obj)
    front = np.r_[
        np.array(reference_min).reshape((1, n_obj)),
        pareto.front,
        np.array(reference_max).reshape((1, n_obj)),
    ]

    ax = np.arange(n_obj)

    # convert to minimization problem
    l = front[pareto.cells.ub, ax] * -1
    u = front[pareto.cells.lb, ax] * -1

    n_cell = pareto.cells.lb.shape[0]

    # shape: (n_cell, 1, n_cell, n_obj)
    l = np.tile(l, (n_cell, 1, 1, 1))
    u = np.tile(u, (n_cell, 1, 1, 1))
    a = l.transpose((2, 1, 0, 3))
    b = u.transpose((2, 1, 0, 3))

    # convert to minimization problem
    fmean = fmean.reshape((1, N, 1, n_obj)) * -1
    fstd = fstd.reshape((1, N, 1, n_obj))

    # calculate pdf, cdf
    phi_min_bu = scipy.stats.norm.pdf((np.minimum(b, u) - fmean) / fstd)
    phi_max_al = scipy.stats.norm.pdf((np.maximum(a, l) - fmean) / fstd)
    Phi_l = scipy.stats.norm.cdf((l - fmean) / fstd)
    Phi_u = scipy.stats.norm.cdf((u - fmean) / fstd)
    Phi_a = scipy.stats.norm.cdf((a - fmean) / fstd)
    Phi_b = scipy.stats.norm.cdf((b - fmean) / fstd)

    ## calculate G
    is_type_A = np.logical_and(a < u, l < b)
    is_type_B = u <= a

    # note: Phi[max_or_min(x,y)] = max_or_min(Phi[x], Phi[y])
    EI_A = (
        (b - a) * (np.maximum(Phi_a, Phi_l) - Phi_l)
        + (b - fmean) * (np.minimum(Phi_b, Phi_u) - np.maximum(Phi_a, Phi_l))
        + fstd * (phi_min_bu - phi_max_al)
    )
    EI_B = (b - a) * (Phi_u - Phi_l)

    G = EI_A * is_type_A + EI_B * is_type_B
    score = np.sum(np.sum(np.prod(G, axis=3), axis=0), axis=1)  # shape: (N, 1)
    return score


def TS(predictor_list, training_list, test, alpha=1, reduced_candidate_num=None):
    score = [
        predictor.get_post_samples(training, test, alpha=alpha)
        for predictor, training in zip(predictor_list, training_list)
    ]
    score = np.array(score).reshape((len(predictor_list), test.X.shape[0])).T
    pareto = Pareto(num_objectives=len(predictor_list))

    if reduced_candidate_num is None or score.shape[0] <= reduced_candidate_num:
        use_idx = np.arange(score.shape[0])
    else:
        use_idx = np.arange(reduced_candidate_num)
        use_idx = np.random.choice(use_idx, reduced_candidate_num, replace=False)

    # pareto.update_front(score)
    pareto.update_front(score[use_idx, :])

    # randomly choose candidate from pareto frontiers
    chosen_idx = np.random.choice(pareto.front_num)
    score_res = np.zeros(score.shape[0])
    score_res[use_idx[chosen_idx]] = 1  # only chosen_idx th value is one.

    return score_res


def _get_fmean_fstd(predictor_list, training_list, test):
    fmean = [
        predictor.get_post_fmean(training, test)
        for predictor, training in zip(predictor_list, training_list)
    ]
    fcov = [
        predictor.get_post_fcov(training, test)
        for predictor, training in zip(predictor_list, training_list)
    ]

    # shape: (N, n_obj)
    fmean = np.array(fmean).T
    fstd = np.sqrt(np.array(fcov)).T
    return fmean, fstd
