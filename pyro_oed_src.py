import sys
sys.path.append('/Users/ashandonay/Desktop/Research')
sys.path

import torch
import pyro
import pyro.poutine as poutine
from pyro.contrib.util import lexpand
from pyro.infer.autoguide.utils import mean_field_entropy
import math



def posterior_eig(model, design, observation_labels, target_labels, num_samples, num_steps, guide, optim,
                  return_history=False, final_design=None, final_num_samples=None, eig=True, prior_entropy_kwargs={},
                  *args, **kwargs):
                  
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels] # change str to list of string

    if return_history:

        ape, history = _posterior_ape(model, design, observation_labels, target_labels, num_samples, num_steps, guide, optim,
                            return_history=return_history, final_design=final_design, final_num_samples=final_num_samples,
                            *args, **kwargs)
        return _eig_from_ape(model, design, target_labels, ape, eig, prior_entropy_kwargs), history # calculate prior_entropy - ape (if eig=True)
    
    else:
        ape = _posterior_ape(model, design, observation_labels, target_labels, num_samples, num_steps, guide, optim,
                            return_history=return_history, final_design=final_design, final_num_samples=final_num_samples,
                            *args, **kwargs)
        return _eig_from_ape(model, design, target_labels, ape, eig, prior_entropy_kwargs)


# APE: average posterior entropy
def _posterior_ape(model, design, observation_labels, target_labels,
                   num_samples, num_steps, guide, optim, return_history=False,
                   final_design=None, final_num_samples=None, *args, **kwargs):

    loss = _posterior_loss(model, guide, observation_labels, target_labels, *args, **kwargs) # calculate loss
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history,
                            final_design, final_num_samples) # apply steps for optimization

def _posterior_loss(model, guide, observation_labels, target_labels, analytic_entropy=False):
    """Posterior loss: to evaluate directly use `posterior_eig` setting `num_steps=0`, `eig=False`."""

    def loss_fn(design, num_particles, evaluation=False, **kwargs):

        # num_particles = num_samples
        expanded_design = lexpand(design, num_particles) # ???

        # Sample from p(y, theta | d) the model
        trace = poutine.trace(model).get_trace(expanded_design) # trace: graph data structure denoting relationships amongst different pyro primitives
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels} # trace.nodes contains a collection (OrderedDict) of site names and metadata
        theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}

        # Run through q(theta | y, d)
        conditional_guide = pyro.condition(guide, data=theta_dict) # change the sample statements in dictionary into observations with those values
        cond_trace = poutine.trace(conditional_guide).get_trace(
            y_dict, expanded_design, observation_labels, target_labels) 
        cond_trace.compute_log_prob() # compute site-wise log probabilities of each trace. (shape = batch_shape)
        if evaluation and analytic_entropy:
            loss = mean_field_entropy(
                guide, [y_dict, expanded_design, observation_labels, target_labels],
                whitelist=target_labels).sum(0) / num_particles
            agg_loss = loss.sum()
        else:
            terms = -sum(cond_trace.nodes[l]["log_prob"] for l in target_labels) # forward pass through network and evaluate loss func
            agg_loss, loss = _safe_mean_terms(terms)

        return agg_loss, loss

    return loss_fn

def vnmc_eig(
    model,
    design,
    observation_labels,
    target_labels,
    num_samples,
    num_steps,
    guide,
    optim,
    return_history=False,
    final_design=None,
    final_num_samples=None,
    contrastive=False,
):
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    loss = _vnmc_eig_loss(model, guide, observation_labels, target_labels, contrastive)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history, contrastive, final_design, final_num_samples,)

def _vnmc_eig_loss(model, guide, observation_labels, target_labels, contrastive=False):
    """VNMC loss: to evaluate directly use `vnmc_eig` setting `num_steps=0`."""

    def loss_fn(design, num_particles, evaluation=False, **kwargs):
        N, M = num_particles
        expanded_design = lexpand(design, N)

        # Sample from p(y, theta | d)
        trace = poutine.trace(model).get_trace(expanded_design)
        y_dict = {l: lexpand(trace.nodes[l]["value"], M) for l in observation_labels}

        # Sample M times from q(theta | y, d) for each y
        reexpanded_design = lexpand(expanded_design, M)
        conditional_guide = pyro.condition(guide, data=y_dict)
        guide_trace = poutine.trace(conditional_guide).get_trace(
            y_dict, reexpanded_design, observation_labels, target_labels)
        theta_y_dict = {l: guide_trace.nodes[l]["value"] for l in target_labels}
        theta_y_dict.update(y_dict)
        guide_trace.compute_log_prob()

        if contrastive:
            contrastive_samples = torch.cat([lexpand(trace.nodes[l]["value"], 1) for l in target_labels], dim=0)

        # Re-run that through the model to compute the joint
        modelp = pyro.condition(model, data=theta_y_dict)
        model_trace = poutine.trace(modelp).get_trace(reexpanded_design)
        model_trace.compute_log_prob()

        terms = -sum(guide_trace.nodes[l]["log_prob"] for l in target_labels) # q(theta_nm | y_n)
        terms += sum(model_trace.nodes[l]["log_prob"] for l in target_labels) # p(theta_nm)
        terms = terms.squeeze()
        terms += sum(model_trace.nodes[l]["log_prob"] for l in observation_labels) # p(y_n | theta_nm, d)

        # to calculate lower and upper bounds:
        if contrastive:
            # including the original sample theta_0 from which y was sampled to get the lower bound:
            lower_terms = -terms.logsumexp(0) + math.log(M) 
            # returns log summed exponentials log(exp(x_1)+exp(x_2)..) of each row of the input tensor in the given dim (0)
            # excluding the original sample to get the upper bound:
            upper_terms = -terms[1:].logsumexp(0) + math.log(M-1)
        else:
            terms = -terms.logsumexp(0) + math.log(M)

        if evaluation:
            trace.compute_log_prob() # compute likelihood p(y_n | theta_n,0, d)
            conditional_lp = sum(trace.nodes[l]["log_prob"] for l in observation_labels) # p(y_n | theta_n, d)
            
        if contrastive:
            if evaluation:
                lower_terms += conditional_lp
                upper_terms += conditional_lp
            lower_agg_loss, lower_loss = _safe_mean_terms(lower_terms)
            upper_agg_loss, upper_loss = _safe_mean_terms(upper_terms)
            agg_loss = (lower_agg_loss, upper_agg_loss)
            loss = (lower_loss, upper_loss)
        else:
            if evaluation:
                terms += conditional_lp
            agg_loss, loss = _safe_mean_terms(terms)

        return agg_loss, loss
    
    return loss_fn


def opt_eig_ape_loss(design, loss_fn, num_samples, num_steps, optim, return_history=False, contrastive=False,
                     final_design=None, final_num_samples=None):

    if final_design is None:
        final_design = design
    if final_num_samples is None:
        final_num_samples = num_samples

        if contrastive:
            params = None
            history_upper = []
            history_lower = []
            for step in range(num_steps):
                if params is not None:
                    pyro.infer.util.zero_grads(params)
                with poutine.trace(param_only=True) as param_capture:
                    agg_loss, loss = loss_fn(design, num_samples, evaluation=return_history)
                params = set(site["value"].unconstrained()
                            for site in param_capture.trace.nodes.values())
                if torch.isnan(agg_loss[0]) or torch.isnan(agg_loss[1]):
                    raise ArithmeticError("Encountered NaN loss in opt_eig_ape_loss")
                agg_loss[0].backward(retain_graph=True)
                agg_loss[1].backward(retain_graph=True)
                if return_history:
                    history_lower.append(loss[0])
                    history_upper.append(loss[1])
                optim(params)
                try:
                    optim.step()
                except AttributeError:
                    pass

            _, loss = loss_fn(final_design, final_num_samples, evaluation=True, contrastive=True)
            if return_history:
                return loss, torch.stack(history_lower), torch.stack(history_upper)
            else:
                return loss


        else:
            params = None
            history = []
            for step in range(num_steps):
                if params is not None:
                    pyro.infer.util.zero_grads(params)
                with poutine.trace(param_only=True) as param_capture:
                    agg_loss, loss = loss_fn(design, num_samples, evaluation=return_history)
                params = set(
                    site["value"].unconstrained() for site in param_capture.trace.nodes.values()
                )
                if torch.isnan(agg_loss):
                    raise ArithmeticError("Encountered NaN loss in opt_eig_ape_loss")
                agg_loss.backward(retain_graph=True)
                if return_history:
                    history.append(loss)
                optim(params)
                try:
                    optim.step()
                except AttributeError:
                    pass  
        
        
            _, loss = loss_fn(final_design, final_num_samples, evaluation=True, contrastive=False)
            if return_history:
                return loss, torch.stack(history)
            else:
                return loss

def _eig_from_ape(model, design, target_labels, ape, eig, prior_entropy_kwargs):
    mean_field = prior_entropy_kwargs.get("mean_field", True)
    if eig:
        if mean_field:
            try:
                prior_entropy = mean_field_entropy(
                    model, [design], whitelist=target_labels
                )
            except NotImplemented:
                prior_entropy = monte_carlo_entropy(
                    model, design, target_labels, **prior_entropy_kwargs
                )
        else:
            prior_entropy = monte_carlo_entropy(
                model, design, target_labels, **prior_entropy_kwargs
            )
        return prior_entropy - ape
    else:
        return ape


def nmc_eig(
    model,
    design,
    observation_labels,
    target_labels=None,
    N=100,
    M=10,
    M_prime=None,
    independent_priors=False,
    contrastive=False
):
    """
    Based on Pyro implementation from:

    https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/oed/eig.py

    adding an option for contrastive sampling to convert to a lower bound

    """

    if isinstance(observation_labels, str):  # list of strings instead of strings
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    # Take N samples of the model
    expanded_design = lexpand(design, N)  # N copies of the model
    trace = poutine.trace(model).get_trace(expanded_design) # sample y_n and theta_n N times
    trace.compute_log_prob() # compute likelihood p(y_n | theta_n,0, d)

    if M_prime is not None:
        y_dict = {
            l: lexpand(trace.nodes[l]["value"], M_prime) for l in observation_labels
        }
        theta_dict = {
            l: lexpand(trace.nodes[l]["value"], M_prime) for l in target_labels
        }
        theta_dict.update(y_dict)
        # Resample M values of u and compute conditional probabilities
        # WARNING: currently the use of condition does not actually sample
        # the conditional distribution!
        # We need to use some importance weighting
        conditional_model = pyro.condition(model, data=theta_dict)
        if independent_priors:
            reexpanded_design = lexpand(design, M_prime, 1)
        else:
            # Not acceptable to use (M_prime, 1) here - other variables may occur after
            # theta, so need to be sampled conditional upon it
            reexpanded_design = lexpand(design, M_prime, N)
        retrace = poutine.trace(conditional_model).get_trace(reexpanded_design)
        retrace.compute_log_prob()
        conditional_lp = sum(
            retrace.nodes[l]["log_prob"] for l in observation_labels
        ).logsumexp(0) - math.log(M_prime)
    else:
        # This assumes that y are independent conditional on theta
        # Furthermore assume that there are no other variables besides theta

        # sum together likelihood terms for p(y_n|theta_n,0, d)
        conditional_lp = sum(trace.nodes[l]["log_prob"] for l in observation_labels)

    # calculate y_n from the model:
    y_dict = {l: lexpand(trace.nodes[l]["value"], M) for l in observation_labels}
    # Resample M values of theta and compute conditional probabilities
    conditional_model = pyro.condition(model, data=y_dict)
    # Using (M, 1) instead of (M, N) - acceptable to re-use thetas between ys because
    # theta comes before y in graphical model
    reexpanded_design = lexpand(design, M, 1) 
    retrace = poutine.trace(conditional_model).get_trace(reexpanded_design) # sample theta_n,m M times conditioned on y_n
    retrace.compute_log_prob() # compute likelihood p(y_n|theta_n,m, d)
    if not contrastive:
        # sum together likelihood terms for p(y_n|theta_n,m, d) with extra term from 1/M
        marginal_lp = sum(
            retrace.nodes[l]["log_prob"] for l in observation_labels
        ).logsumexp(0) - math.log(M)

        terms = conditional_lp - marginal_lp
        nonnan = (~torch.isnan(terms)).sum(0).type_as(terms)
        terms[torch.isnan(terms)] = 0.0
        return terms.sum(0) / nonnan
    else:
        marginal_log_probs = torch.cat([lexpand(conditional_lp, 1),
                                        sum(retrace.nodes[l]["log_prob"] for l in observation_labels)])
        marginal_lp_lower = marginal_log_probs.logsumexp(0) - math.log(M+1)
        marginal_lp_upper = marginal_log_probs[1:].logsumexp(0) - math.log(M)
        return _safe_mean_terms(conditional_lp - marginal_lp_lower)[1], _safe_mean_terms(conditional_lp - marginal_lp_upper)[1]


def monte_carlo_entropy(model, design, target_labels, num_prior_samples=1000):
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    expanded_design = lexpand(design, num_prior_samples)
    trace = pyro.poutine.trace(model).get_trace(expanded_design)
    trace.compute_log_prob()
    lp = sum(trace.nodes[l]["log_prob"] for l in target_labels)
    return -lp.sum(0) / num_prior_samples

def _safe_mean_terms(terms):
    mask = torch.isnan(terms) | (terms == float("-inf")) | (terms == float("inf"))
    if terms.dtype is torch.float32:
        nonnan = (~mask).sum(0).float()
    elif terms.dtype is torch.float64:
        nonnan = (~mask).sum(0).double()
    terms[mask] = 0.0
    loss = terms.sum(0) / nonnan
    agg_loss = loss.sum()
    return agg_loss, loss

def _create_condition_input(design, y_dict, observation_labels, condition_design=True):
    ys = [design]
    for l in observation_labels:
        if y_dict[l].ndim == design.ndim:
            ys.append(y_dict[l])
        else:
            ys.append(y_dict[l].unsqueeze(dim=-1))

    if condition_design:
        return torch.cat(ys, dim=-1)
    else:
        return torch.cat(ys[1:], dim=-1).float()