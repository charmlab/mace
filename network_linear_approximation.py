import gurobipy as grb
import math
import torch
import numpy as np

from torch import nn
from torch.autograd import Variable

from applyMIPConstraints import *


class View(nn.Module):
    '''
    This is necessary in order to reshape "flat activations" such as used by
    nn.Linear with those that comes from MaxPooling
    '''
    def __init__(self, out_shape):
        super(View, self).__init__()
        self.out_shape = out_shape

    def forward(self, inp):
        # We make the assumption that all the elements in the tuple have
        # the same batchsize and need to be brought to the same size

        # We assume that the first dimension is the batch size
        batch_size = inp.size(0)
        out_size = (batch_size, ) + self.out_shape
        out = inp.view(out_size)
        return out


class LinearizedNetwork:

    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU/MaxPools
        '''
        self.layers = layers
        self.net = nn.Sequential(*layers)
        # Skip all gradient computation for the weights of the Net
        for param in self.net.parameters():
            param.requires_grad = False

    def remove_maxpools(self, domain):
        from plnn.model import reluify_maxpool, simplify_network
        if any(map(lambda x: type(x) is nn.MaxPool1d, self.layers)):
            new_layers = simplify_network(reluify_maxpool(self.layers, domain))
            self.layers = new_layers


    def get_upper_bound_random(self, domain):
        '''
        Compute an upper bound of the minimum of the network on `domain`
        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        '''
        nb_samples = 1024
        nb_inp = domain.size(0)
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        rand_samples = torch.Tensor(nb_samples, nb_inp)
        rand_samples.uniform_(0, 1)

        domain_lb = domain.select(1, 0).contiguous()
        domain_ub = domain.select(1, 1).contiguous()
        domain_width = domain_ub - domain_lb

        domain_lb = domain_lb.view(1, nb_inp).expand(nb_samples, nb_inp)
        domain_width = domain_width.view(1, nb_inp).expand(nb_samples, nb_inp)

        with torch.no_grad():
            inps = domain_lb + domain_width * rand_samples
            outs = self.net(inps)

            upper_bound, idx = torch.min(outs, dim=0)

            upper_bound = upper_bound[0].item()
            ub_point = inps[idx].squeeze()

        return ub_point, upper_bound

    def get_upper_bound_pgd(self, domain):
        '''
        Compute an upper bound of the minimum of the network on `domain`
        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        '''
        nb_samples = 2056
        torch.set_num_threads(1)
        nb_inp = domain.size(0)
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        rand_samples = torch.Tensor(nb_samples, nb_inp)
        rand_samples.uniform_(0, 1)

        best_ub = float('inf')
        best_ub_inp = None

        domain_lb = domain.select(1, 0).contiguous()
        domain_ub = domain.select(1, 1).contiguous()
        domain_width = domain_ub - domain_lb

        domain_lb = domain_lb.view(1, nb_inp).expand(nb_samples, nb_inp)
        domain_width = domain_width.view(1, nb_inp).expand(nb_samples, nb_inp)

        inps = (domain_lb + domain_width * rand_samples)

        with torch.enable_grad():
            batch_ub = float('inf')
            for i in range(1000):
                prev_batch_best = batch_ub

                self.net.zero_grad()
                if inps.grad is not None:
                    inps.grad.zero_()
                inps = inps.detach().requires_grad_()
                out = self.net(inps)

                batch_ub = out.min().item()
                if batch_ub < best_ub:
                    best_ub = batch_ub
                    # print(f"New best lb: {best_lb}")
                    _, idx = out.min(dim=0)
                    best_ub_inp = inps[idx[0]]

                if batch_ub >= prev_batch_best:
                    break

                all_samp_sum = out.sum() / nb_samples
                all_samp_sum.backward()
                grad = inps.grad

                max_grad, _ = grad.max(dim=0)
                min_grad, _ = grad.min(dim=0)
                grad_diff = max_grad - min_grad

                lr = 1e-2 * domain_width / grad_diff
                min_lr = lr.min()

                step = -min_lr*grad
                inps = inps + step

                inps = torch.max(inps, domain_lb)
                inps = torch.min(inps, domain_ub)

        return best_ub_inp, best_ub

    get_upper_bound = get_upper_bound_random

    def get_lower_bound(self, domain):
        '''
        Update the linear approximation for `domain` of the network and use it
        to compute a lower bound on the minimum of the output.
        domain: Tensor containing in each row the lower and upper bound for
                the corresponding dimension
        '''
        self.define_linear_approximation(domain)
        return self.compute_lower_bound(domain)

    def compute_lower_bound(self, domain):
        '''
        Compute a lower bound of the function on `domain`
        Note that this doesn't change the approximation that is made to tailor
        it to `domain`, which would lead to a better approximation.
        domain: Tensor containing in each row the lower and upper bound for the
                corresponding dimension.
        '''
        # We will first setup the appropriate bounds for the elements of the
        # input
        for var_idx, inp_var in enumerate(self.gurobi_vars[0]):
            inp_var.lb = domain[var_idx, 0]
            inp_var.ub = domain[var_idx, 1]

        # We will make sure that the objective function is properly set up
        self.model.setObjective(self.gurobi_vars[-1][0], grb.GRB.MINIMIZE)

        # We will now compute the requested lower bound
        self.model.update()
        self.model.optimize()
        assert self.model.status == 2, "LP wasn't optimally solved"

        return self.gurobi_vars[-1][0].X

    def define_linear_approximation(self, input_domain, factual_sample, dataset_obj, norm_type, norm_lower, norm_upper):
        '''
        input_domain: Tensor containing in each row the lower and upper bound
                      for the corresponding dimension
        '''
        self.lower_bounds = []
        self.upper_bounds = []
        self.gurobi_vars = []
        # These three are nested lists. Each of their elements will itself be a
        # list of the neurons after a layer.

        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.setParam('Threads', 1)
        self.model.setParam('FeasibilityTol', 1e-9)
        self.model.setParam('OptimalityTol', 1e-9)
        self.model.setParam('IntFeassTol', 1e-9)
        self.model.setParam('MIPGap', 0)
        self.model.setParam('MIPGapAbs', 0)

        ## Do the input layer, which is a special case
        inp_lb = []
        inp_ub = []
        inp_gurobi_vars = []
        attr_names = list(factual_sample.keys())
        for dim, (lb, ub) in enumerate(input_domain):
            attr_name = attr_names[dim]
            attr_type = dataset_obj.attributes_kurz[attr_name].attr_type
            if attr_type == 'numeric-real':
                v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                      vtype=grb.GRB.CONTINUOUS,
                                      name=attr_name)
            elif attr_type == 'numeric-int':
                v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                      vtype=grb.GRB.INTEGER,
                                      name=attr_name)
            elif 'cat' in attr_type or 'ord' in attr_type:
                v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                      vtype=grb.GRB.BINARY,
                                      name=attr_name)
            elif attr_type == 'binary':
                v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                      vtype=grb.GRB.BINARY,
                                      name=attr_name)
            else:
                raise Exception(f'Uknonwn attribute type {attr_name}: {attr_type}')
            inp_gurobi_vars.append(v)

        self.model.update()
        if 'obj' not in norm_type:
            applyDistanceConstrs(self.model, dataset_obj, factual_sample, norm_type, norm_lower, norm_upper)
            self.model.update()
        applyPlausibilityConstrs(self.model, dataset_obj)
        self.model.update()

        # Compute new bounds on input w.r.t. distance constraint
        for i, inp_v in enumerate(inp_gurobi_vars):

            # lower bound
            self.model.setObjective(inp_v, grb.GRB.MINIMIZE)
            self.model.update()
            self.model.reset()
            self.model.optimize()
            if self.model.status == 3:  # Infeasible
                return False
            assert self.model.status == 2, "LP wasn't optimally solved"
            lb = inp_v.X
            inp_v.lb = lb

            # upper bound
            self.model.setObjective(inp_v, grb.GRB.MAXIMIZE)
            self.model.update()
            self.model.reset()
            self.model.optimize()
            if self.model.status == 3:  # Infeasible
                return False
            assert self.model.status == 2, "LP wasn't optimally solved"
            ub = inp_v.X
            inp_v.ub = ub

            inp_lb.append(lb)
            inp_ub.append(ub)

        self.model.update()
        self.model.reset()

        self.lower_bounds.append(inp_lb)
        self.upper_bounds.append(inp_ub)
        self.gurobi_vars.append(inp_gurobi_vars)

        ## Do the other layers, computing for each of the neuron, its upper
        ## bound and lower bound
        layer_idx = 1
        for layer in self.layers:
            new_layer_lb = []
            new_layer_ub = []
            new_layer_gurobi_vars = []
            if type(layer) is nn.Linear:
                for neuron_idx in range(layer.weight.size(0)):
                    ub = layer.bias[neuron_idx].item()
                    lb = layer.bias[neuron_idx].item()
                    lin_expr = layer.bias[neuron_idx].item()
                    for prev_neuron_idx in range(layer.weight.size(1)):
                        coeff = layer.weight[neuron_idx, prev_neuron_idx].item()
                        if coeff >= 0:
                            ub += coeff*self.upper_bounds[-1][prev_neuron_idx]
                            lb += coeff*self.lower_bounds[-1][prev_neuron_idx]
                        else:
                            ub += coeff*self.lower_bounds[-1][prev_neuron_idx]
                            lb += coeff*self.upper_bounds[-1][prev_neuron_idx]
                        lin_expr += coeff * self.gurobi_vars[-1][prev_neuron_idx]
                    v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'lay{layer_idx}_{neuron_idx}')
                    self.model.addConstr(v == lin_expr)
                    self.model.update()

                    self.model.setObjective(v, grb.GRB.MINIMIZE)
                    self.model.optimize()
                    if self.model.status == 3:  # Infeasible
                        return False
                    assert self.model.status == 2, "LP wasn't optimally solved"
                    # We have computed a lower bound
                    lb = v.X
                    v.lb = lb

                    # Let's now compute an upper bound
                    self.model.setObjective(v, grb.GRB.MAXIMIZE)
                    self.model.update()
                    self.model.reset()
                    self.model.optimize()
                    if self.model.status == 3:  # Infeasible
                        return False
                    assert self.model.status == 2, "LP wasn't optimally solved"
                    ub = v.X
                    v.ub = ub

                    new_layer_lb.append(lb)
                    new_layer_ub.append(ub)
                    new_layer_gurobi_vars.append(v)
            elif type(layer) == nn.ReLU:
                for neuron_idx, pre_var in enumerate(self.gurobi_vars[-1]):
                    pre_lb = self.lower_bounds[-1][neuron_idx]
                    pre_ub = self.upper_bounds[-1][neuron_idx]

                    v = self.model.addVar(lb=max(0, pre_lb),
                                          ub=max(0, pre_ub),
                                          obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'ReLU{layer_idx}_{neuron_idx}')
                    if pre_lb >= 0 and pre_ub >= 0:
                        # The ReLU is always passing
                        self.model.addConstr(v == pre_var)
                        lb = pre_lb
                        ub = pre_ub
                    elif pre_lb <= 0 and pre_ub <= 0:
                        lb = 0
                        ub = 0
                        # No need to add an additional constraint that v==0
                        # because this will be covered by the bounds we set on
                        # the value of v.
                    else:
                        lb = 0
                        ub = pre_ub
                        self.model.addConstr(v >= pre_var)

                        slope = pre_ub / (pre_ub - pre_lb)
                        bias = - pre_lb * slope
                        self.model.addConstr(v <= slope * pre_var + bias)

                    new_layer_lb.append(lb)
                    new_layer_ub.append(ub)
                    new_layer_gurobi_vars.append(v)
            elif type(layer) == nn.MaxPool1d:
                assert layer.padding == 0, "Non supported Maxpool option"
                assert layer.dilation == 1, "Non supported MaxPool option"
                nb_pre = len(self.gurobi_vars[-1])
                window_size = layer.kernel_size
                stride = layer.stride

                pre_start_idx = 0
                pre_window_end = pre_start_idx + window_size

                while pre_window_end <= nb_pre:
                    lb = max(self.lower_bounds[-1][pre_start_idx:pre_window_end])
                    ub = max(self.upper_bounds[-1][pre_start_idx:pre_window_end])

                    neuron_idx = pre_start_idx // stride

                    v = self.model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS,
                                          name=f'Maxpool{layer_idx}_{neuron_idx}')
                    all_pre_var = 0
                    for pre_var in self.gurobi_vars[-1][pre_start_idx:pre_window_end]:
                        self.model.addConstr(v >= pre_var)
                        all_pre_var += pre_var
                    all_lb = sum(self.lower_bounds[-1][pre_start_idx:pre_window_end])
                    max_pre_lb = lb
                    self.model.addConstr(all_pre_var >= v + all_lb - max_pre_lb)

                    pre_start_idx += stride
                    pre_window_end = pre_start_idx + window_size

                    new_layer_lb.append(lb)
                    new_layer_ub.append(ub)
                    new_layer_gurobi_vars.append(v)
            elif type(layer) == View:
                continue
            else:
                raise NotImplementedError

            self.lower_bounds.append(new_layer_lb)
            self.upper_bounds.append(new_layer_ub)
            self.gurobi_vars.append(new_layer_gurobi_vars)

            layer_idx += 1

        # Assert that this is as expected a network with a single output
        assert len(self.gurobi_vars[-1]) == 1, "Network doesn't have scalar output"

        self.model.update()

        return True
