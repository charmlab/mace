import gurobipy as grb
import torch

from torch import nn
from torch.autograd import Variable


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

    def get_upper_bound(self, domain):
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

        inps = domain_lb + domain_width * rand_samples

        var_inps = Variable(inps, volatile=True)
        outs = self.net(var_inps)

        upper_bound, idx = torch.min(outs.data, dim=0)

        upper_bound = upper_bound[0]
        ub_point = inps[idx].squeeze()

        return ub_point, upper_bound

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

    def define_linear_approximation(self, input_domain):
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

        ## Do the input layer, which is a special case
        inp_lb = []
        inp_ub = []
        inp_gurobi_vars = []
        for dim, (lb, ub) in enumerate(input_domain):
            v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                  vtype=grb.GRB.CONTINUOUS,
                                  name=f'inp_{dim}')
            inp_gurobi_vars.append(v)
            inp_lb.append(lb)
            inp_ub.append(ub)
        self.model.update()

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
                    ub = layer.bias.data[neuron_idx]
                    lb = layer.bias.data[neuron_idx]
                    lin_expr = layer.bias.data[neuron_idx]
                    for prev_neuron_idx in range(layer.weight.size(1)):
                        coeff = layer.weight.data[neuron_idx, prev_neuron_idx]
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
                    assert self.model.status == 2, "LP wasn't optimally solved"
                    # We have computed a lower bound
                    lb = v.X
                    v.lb = lb

                    # Let's now compute an upper bound
                    self.model.setObjective(v, grb.GRB.MAXIMIZE)
                    self.model.update()
                    self.model.reset()
                    self.model.optimize()
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
