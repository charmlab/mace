import gurobipy as grb
import numpy as np
import torch

from torch import nn
# from plnn.dual_network_linear_approximation import LooseDualNetworkApproximation
from network_linear_approximation import LinearizedNetwork
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


class MIPNetwork:

    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU/MaxPools
        '''
        self.layers = layers
        self.net = nn.Sequential(*layers)

        # Initialize a LinearizedNetwork object to determine the lower and
        # upper bounds at each layer.
        self.lin_net = LinearizedNetwork(layers)

    def solve(self, inp_domain, factual_sample, timeout=None):
        '''
        inp_domain: Tensor containing in each row the lower and upper bound
                    for the corresponding dimension
        Returns:
        sat     : boolean indicating whether the MIP is satisfiable.
        solution: Feasible point if the MIP is satisfiable,
                  None otherwise.
        timeout : Maximum allowed time to run, if is not None
        '''

        if factual_sample['y'] is True and self.lower_bounds[-1][0] > 0:
            # The problem is infeasible, and we haven't setup the MIP
            return (False, None, 0)
        elif factual_sample['y'] is False and self.upper_bounds[-1][0] < 0:
            # The problem is infeasible, and we haven't setup the MIP
            return (False, None, 0)

        if timeout is not None:
            self.model.setParam('TimeLimit', timeout)

        if self.check_obj_value_callback:
            def early_stop_cb(model, where):
                if where == grb.GRB.Callback.MIP:
                    best_bound = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
                    if factual_sample['y'] is True and best_bound >= 0:
                        model.terminate()
                    if factual_sample['y'] is False and best_bound <= 0:
                        model.terminate()

                if where == grb.GRB.Callback.MIPNODE:
                    nodeCount = model.cbGet(grb.GRB.Callback.MIPNODE_NODCNT)
                    # if (nodeCount % 100) == 0:
                    #     print(f"Running Nb states visited: {nodeCount}")

                if where == grb.GRB.Callback.MIPSOL:
                    obj = model.cbGet(grb.GRB.Callback.MIPSOL_OBJ)
                    if factual_sample['y'] is True and obj <= 0:
                        # Does it have a chance at being a valid
                        # counter-example?

                        # Check it with the network
                        input_vals = model.cbGetSolution(self.gurobi_vars[0])

                        with torch.no_grad():
                            inps = torch.tensor(input_vals, dtype=torch.float64).view(1, -1)
                            out = self.net(inps).squeeze().item()

                        if out <= 0:
                            model.terminate()

                    if factual_sample['y'] is False and obj >= 0:
                        # Does it have a chance at being a valid
                        # counter-example?

                        # Check it with the network
                        input_vals = model.cbGetSolution(self.gurobi_vars[0])

                        with torch.no_grad():
                            inps = torch.tensor(input_vals, dtype=torch.float64).view(1, -1)
                            out = self.net(inps).squeeze().item()

                        if out >= 0:
                            model.terminate()
        else:
            def early_stop_cb(model, where):
                if where == grb.GRB.Callback.MIPNODE:
                    nodeCount = model.cbGet(grb.GRB.Callback.MIPNODE_NODCNT)
                    # if (nodeCount % 100) == 0:
                    #     print(f"Running Nb states visited: {nodeCount}")

        self.model.optimize(early_stop_cb)
        nb_visited_states = self.model.nodeCount

        if self.model.status is grb.GRB.INFEASIBLE:
            # Infeasible: No solution
            print("-------INFEASIBLE")
            return (False, None, nb_visited_states)
        elif self.model.status is grb.GRB.OPTIMAL:
            # There is a feasible solution. Return the feasible solution as well.
            len_inp = len(self.gurobi_vars[0])

            # Get the input that gives the feasible solution.
            cfe = {}
            for idx, var in enumerate(self.gurobi_vars[0]):
                cfe[var.varName] = var.x
            optim_val = self.gurobi_vars[-1][-1].x
            cfe['y'] = not(factual_sample['y']) if optim_val == 0 else (optim_val > 0)

            # print("-----OPTIMAL")
            # # Check it with the network
            # input_vals = [var.x for var in self.gurobi_vars[0]]
            # with torch.no_grad():
            #     inps = torch.Tensor(input_vals).view(1, -1)
            #     # print(inps)
            #     out = self.net(inps).squeeze().item()
            # print("torch out: ", out)
            # print("optimal val: ", optim_val)
            # print("optimal distance: ", self.model.getVarByName('normalized_distance').x)

            return (cfe['y'] != factual_sample['y'], cfe, nb_visited_states)

        elif self.model.status is grb.GRB.INTERRUPTED:
            obj_bound = self.model.ObjBound

            if factual_sample['y'] is True and obj_bound > 0:
                return (False, None, nb_visited_states)
            elif factual_sample['y'] is False and obj_bound < 0:
                return (False, None, nb_visited_states)
            else:
                # There is a feasible solution. Return the feasible solution as well.
                len_inp = len(self.gurobi_vars[0])

                # Get the input that gives the feasible solution.
                cfe = {}
                for idx, var in enumerate(self.gurobi_vars[0]):
                    cfe[var.varName] = var.x
                optim_val = self.gurobi_vars[-1][-1].x
                cfe['y'] = not(factual_sample['y']) if optim_val == 0 else (optim_val > 0)

            # print("------INTERUPTED")
            # # Check it with the network
            # input_vals = [var.x for var in self.gurobi_vars[0]]
            # with torch.no_grad():
            #     inps = torch.Tensor(input_vals).view(1, -1)
            #     print(inps)
            #     out = self.net(inps).squeeze().item()
            # print("torch out: ", out)
            # print("optimal val: ", optim_val)
            # print("optimal distance: ", self.model.getVarByName('normalized_distance').x)

            return (cfe['y'] != factual_sample['y'], cfe, nb_visited_states)

        elif self.model.status is grb.GRB.TIME_LIMIT:
            # We timed out, return a None Status
            return (None, None, nb_visited_states)
        else:
            raise Exception("Unexpected Status code")

    def tune(self, param_outfile, tune_timeout):
        self.model.Params.tuneOutput = 1
        self.model.Params.tuneTimeLimit = tune_timeout
        self.model.tune()

        # Get the best set of parameters
        self.model.getTuneResult(0)

        self.model.write(param_outfile)

    def do_interval_analysis(self, inp_domain):
        self.lower_bounds = []
        self.upper_bounds = []

        inp_lb = []
        inp_ub = []

        self.lower_bounds.append(inp_domain[:, 0])
        self.upper_bounds.append(inp_domain[:, 1])

        layer_idx = 1
        for layer in self.layers:
            new_layer_lb = []
            new_layer_ub = []
            if type(layer) is nn.Linear:
                pos_weights = torch.clamp(layer.weight, min=0)
                neg_weights = torch.clamp(layer.weight, max=0)

                new_layer_lb = torch.mv(pos_weights, self.lower_bounds[-1]) + \
                               torch.mv(neg_weights, self.upper_bounds[-1]) + \
                               layer.bias
                new_layer_ub = torch.mv(pos_weights, self.upper_bounds[-1]) + \
                               torch.mv(neg_weights, self.lower_bounds[-1]) + \
                               layer.bias
            elif type(layer) == nn.ReLU:
                new_layer_lb = torch.clamp(self.lower_bounds[-1], min=0)
                new_layer_ub = torch.clamp(self.upper_bounds[-1], min=0)
            elif type(layer) == nn.MaxPool1d:
                assert layer.padding == 0, "Non supported Maxpool option"
                assert layer.dilation == 1, "Non supported Maxpool option"

                nb_pre = len(self.lower_bounds[-1])
                window_size = layer.kernel_size
                stride = layer.stride

                pre_start_idx = 0
                pre_window_end = pre_start_idx + window_size

                while pre_window_end <= nb_pre:
                    lb = max(self.lower_bounds[-1][pre_start_idx:pre_window_end])
                    ub = max(self.upper_bounds[-1][pre_start_idx:pre_window_end])

                    new_layer_lb.append(lb)
                    new_layer_ub.append(ub)

                    pre_start_idx += stride
                    pre_window_end = pre_start_idx + window_size
                new_layer_lb = torch.tensor(new_layer_lb, dtype=torch.float64)
                new_layer_ub = torch.tensor(new_layer_ub, dtype=torch.float64)
            elif type(layer) == View:
                continue
            else:
                raise NotImplementedError()

            self.lower_bounds.append(new_layer_lb)
            self.upper_bounds.append(new_layer_ub)

            layer_idx += 1


    def setup_model(self, inp_domain, factual_sample, dataset_obj, norm_type, norm_lower, norm_upper, epsilon,
                    sym_bounds=False,
                    dist_as_constr=False,
                    bounds="opt",
                    parameter_file=None):
        '''
        inp_domain: Tensor containing in each row the lower and upper bound
                    for the corresponding dimension
        optimal: If False, don't use any objective function, simply add a constraint on the output
                 If True, perform optimization and use callback to interrupt the solving when a
                          counterexample is found
        bounds: string, indicate what type of method should be used to get the intermediate bounds
        parameter_file: Load a set of parameters for the MIP solver if a path is given.
        Setup the model to be optimized by Gurobi
        '''
        if bounds == "opt":
            # First use define_linear_approximation from LinearizedNetwork to
            # compute upper and lower bounds to be able to define Ms
            feasible = self.lin_net.define_linear_approximation(inp_domain, factual_sample, dataset_obj, norm_type, norm_lower, norm_upper)

            self.lower_bounds = list(map(torch.tensor, self.lin_net.lower_bounds))
            self.upper_bounds = list(map(torch.tensor, self.lin_net.upper_bounds))
        elif bounds == "interval":
            self.do_interval_analysis(inp_domain)
            if self.lower_bounds[-1][0] > 0:
                # The problem is already guaranteed to be infeasible,
                # Let's not waste time setting up the MIP
                return
        elif bounds == "interval-kw":
            self.do_interval_analysis(inp_domain)
            kw_dual = LooseDualNetworkApproximation(self.layers)
            kw_dual.remove_maxpools(inp_domain, no_opt=True)
            lower_bounds, upper_bounds = kw_dual.get_intermediate_bounds(inp_domain)

            # We want to get the best out of interval-analysis and K&W

            # TODO: There is a slight problem. To use the K&W code directly, we
            # need to make a bunch of changes, notably remove all of the
            # Maxpooling and convert them to ReLUs. Quick and temporary fix:
            # take the max of both things if the shapes are all the same so
            # far, and use the one from interval analysis after the first
            # difference.

            # If the network are full ReLU, there should be no problem.
            # If the network are just full ReLU with a MaxPool at the end,
            # that's still okay because we get the best bounds until the
            # maxpool, and that's the last thing that we use the bounds for
            # This is just going to suck if we have a Maxpool early in the
            # network, and even then, that just means we use interval analysis
            # so stop complaining.
            for i in range(len(lower_bounds)):
                if lower_bounds[i].shape == self.lower_bounds[i].shape:
                    # Keep the best lower bound
                    torch.max(lower_bounds[i], self.lower_bounds[i], out=self.lower_bounds[i])
                    torch.min(upper_bounds[i], self.upper_bounds[i], out=self.upper_bounds[i])
                else:
                    # Mismatch in dimension.
                    # Drop it and stop trying to improve the stuff of interval analysis
                    break
            if self.lower_bounds[-1][0] > 0:
                # The problem is already guaranteed to be infeasible,
                # Let's not waste time setting up the MIP
                return
        else:
            raise NotImplementedError("Unknown bound computation method.")

        if not feasible:
            return False

        self.gurobi_vars = []
        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.setParam('Threads', 1)
        self.model.setParam('DualReductions', 0)
        if 'two_norm' in norm_type:
            self.model.setParam('NonConvex', 2)
        if 'obj' in norm_type:
            self.model.setParam('OptimalityTol', epsilon)
        # self.model.setParam('FeasibilityTol', 1e-9)
        # self.model.setParam('OptimalityTol', 1e-9)
        # self.model.setParam('IntFeassTol', 1e-9)
        # self.model.setParam('MIPGap', 0)
        # self.model.setParam('MIPGapAbs', 0)
        self.model.update()

        if parameter_file is not None:
            self.model.read(parameter_file)

        # First add the input variables as Gurobi variables.
        inp_gurobi_vars = []
        attr_names = list(factual_sample.keys())
        for dim in range(len(inp_domain)):

            attr_name = attr_names[dim]
            attr_type = dataset_obj.attributes_kurz[attr_name].attr_type
            lb = self.lower_bounds[0][dim]
            ub = self.upper_bounds[0][dim]

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

        self.gurobi_vars.append(inp_gurobi_vars)
        self.model.update()
        applyDistanceConstrs(self.model, dataset_obj, factual_sample, norm_type, norm_lower, norm_upper)
        self.model.update()
        applyPlausibilityConstrs(self.model, dataset_obj)
        self.model.update()

        layer_idx = 1
        for layer in self.layers:
            new_layer_gurobi_vars = []
            if type(layer) is nn.Linear:
                for neuron_idx in range(layer.weight.size(0)):
                    lin_expr = layer.bias[neuron_idx].item()
                    for prev_neuron_idx_ten in torch.nonzero(layer.weight[neuron_idx]):
                        prev_neuron_idx = prev_neuron_idx_ten[0]
                        coeff = layer.weight[neuron_idx, prev_neuron_idx].item()
                        lin_expr += coeff * self.gurobi_vars[-1][prev_neuron_idx]
                    # TODO: use lin_net bounds
                    v = self.model.addVar(lb=-grb.GRB.INFINITY,
                                          ub=grb.GRB.INFINITY,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'lin_v_{layer_idx}_{neuron_idx}')
                    self.model.addConstr(v == lin_expr)
                    self.model.update()

                    # We are now done with this neuron.
                    new_layer_gurobi_vars.append(v)

            elif type(layer) == nn.ReLU:

                for neuron_idx, pre_var in enumerate(self.gurobi_vars[-1]):
                    pre_lb = self.lower_bounds[layer_idx-1][neuron_idx].item()
                    pre_ub = self.upper_bounds[layer_idx-1][neuron_idx].item()

                    # Use the constraints specified by
                    # Verifying Neural Networks with Mixed Integer Programming
                    # MIP formulation of ReLU:
                    #
                    # x = max(pre_var, 0)
                    #
                    # Introduce binary variable b, such that:
                    # b = 1 if inp is the maximum value, 0 otherwise
                    #
                    # We know the lower (pre_lb) and upper bounds (pre_ub) for pre_var
                    # We can thus write the following:
                    #
                    # MIP must then satisfy the following constraints:
                    # Constr_13: x <= pre_var - pre_lb (1-b)
                    # Constr_14: x >= pre_var
                    # Constr_15: x <= b* pre_ub
                    # Constr_16: x >= 0

                    if sym_bounds:
                        # We're going to use the big-M encoding of the other papers.
                        M = max(-pre_lb, pre_ub)
                        pre_lb = -M
                        pre_ub = M

                    if pre_lb <= 0 and pre_ub <=0:
                        # x = self.model.addVar(lb=0, ub=0,
                        #                       vtype=grb.GRB.CONTINUOUS,
                        #                       name = f'ReLU_x_{layer_idx}_{neuron_idx}')
                        x = 0
                    elif (pre_lb >= 0) and (pre_ub >= 0):
                        # x = self.model.addVar(lb=pre_lb, ub=pre_ub,
                        #                       vtype=grb.GRB.CONTINUOUS,
                        #                       name = f'ReLU_x_{layer_idx}_{neuron_idx}')
                        # self.model.addConstr(x == pre_var, f'constr_{layer_idx}_{neuron_idx}_fixedpassing')
                        x = pre_var
                    else:
                        x = self.model.addVar(lb=0,
                                              ub=grb.GRB.INFINITY,
                                              vtype=grb.GRB.CONTINUOUS,
                                              name = f'ReLU_x_{layer_idx}_{neuron_idx}')
                        b = self.model.addVar(vtype=grb.GRB.BINARY,
                                              name= f'ReLU_b_{layer_idx}_{neuron_idx}')

                        self.model.addConstr(x <= pre_var - pre_lb * (1-b), f'constr_{layer_idx}_{neuron_idx}_c13')
                        self.model.addConstr(x >= pre_var, f'constr_{layer_idx}_{neuron_idx}_c14')
                        self.model.addConstr(x <= b * pre_ub, f'constr_{layer_idx}_{neuron_idx}_c15')
                        # self.model.addConstr(x >= 0, f'constr_{layer_idx}_{neuron_idx}_c16')
                        # (implied already by bound on x)

                    self.model.update()

                    new_layer_gurobi_vars.append(x)
            elif type(layer) == nn.MaxPool1d:
                assert layer.padding == 0, "Non supported Maxpool option"
                assert layer.dilation == 1, "Non supported MaxPool option"
                nb_pre = len(self.gurobi_vars[-1])
                window_size = layer.kernel_size
                stride = layer.stride

                pre_start_idx = 0
                pre_window_end = pre_start_idx + window_size

                while pre_window_end <= nb_pre:
                    ub_max = max(self.upper_bounds[layer_idx-1][pre_start_idx:pre_window_end]).item()
                    window_bin_vars = []
                    neuron_idx = pre_start_idx % stride
                    v = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                          lb=-grb.GRB.INFINITY,
                                          ub=grb.GRB.INFINITY,
                                          name=f'MaxPool_out_{layer_idx}_{neuron_idx}')
                    for pre_var_idx, pre_var in enumerate(self.gurobi_vars[-1][pre_start_idx:pre_window_end]):
                        lb = self.lower_bounds[layer_idx-1][pre_start_idx + pre_var_idx].item()
                        b = self.model.addVar(vtype=grb.GRB.BINARY,
                                              name= f'MaxPool_b_{layer_idx}_{neuron_idx}_{pre_var_idx}')
                        # MIP formulation of max pooling:
                        #
                        # y = max(x_1, x_2, ..., x_n)
                        #
                        # Introduce binary variables d_1, d_2, ..., d_n:
                        # d_i = i if x_i is the maximum value, 0 otherwise
                        #
                        # We know the lower (l_i) and upper bounds (u_i) for x_i
                        #
                        # Denote the maximum of the upper_bounds of all inputs x_i as u_max
                        #
                        # MIP must then satisfy the following constraints:
                        #
                        # Constr_1: l_i <= x_i <= u_i
                        # Constr_2: y >= x_i
                        # Constr_3: y <= x_i + (u_max - l_i)*(1 - d_i)
                        # Constr_4: sum(d_1, d_2, ..., d_n) = 1

                        # Constr_1 is already satisfied due to the implementation of LinearizedNetworks.
                        # Constr_2
                        self.model.addConstr(v >= pre_var)
                        # Constr_3
                        self.model.addConstr(v <= pre_var + (ub_max - lb)*(1-b))

                        window_bin_vars.append(b)
                    # Constr_4
                    self.model.addConstr(sum(window_bin_vars) == 1)
                    self.model.update()
                    pre_start_idx += stride
                    pre_window_end = pre_start_idx + window_size
                    new_layer_gurobi_vars.append(v)
            elif type(layer) == View:
                continue
            else:
                raise NotImplementedError

            self.gurobi_vars.append(new_layer_gurobi_vars)
            layer_idx += 1
        # Assert that this is as expected: a network with a single output
        assert len(self.gurobi_vars[-1]) == 1, "Network doesn't have scalar output"

        # Add the final constraint that the output must be less than or equal
        # to zero.
        if not dist_as_constr:
            # TODO: This tolerance is quite sensitive, if there are assertion erros it means that this should be increased
            if factual_sample['y'] is True:
                self.model.addConstr(self.gurobi_vars[-1][-1] <= -1e-3)
            else:
                self.model.addConstr(self.gurobi_vars[-1][-1] >= 1e-3)

            self.model.setObjective(self.model.getVarByName('normalized_distance'), grb.GRB.MINIMIZE)
            self.check_obj_value_callback = False
        else:
            if factual_sample['y'] is True:
                self.model.setObjective(self.gurobi_vars[-1][-1], grb.GRB.MINIMIZE)
            else:
                self.model.setObjective(self.gurobi_vars[-1][-1], grb.GRB.MAXIMIZE)
            self.check_obj_value_callback = True

        # Optimize the model.
        self.model.update()

        return True