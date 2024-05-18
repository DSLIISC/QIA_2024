# from qiskit_optimization.applications import Knapsack
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.utils import algorithm_globals
from qiskit.algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, ADAM
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit import Aer, QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit import transpile

# Python Libraries
import numpy as np
from numpy import sort
import json
import pandas as pd
import dataframe_image as dfi
import matplotlib.pyplot as plt
import os, sys
import copy
import math

# Qiskit Advanced
from qiskit.primitives import BackendSampler
from qiskit_ibm_provider import IBMProvider
from qiskit.providers.options import Options

from utils import NpEncoder, blockPrint, enablePrint

from docplex.mp.model import Model
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from typing import List, Union
from qiskit_optimization.algorithms import OptimizationResult

class Knapsack():

    def __init__(self, values: List[int], weights: List[int], max_weight: int, instance_name: str) -> None:
        self._values = values
        self._weights = weights
        self._max_weight = max_weight
        self._instance_name = instance_name

    def to_quadratic_program(self) -> QuadraticProgram:
        mdl = Model(name=self._instance_name)
        x = {i: mdl.binary_var(name=f"x_{i}") for i in range(len(self._values))}
        mdl.maximize(mdl.sum(self._values[i] * x[i] for i in x))
        mdl.add_constraint(mdl.sum(self._weights[i] * x[i] for i in x) <= self._max_weight)
        op = from_docplex_mp(mdl)
        return op
        
    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[int]:
        x = self._result_to_x(result)
        return [i for i, value in enumerate(x) if value]

    def _result_to_x(self, result: Union[OptimizationResult, np.ndarray]) -> np.ndarray:
        if isinstance(result, OptimizationResult):
            x = result.x
        elif isinstance(result, np.ndarray):
            x = result
        else:
            print('Unsupported format of result.')
        return x


def solve_problem_qaoa(repeat, p, shots):
    from utils import problems, filepath, backend, backend_name, version

    res_level = [0]
    p_list = list(problems.keys())
    opt_level = [0]

    optimizers = [COBYLA()]
    
    # Create folders
    res_location = os.path.join(filepath, 'p_' + str(p) + '_repeat_' + str(repeat))
    
    try:
        os.makedirs(res_location, exist_ok = True)
        print("Directory '%s' created successfully" %res_location)
    except OSError as error:
        print("Directory '%s' can not be created")
    
    sol_list = list()
    wt_list = list()
    val_list = list()
    shot_list = list()
    status_list = list()
    count_list = list()
    
    final_res = []
    
    for r in range(1, repeat+1):
    
        for shot in shots:
            for rl in res_level:
                for ol in opt_level:
                    for cur_prob in p_list:
                        for optimizer in optimizers:

                            # Init Sampler
                            options = Options()
                            options.resilience_level = rl
                            options.optimization_level = ol
                            options.shots = shot
                            sampler = BackendSampler(backend=backend, options=options)
                            print('Backend:', backend_name)
                            print('Store Location:', filepath)
                            # Init Problem
                            vals = problems[cur_prob]['profits']
                            wgts = problems[cur_prob]['weights']
                            max_wgt = problems[cur_prob]['max_wgt']
                            prob = Knapsack(values=vals, weights=wgts, max_weight=max_wgt, instance_name=problems[cur_prob]['name'])
                            qp = prob.to_quadratic_program()

                            # Persist Intermediate Results
                            counts = list()
                            params = list()
                            values = list()
                            metadata = list()
                            def store_intermediate_result(eval_count, parameters, value, metad):
                                    counts.append(eval_count)
                                    params.append(parameters)
                                    values.append(value)
                                    metadata.append(metad)

                            init_point = [0 for x in range(2*p)]
                            meo = MinimumEigenOptimizer(min_eigen_solver=QAOA(reps=p, sampler=sampler, optimizer=optimizer, 
                                                                  initial_point=init_point, callback=store_intermediate_result))


                            result = meo.solve(qp)

                            # Persist Results
                            opt_details = dict()
                            opt_details['Problem'] = problems[cur_prob]
                            opt_details['Problem_ID'] = cur_prob
                            opt_details['KP_Sol'] = prob.interpret(result)
                            opt_details['KP_Val'] = sum([problems[cur_prob]['profits'][x] for x in prob.interpret(result)])
                            opt_details['KP_Wt'] = sum([problems[cur_prob]['weights'][x] for x in prob.interpret(result)])
                            opt_details['Res_Pretty'] = result.prettyprint()
                            opt_details['Res_Raw'] = str(result)
                            opt_details['Res_Status'] = result.status.name

                            opt_details['Time'] = result.min_eigen_solver_result.optimizer_time
                            opt_details['Resiliency_Level'] = rl
                            opt_details['Optimization_Level'] = ol
                            opt_details['Optimizer'] = type(optimizer).__name__
                            opt_details['Backend'] = backend_name
                            opt_details['Shots'] = shot
                            opt_details['P'] = p


                            # Extract the quantum circuit
                            qc_res = result.min_eigen_solver_result.optimal_circuit.bind_parameters(result.min_eigen_solver_result.optimal_parameters)
                            
                            dp_arr = []
                            qc_res_clone = copy.deepcopy(qc_res)
                            dp_arr.append(qc_res_clone.depth())
                            for tt in range(20):
                                qc_res_clone = qc_res_clone.decompose()
                                dp_arr.append(qc_res_clone.depth())
                              
                            
                            qc_basis = transpile(qc_res, backend)
                            opt_details['Transpile_Depth'] = qc_basis.depth()
                            
                            qc_basis_arr = []
                            qc_basis_arr.append(qc_basis.depth())
                            for tt in range(20):
                                qc_basis = qc_basis.decompose()
                                qc_basis_arr.append(qc_basis.depth())
                            opt_details['Transpile_Depth_Arr'] = qc_basis_arr
                            
                            opt_details['Depth_Arr'] = dp_arr
#                             opt_details['Depth'] = qc_res.decompose().depth()
                            opt_details['Depth'] = qc_basis_arr[1]
                            opt_details['Qubits'] = qc_res.num_qubits
                            opt_details['Energy_Values'] = values
                            opt_details['Count_Values'] = counts
                            opt_details['Param_List'] = str(params)
                            opt_details['Metadata_List'] = metadata
                            opt_details['Total_Iterations'] = len(counts)
                            opt_details['Opt_Params'] = str(result.min_eigen_solver_result.optimal_parameters)
                            opt_details['Opt_Value'] = str(result.min_eigen_solver_result.optimal_value)
                            opt_details['best_measurement'] = str(result.min_eigen_solver_result.best_measurement)
                            opt_details['optimal_point'] = str(result.min_eigen_solver_result.optimal_point)

                            opt_details['IsOptimal'] = sorted(opt_details['KP_Sol']) == sorted(problems[cur_prob]['opt_sol']['opt_index'])

                            # Save Location

                            filename = 'Prob_'+ str(cur_prob) + '_RL_' + str(rl) + '_OL_' + str(ol) \
                                                + '_opt_' + str(opt_details['Optimizer']) + '_bd_' + str(opt_details['Backend']) \
                                                + '_shot_' + str(shot) + '_p_'+ str(p) + '_rep_' + str(r) 
                            # Persist the quantum circuit
                            qc_file = os.path.join(res_location, filename + '_qc.qasm')


                            # Persist Json result
                            json_object = json.dumps(opt_details, indent=4, cls=NpEncoder)
                            res_file = os.path.join(res_location, filename + '_result.json')
                            with open(res_file, "w") as outfile:
                                outfile.write(json_object)

                            print(opt_details)
                            final_res.append(opt_details)
                            print('Iteration Done:' + filename)
                            print('\n****************************************************************************\n')

                            blockPrint()
                            qc_res.qasm(formatted=True, filename=qc_file)
                            enablePrint()

                            sol_list.append(opt_details['KP_Sol'])
                            wt_list.append(opt_details['KP_Wt'])
                            val_list.append(opt_details['KP_Val'])
                            shot_list.append(opt_details['Shots'])
                            status_list.append(opt_details['Res_Status'])
                            count_list.append(len(opt_details['Count_Values']))
        
    return final_res

