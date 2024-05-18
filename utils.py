# Python Libraries
import numpy as np
from numpy import sort
import json
import pandas as pd
import dataframe_image as dfi
import matplotlib.pyplot as plt
import os, sys
import copy
import imgkit
from math import log, floor, ceil, sqrt

# Qiskit Advanced
from qiskit.primitives import BackendSampler
from qiskit_ibm_provider import IBMProvider
from qiskit.providers.options import Options
from qiskit.providers.fake_provider import FakeAuckland
from qiskit import Aer, QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService

# Global Variables
is_Noisy = None
backend = None
backend_name = None
is_local = None
filepath = None
version = None
problems = dict()

# Disable Print
def blockPrint():
    sys.__stdout__ = sys.stdout
    sys.stdout = open(os.devnull, 'w')

# Restore Print
def enablePrint():
    sys.stdout = sys.__stdout__
    
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.complex128):
            return obj.real
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def init_global_param(noisy, base_path):
    global is_Noisy, backend, backend_name, filepath
    
    is_Noisy = noisy
                                           
    
    if is_Noisy:
        # backend = FakeAuckland()
        # backend_name = backend.backend_name
        service = QiskitRuntimeService(channel="ibm_quantum")
        backend = service.backend("ibm_brisbane")
        backend_name = backend.name
    else:
        backend = Aer.get_backend('qasm_simulator')
        backend_name = backend.name()
        # backend = service.backend("ibm_brisbane")
        # backend_name = backend.name
        

    # backend.set_options(max_parallel_threads = 0)
    
    if is_Noisy:
        filepath = os.path.join(base_path, 'noisy')
    else:
        filepath = os.path.join(base_path, 'noiseless')
    
    
    try:
        os.makedirs(filepath, exist_ok = True)
        print("Directory '%s' created successfully" %filepath)
    except OSError as error:
        print("Directory '%s' can not be created")
        
    print('Backend:', backend_name, ' Filepath:', filepath)




def populate_prob_instance(inst_obj):

    global problems
    problems[1] = inst_obj # Adding the problem instance in the problems dict
    return problems


def opt_knapSack(W, wt, val, n):
    final_list = []
    final_val = 0
    K = [[0 for w in range(W + 1)]
            for i in range(n + 1)]
             
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1]
                  + K[i - 1][w - wt[i - 1]],
                               K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    res = K[n][W]
    final_val = res
     
    w = W
    for i in range(n, 0, -1):
        if res <= 0:
            break

        if res == K[i - 1][w]:
            continue
        else:
 

            final_list.append(i-1)

            res = res - val[i - 1]
            w = w - wt[i - 1]
            
    return final_val, final_list


def add_optimal():
    global problems
    
    for p in problems.keys():
        opt_val, opt_in = opt_knapSack(problems[p]['max_wgt'], problems[p]['weights'], problems[p]['profits'], len(problems[p]['profits']))
        opt_val_dict = dict()
        
        opt_val_dict['opt_val'] = opt_val
        opt_val_dict['opt_index'] = sorted(opt_in)
        opt_val_dict['opt_wt'] = sum([problems[p]['weights'][x] for x in opt_val_dict['opt_index']])
        problems[p]['opt_sol'] = opt_val_dict
        
        
        
# Here our objective is to show first that Greedy is non optimal on these problem instances
def greedy_roi(cur_prob):
    vals = problems[cur_prob]['profits']
    wgts = problems[cur_prob]['weights']
    max_wgt = problems[cur_prob]['max_wgt']
    v_w_r = { i: vals[i]/wgts[i] for i  in range(len(vals))}
    v_w_r_sorted = sorted(v_w_r.items(), key=lambda x:x[1], reverse=True)
    final_set = []
    total_wt = 0
    for k,v in v_w_r_sorted:
        if total_wt <= max_wgt and wgts[k] + total_wt <= max_wgt:
            final_set.append(k)
            total_wt += wgts[k]
    return final_set, total_wt

def add_greedy():
    global problems
    
    greedy_res = {}
    for k in problems:
        greedy_indexes, greedy_wgts = greedy_roi(k)
        greedy_profit = sum([problems[k]['profits'][i] for i in greedy_indexes])
        opt_indexes =  problems[k]['opt_sol']['opt_index']
        opt_profit = sum([problems[k]['profits'][i] for i in opt_indexes])
        opt_weight = sum([problems[k]['weights'][i] for i in opt_indexes])

        inst_dict = dict()
        inst_dict['greedy_ind'] = sorted(greedy_indexes)
        inst_dict['greedy_weight'] = greedy_wgts
        inst_dict['greedy_profit'] = greedy_profit
#         inst_dict['opt_ind'] = sorted(opt_indexes)
#         inst_dict['opt_weight'] = opt_weight
#         inst_dict['opt_profit'] = opt_profit
        inst_dict['isOptimal'] = sorted(greedy_indexes) == sorted(opt_indexes)
        greedy_res[k] = inst_dict
        problems[k]['greedy_sol'] = inst_dict
    return greedy_res
    
def persist_json_file(data, res_file):
    '''
    Just to persist any json file
    '''

    json_object = json.dumps(data, indent=4, cls=NpEncoder)

    with open(res_file, "w") as outfile:
        outfile.write(json_object)
        
    print('Data Saved')
    




# Get problem instances
def get_problem_instance(problem_id):
    inst_obj = dict()

    
    if problem_id == 5: # Paper I5
        inst_obj['weights'] =  [7, 6, 4, 2, 1]
        inst_obj['profits'] = [14, 12, 10, 8, 3]
        inst_obj['max_wgt'] = 19
        inst_obj['details'] = 'Paper I5 Instance'
        inst_obj['name'] = 'Paper_I5'
    elif problem_id == 6: # Paper I6
        inst_obj['weights'] =  [7, 6, 4, 2, 1, 7]
        inst_obj['profits'] = [14, 12, 10, 8, 3, 13]
        inst_obj['max_wgt'] = 19
        inst_obj['details'] = 'Paper I6 Instance'
        inst_obj['name'] = 'Paper_I6'
    elif problem_id == 7: # Paper I7
        inst_obj['weights'] =  [7, 6, 4, 2, 1, 7, 7]
        inst_obj['profits'] = [14, 12, 10, 8, 3, 13, 12]
        inst_obj['max_wgt'] = 19
        inst_obj['details'] = 'Paper I7 Instance'
        inst_obj['name'] = 'Paper_I7'
    elif problem_id == 71: # Original DB2 I7
        inst_obj['weights'] =  [266, 232, 8, 132, 199, 2, 9]
        inst_obj['profits'] = [165811, 178871, 1213770, 1213770, 1213770, 44370, 44370]
        inst_obj['max_wgt'] = 140
        inst_obj['details'] = 'Paper DB2 I7 Original Instance'
        inst_obj['name'] = 'Paper_DB2_I7_Original'
    elif problem_id == 72: # Dummy DB2 I7 for SQIA
        inst_obj['profits'] = [4, 5, 27, 27, 27, 1, 1]
        inst_obj['weights'] = [126, 114, 3, 72, 95, 1, 4]
        inst_obj['max_wgt'] = 75
        inst_obj['details'] = 'Paper DB2 I7 Reduced Instance'
        inst_obj['name'] = 'Paper_DB2_I7_Scaled'
    else:
        print('Invalid Problem ID. Currently Acceptable Ids are >5 only')
    return inst_obj



def get_fudge(problems, alpha, bs_loop, Num_runs):
    L = 2 ** len(problems[1]['profits'])
    print('BF Calls:', L)
    fudge = 0
    for f in range(1, 100):
        o_calls = floor(round(alpha,2) * sqrt(L)) * (Num_runs + f) * bs_loop * 2
        print('Fudge:', f, ' O_calls:',  o_calls)
        if o_calls > L:
            fudge = f - 1
            print('Selected Fudge:', fudge)
            break
    return fudge


# Returns an adjusted alpha incase of timeout is 0
# This method returns additional fields to help calculate the fudge
def get_alpha_and_fudge(problems, delta, shots = 1):
    V_max = sum(problems[1]['profits'])
    L = 2 ** len(problems[1]['profits'])
    

    alpha = None
    
    epsilon = 1 - delta
    Num_runs = ceil(log(1/epsilon,3))
    
    bs_loop = ceil(log(V_max, 2))
    print('bs_loop', bs_loop)
    print('Num_runs', Num_runs)
    
    raw_alpha =  floor(sqrt(L)) / (2 * bs_loop * Num_runs * shots)
    print('raw alpha <', raw_alpha)
    
    alpha = round(raw_alpha,2)
    
    # Checking if alpha is feasible
    timeout = floor(alpha * sqrt(L))
    if timeout == 0:        
        for i in range(200):
            alpha = alpha + 0.01
            if floor(alpha * sqrt(L)) > 0:
                break

    alpha = round(alpha,2)
    print('Final alpha:', alpha)  
    
    # Calculating fudge
    fudge = get_fudge(problems, alpha, bs_loop, Num_runs)
    return raw_alpha, alpha, fudge
