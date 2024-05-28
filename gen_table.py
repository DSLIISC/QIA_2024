import os
from utils import *
import argparse

def generate_table(res_location, repeat):
    BASE_PATH = res_location
    
    # SQIA Experiments
    proposed_sol = 'sqia'
    shots = 1
    exp_env = 'noiseless'
    gs_filename = 'all_res_repeat_'+str(repeat)+'.json'
      
    prob_id = 5
    gs_res_json = os.path.join(BASE_PATH, str(prob_id), str(shots), exp_env, gs_filename)
    with open(gs_res_json, "r") as in_file:
        results = json.load(in_file)
    all_results = results['Results']
    gs_i5_res = getScore_GS_relaxed(all_results[:])
    
    prob_id = 6
    gs_res_json = os.path.join(BASE_PATH, str(prob_id), str(shots), exp_env, gs_filename)
    with open(gs_res_json, "r") as in_file:
        results = json.load(in_file)
    all_results = results['Results']
    gs_i6_res = getScore_GS_relaxed(all_results[:])
    
    prob_id = 7
    gs_res_json = os.path.join(BASE_PATH, str(prob_id), str(shots), exp_env, gs_filename)
    with open(gs_res_json, "r") as in_file:
        results = json.load(in_file)
    all_results = results['Results']
    gs_i7_res = getScore_GS_relaxed(all_results[:])
    
    prob_id = 72
    gs_res_json = os.path.join(BASE_PATH, str(prob_id), str(shots), exp_env, gs_filename)
    with open(gs_res_json, "r") as in_file:
        results = json.load(in_file)
    all_results = results['Results']
    gs_i72_res = getScore_GS_relaxed(all_results[:])

    # OQIA Experiments
    proposed_sol = 'oqia'
    shot_inc = 'inc_shot'
    exp_env = 'noiseless'
    filename = 'all_res_inc_shot_repeat_'+str(repeat)+'.json'
    
    prob_id = 5
    res_json = os.path.join(BASE_PATH, str(prob_id), shot_inc, exp_env, filename)
    with open(res_json, "r") as in_file:
        results = json.load(in_file)
    all_results = results['Results']
    oqia_i5_res = getScore_OQIA(all_results[2])
    
    prob_id = 6
    res_json = os.path.join(BASE_PATH, str(prob_id), shot_inc, exp_env, filename)
    with open(res_json, "r") as in_file:
        results = json.load(in_file)
    all_results = results['Results']
    oqia_i6_res = getScore_OQIA(all_results[2])
    
    prob_id = 7
    res_json = os.path.join(BASE_PATH, str(prob_id), shot_inc, exp_env, filename)
    with open(res_json, "r") as in_file:
        results = json.load(in_file)
    all_results = results['Results']
    oqia_i7_res = getScore_OQIA(all_results[2])
    
    prob_id = 71
    res_json = os.path.join(BASE_PATH, str(prob_id), shot_inc, exp_env, filename)
    with open(res_json, "r") as in_file:
        results = json.load(in_file)
    all_results = results['Results']
    oqia_i71_res = getScore_OQIA(all_results[2])
    
    
    problem_set = []
    avg_scores = []
    ideal_score = []
    min_score = []
    com_overhead = []
    qubits = []
    depth = []


    problem_set.append('CDB_I7_SQIA')
    avg_scores.append(gs_i72_res['gs_score_relaxed'])
    ideal_score.append(gs_i72_res['gs_score_ideal'])
    min_score.append(gs_i72_res['Min_score'])
    com_overhead.append(gs_i72_res['computational_overhead'])
    qubits.append(gs_i72_res['Qubits'])
    depth.append(gs_i72_res['Depth'])

    problem_set.append('CDB_I7_OQIA')
    avg_scores.append(oqia_i71_res['oqia_score_weighted'])
    ideal_score.append(oqia_i71_res['oqia_score_opt'])
    min_score.append(oqia_i71_res['oqia_min_score'])
    com_overhead.append(oqia_i71_res['computational_overhead'])
    qubits.append(oqia_i71_res['Qubits'])
    depth.append(oqia_i71_res['Depth'])

    problem_set.append('I5_SQIA')
    avg_scores.append(gs_i5_res['gs_score_relaxed'])
    ideal_score.append(gs_i5_res['gs_score_ideal'])
    min_score.append(gs_i5_res['Min_score'])
    com_overhead.append(gs_i5_res['computational_overhead'])
    qubits.append(gs_i5_res['Qubits'])
    depth.append(gs_i5_res['Depth'])

    problem_set.append('I5_OQIA')
    avg_scores.append(oqia_i5_res['oqia_score_weighted'])
    ideal_score.append(oqia_i5_res['oqia_score_opt'])
    min_score.append(oqia_i5_res['oqia_min_score'])
    com_overhead.append(oqia_i5_res['computational_overhead'])
    qubits.append(oqia_i5_res['Qubits'])
    depth.append(oqia_i5_res['Depth'])

    problem_set.append('I6_SQIA')
    avg_scores.append(gs_i6_res['gs_score_relaxed'])
    ideal_score.append(gs_i6_res['gs_score_ideal'])
    min_score.append(gs_i6_res['Min_score'])
    com_overhead.append(gs_i6_res['computational_overhead'])
    qubits.append(gs_i6_res['Qubits'])
    depth.append(gs_i6_res['Depth'])

    problem_set.append('I6_OQIA')
    avg_scores.append(oqia_i6_res['oqia_score_weighted'])
    ideal_score.append(oqia_i6_res['oqia_score_opt'])
    min_score.append(oqia_i6_res['oqia_min_score'])
    com_overhead.append(oqia_i6_res['computational_overhead'])
    qubits.append(oqia_i6_res['Qubits'])
    depth.append(oqia_i6_res['Depth'])

    problem_set.append('I7_SQIA')
    avg_scores.append(gs_i7_res['gs_score_relaxed'])
    ideal_score.append(gs_i7_res['gs_score_ideal'])
    min_score.append(gs_i7_res['Min_score'])
    com_overhead.append(gs_i7_res['computational_overhead'])
    qubits.append(gs_i7_res['Qubits'])
    depth.append(gs_i7_res['Depth'])

    problem_set.append('I7_OQIA')
    avg_scores.append(oqia_i7_res['oqia_score_weighted'])
    ideal_score.append(oqia_i7_res['oqia_score_opt'])
    min_score.append(oqia_i7_res['oqia_min_score'])
    com_overhead.append(oqia_i7_res['computational_overhead'])
    qubits.append(oqia_i7_res['Qubits'])
    depth.append(oqia_i7_res['Depth'])
    
    labels = ['Problems', 'Weighted Average', 'Optimal Fraction', 'Worst Case', 
          'Qubits', 'Depth', 'Computation Overhead %']
    df_res = pd.DataFrame(list(zip(problem_set, all_scores, ideal_score, min_score,
                                  qubits, depth, com_overhead)),
               columns =labels)
    print(df_res)

def main():
    print('Generating Table (Assuming all the executions are done for OQIA and SQIA)')
    parser = argparse.ArgumentParser()
    # parser.add_argument('-n', '--base_path', default=3.14)
    parser.add_argument('-bp', '--base_path', type=str)
    parser.add_argument('-r', '--repeat', type=int, default=10)
    args = parser.parse_args()

    # args = parser.parse_args()
    print (args.base_path)
    print (args.repeat)
    
    generate_table(args.base_path, args.repeat)

if __name__ == "__main__":
    main()