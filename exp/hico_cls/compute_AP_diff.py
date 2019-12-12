import os

from global_constants import hico_paths
import utils.io as io

def main():
    results1_json = os.path.join(
        hico_paths['exp_dir'],
        'context_det_frozen/eval_results_best.json')
    results2_json = os.path.join(
        hico_paths['exp_dir'],
        'det/eval_results_best.json')

    results1 = io.load_json_object(results1_json)
    results2 = io.load_json_object(results2_json)

    diff_APs = []
    for ap1, ap2 in zip(results1['APs'],results2['APs']):
        diff_APs.append((ap1,ap2,ap1-ap2))

    for v in sorted(diff_APs,key=lambda x: x[2]):
        print(v)

if __name__=='__main__':
    main()

    

