import os
import numpy as np

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
    hois = io.load_json_object(os.path.join(
        hico_paths['proc_dir'],
        hico_paths['hoi_list_json']))

    labels_npy = os.path.join(
        hico_paths['proc_dir'],
        hico_paths['labels_npy']['test'])

    labels = np.load(labels_npy)
    pos_labels = labels==1
    pos_counts = np.sum(pos_labels,0)

    diff_APs = []
    for ap1,ap2,hoi,count in \
        zip(results1['APs'],results2['APs'],hois,pos_counts):
        diff_APs.append((
            round(100*ap1,2),
            round(100*ap2,2),
            round(100*(ap1-ap2),2),
            hoi['interaction'],
            hoi['object'],
            count))

    object_APs = {}
    interaction_APs = {}
    for diff_ap in diff_APs:
        obj = diff_ap[4]
        interaction = diff_ap[3]
        
        
        if obj not in object_APs:
            object_APs[obj] = []
        
        object_APs[obj].append(diff_ap)

        if interaction not in interaction_APs:
            interaction_APs[interaction] = []
        
        interaction_APs[interaction].append(diff_ap)

    for APs in [object_APs,interaction_APs]:
        for k,v  in APs.items():
            mean_diff_ap = 0
            num_items = 0
            for diff_ap in v:
                mean_diff_ap += diff_ap[2]
                num_items += 1

            APs[k] = mean_diff_ap / (num_items+1e-6)

    # for v in sorted(interaction_APs.items(),key=lambda x: x[1]):
    #     print(v)

    weighted_AP_diff = 0
    weighted_AP1 = 0
    weighted_AP2 = 0
    count = 0
    for v in sorted(diff_APs,key=lambda x: x[5]):
        print(v)
        weighted_AP_diff += (v[5]*100/(100-v[1]))
        weighted_AP1 += (v[0])
        weighted_AP2 += (v[1])
        count += v[5]
    
    weighted_AP_diff = weighted_AP_diff / 600
    weighted_AP1 = weighted_AP1 / 600
    weighted_AP2 = weighted_AP2 / 600
    print(weighted_AP1, weighted_AP2, weighted_AP_diff)

if __name__=='__main__':
    main()

    

