import click
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

import utils.io as io


def compute_cap_score(query_caps,retrieved_caps,weights=(1.0,)):
    best_score = 0
    ref = [cap.lower().split() for cap in query_caps]
    for rcap in retrieved_caps:
        hyp = rcap.lower().split()
        score = sentence_bleu(ref,hyp,weights=weights)
        if score > best_score:
            best_score = score
    
    return best_score


@click.command()
@click.option(
    '--knn_json',
    type=str,
    help='Path to knn json file')
@click.option(
    '--k',
    type=int,
    default=3,
    help='Number of nearest neighbors')
def main(**kwargs):
    knn = io.load_json_object(kwargs['knn_json'])
    num_objects = len(knn)
    cap_scores = []
    best_in_k_cap_scores = []
    for i in tqdm(range(num_objects)):
        for query_info in knn[i]:
            if query_info is None:
                continue

            num_retrieved = len(query_info['retrieved_caps'])
            query_scores = [None]*num_retrieved
            for j in range(kwargs['k']):
                if j >= num_retrieved:
                    query_scores[j] = 0.0
                    continue

                query_scores[j] = compute_cap_score(
                    query_info['query_cap'],
                    query_info['retrieved_caps'][j])

            best_in_k_query_scores = [
                max(query_scores[:j+1]) for j in range(kwargs['k'])]

            cap_scores.append(query_scores)
            best_in_k_cap_scores.append(best_in_k_query_scores)

    cap_scores = np.array(cap_scores)
    cap_scores = np.mean(cap_scores,0)
    print(cap_scores)

    best_in_k_cap_scores = np.array(best_in_k_cap_scores)
    best_in_k_cap_scores = np.mean(best_in_k_cap_scores,0)
    print(best_in_k_cap_scores)




if __name__=='__main__':
    main()