SUBSET=$1
python -m exp.gen_noun_negatives.sample_neg_bert \
    --subset $SUBSET \
    --rank 30 \
    --select 25