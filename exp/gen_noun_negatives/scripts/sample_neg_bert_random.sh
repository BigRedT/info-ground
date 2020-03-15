SUBSET=$1
python -m exp.gen_noun_negatives.sample_neg_bert_random \
    --subset $SUBSET \
    --select 25