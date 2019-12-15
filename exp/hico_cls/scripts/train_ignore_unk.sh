export HDF5_USE_FILE_LOCKING=FALSE

pretrained_object_encoder_path='/home/workspace/Data/context-regions/coco_exp/bert_negs_lang_loss_1_neg_noun_loss_1_neg_verb_loss_1/models/best_object_encoder'

EXP=$1

if [ "$EXP" == 'det' ]
then
    python -m exp.hico_cls.run.train \
        --exp_name 'det_wo_unk' \
        --model_num -1 \
        --lr 1e-4 \
        --train_batch_size 100 \
        --finetune_object_encoder \
        --skip_object_context_layer \
        --ignore_unk_labels_during_training

elif [ "$EXP" == 'context_det_scratch' ]
then
    python -m exp.hico_cls.run.train \
        --exp_name 'context_det_scratch_wo_unk' \
        --model_num -1 \
        --lr 1e-4 \
        --train_batch_size 100 \
        --finetune_object_encoder \
        --ignore_unk_labels_during_training

elif [ "$EXP" == 'context_det_frozen' ]
then
    python -m exp.hico_cls.run.train \
        --exp_name 'context_det_frozen_wo_unk' \
        --model_num -1 \
        --lr 1e-4 \
        --train_batch_size 100 \
        --pretrained_object_encoder_path $pretrained_object_encoder_path \
        --ignore_unk_labels_during_training

elif [ "$EXP" == 'context_det_finetune' ]
then
    python -m exp.hico_cls.run.train \
        --exp_name 'context_det_finetune_wo_unk' \
        --model_num -1 \
        --lr 1e-4 \
        --train_batch_size 100 \
        --finetune_object_encoder \
        --warmup \
        --pretrained_object_encoder_path $pretrained_object_encoder_path \
        --ignore_unk_labels_during_training
fi