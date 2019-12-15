export HDF5_USE_FILE_LOCKING=FALSE

EXP=$1

if [ "$EXP" == 'det' ]
then
    python -m exp.hico_cls.run.evaluate \
        --exp_name 'det_wo_unk' \
        --model_num -100 \
        --skip_object_context_layer

elif [ "$EXP" == 'context_det_scratch' ]
then
    python -m exp.hico_cls.run.evaluate \
        --exp_name 'context_det_scratch_wo_unk' \
        --model_num -100

elif [ "$EXP" == 'context_det_frozen' ]
then
    python -m exp.hico_cls.run.evaluate \
        --exp_name 'context_det_frozen_wo_unk' \
        --model_num -100

elif [ "$EXP" == 'context_det_finetune' ]
then
    python -m exp.hico_cls.run.evaluate \
        --exp_name 'context_det_finetune_wo_unk' \
        --model_num -100
fi