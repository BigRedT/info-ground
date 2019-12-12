export HDF5_USE_FILE_LOCKING=FALSE

EXP=$1

if [ "$EXP" == 'det' ]
then
    python -m exp.hico_cls.run.evaluate \
        --exp_name 'det' \
        --model_num -100 \
        --skip_object_context_layer

elif [ "$EXP" == 'context_det_scratch' ]
then
    python -m exp.hico_cls.run.evaluate \
        --exp_name 'context_det_scratch' \
        --model_num -100

elif [ "$EXP" == 'context_det_frozen' ]
then
    python -m exp.hico_cls.run.evaluate \
        --exp_name 'context_det_frozen' \
        --model_num -100

elif [ "$EXP" == 'context_det_frozen_finetune' ]
then
    python -m exp.hico_cls.run.evaluate \
        --exp_name 'context_det_frozen_finetune' \
        --model_num -100

elif [ "$EXP" == 'context_det_finetune' ]
then
    python -m exp.hico_cls.run.evaluate \
        --exp_name 'context_det_finetune' \
        --model_num -100
fi