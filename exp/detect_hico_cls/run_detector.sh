SUBSET=$1
python -m detector.detect \
    --det_input "/home/workspace/Data/context-regions/hico_proc/det_input_${SUBSET}.json" \
    --out_dir "/home/workspace/Data/context-regions/hico_proc/detections/${SUBSET}"