ngc batch run \
    --instance dgx1v.16g.1.norm \
    --name "context-regions" \
    --image "nvcr.io/nvidian/lpr-parasol/context-regions:latest" \
    --result /result \
    --workspace ws-tanmayg:/home/workspace \
    --commandline "sleep 7d" \
    --total-runtime 7D \
    --port 6006 \
    --port 6007 \
    --port 8000 \
    --port 8001

