#!/bin/bash
# USAGE: ./dev_run.sh
# Mounts the local src directory into the container so edited scripts
# are used directly without rebuilding the SIF.

# 1. Define the image path
CONTAINER_IMG="src/dream_rnn_predictor.sif"

# 2. Check that the image exists
if [ ! -f "$CONTAINER_IMG" ]; then
    echo "❌ Error: Could not find container at $CONTAINER_IMG"
    echo "   Make sure you are running this script from the 'DREAM_RNN_GAME' directory."
    exit 1
fi

# 3. Resolve predictor IP and a free port
pred_ip=$(hostname -I | awk '{print $2}')
pred_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

echo "=========================================================="
echo "🧪 STARTING DEV MODE: ${CONTAINER_IMG}"
echo "=========================================================="
echo "   Mapping host './src'  ---> Container '/src'"
echo "   Mapping host '$PWD'   ---> Container '/mnt' (Working Dir)"
echo "   Predictor : http://$pred_ip:$pred_port"
echo "----------------------------------------------------------"

# 4. The Apptainer command
apptainer exec --nv \
    --bind ./src:/src \
    --bind $PWD:/mnt \
    --pwd /mnt \
    --env PYTHONPATH="/src/dream_rnn_script_and_utils:/src/script_and_utils:$PYTHONPATH" \
    "$CONTAINER_IMG" \
    python3 /src/script_and_utils/dream_rnn_predictor_rest_api.py $pred_ip $pred_port