#!/usr/bin/env bash
echo "start train flow ...."
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=$(cd "${DIR}" && pwd)/logs
WKR_LOG_PREFIX="${LOG_DIR}/worker"

PS_HOSTS=localhost:2222
WORKER_HOSTS=localhost:2223,localhost:2224

#start worker 0
python3 dist_train_example_model.py \
    --data_dir="example_train_data/part_1" \
    --job_name=worker \
    --train_steps=500 \
    --sync_replas=True \
    --log_dir=$LOG_DIR \
    --ps_hosts ${PS_HOSTS} \
    --worker_hosts ${WORKER_HOSTS} \
    --task_index=0 \
    2>&1 | tee "${WKR_LOG_PREFIX}_0.log" &

#start worker 1
python3 dist_train_example_model.py \
    --data_dir="example_train_data/part_2" \
    --job_name=worker \
    --train_steps=500 \
    --sync_replas=True \
    --log_dir=$LOG_DIR \
    --ps_hosts ${PS_HOSTS} \
    --worker_hosts ${WORKER_HOSTS} \
    --task_index=1 \
    2>&1 | tee "${WKR_LOG_PREFIX}_1.log" &

echo "worker 0 log : ${WKR_LOG_PREFIX}_0.log"
echo "worker 1 log : ${WKR_LOG_PREFIX}_1.log"