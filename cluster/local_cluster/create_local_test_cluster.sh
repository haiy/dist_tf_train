#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd && cd ..)"
LOG_DIR=$(dirname "${DIR}")/logs

echo $LOG_DIR

echo "ready to create local test cluster with cluster info worker|localhost:2223;localhost:2224,ps|localhost:2222 "
python3 grpc_tensorflow_server.py --cluster_spec="worker|localhost:2223;localhost:2224,ps|localhost:2222" --job_name=worker --task_id=0 2>&1 | tee $LOG_DIR/cluster_worker_0.log &
python3 grpc_tensorflow_server.py --cluster_spec="worker|localhost:2223;localhost:2224,ps|localhost:2222" --job_name=worker --task_id=1 2>&1 | tee $LOG_DIR/cluster_worker_1.log &
python3 grpc_tensorflow_server.py --cluster_spec="worker|localhost:2223;localhost:2224,ps|localhost:2222" --job_name=ps --task_id=0 2>&1 | tee $LOG_DIR/cluster_ps_0.log &
echo "create finished!"