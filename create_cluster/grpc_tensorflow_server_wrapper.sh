#!/usr/bin/env bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Wrapper script for grpc_tensorflow_server.py in test server

LOG_FILE="/tmp/grpc_tensorflow_server.log"

SCRIPT_DIR=$( cd ${0%/*} && pwd -P )

touch "${LOG_FILE}"

export PATH=/data/program/miniconda/envs/py35/bin:$PATH
echo `which  python`
echo `python --version`
echo `pwd`

if [ -z ${CLUSTER_INFO+x} ];then
    CLUSTER_INFO="worker|tf-worker-1:2222;tf-worker-2:2222,ps|tf-ps-1:2222"
fi
if [ -z ${JOB_NAME+x} ];then
    JOB_NAME=worker
fi
if [ -z ${TASK_ID+x} ];then
    TASK_ID=0
fi
echo "Cluster info :${CLUSTER_INFO}."
echo "Job name: ${JOB_NAME}"
echo "Task id: ${TASK_ID}"

python ${SCRIPT_DIR}/grpc_tensorflow_server.py --cluster_spec="${CLUSTER_INFO}" --job_name=${JOB_NAME} --task_id=${TASK_ID}
#--cluster_spec="worker|localhost:2222;foo:2222,ps|bar:2222;qux:2222" --job_name=worker --task_id=0
#python ${SCRIPT_DIR}/grpc_tensorflow_server.py $@ 2>&1 | tee "${LOG_FILE}"
