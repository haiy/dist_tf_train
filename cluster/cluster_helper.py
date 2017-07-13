# encoding: utf-8
"""Distributed MNIST training and validation, with model replicas.

A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on one parameter server (ps), while the ops
are executed on two worker nodes by default. The TF sessions also run on the
worker node.
Multiple invocations of this script can be done in parallel, with different
values for --task_index. There should be exactly one invocation with
--task_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.

The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer("task_index", 0,
                        "Worker task index, should be >= 0. task_index=0 is "
                        "the master worker task the performs the variable "
                        "initialization ")
tf.flags.DEFINE_integer("num_gpus", 0,
                        "Total number of gpus for each machine."
                        "If you don't use GPU, please set it to '0'")
tf.flags.DEFINE_integer("replicas_to_aggregate", None,
                        "Number of replicas to aggregate before parameter update"
                        "is applied (For sync_replicas mode only; default: "
                        "num_workers)")
tf.flags.DEFINE_boolean("sync_replicas", False,
                        "Use the sync_replicas (synchronized replicas) mode, "
                        "wherein the parameter updates from workers are aggregated "
                        "before applied to avoid stale gradients")
tf.flags.DEFINE_boolean("existing_servers", True, "Whether servers already exists. If True, "
                                                  "will use the worker hosts via their GRPC URLs (one client process "
                                                  "per worker host). Otherwise, will create an in-process TensorFlow "
                                                  "server.")
tf.flags.DEFINE_string("ps_hosts", "localhost:2222",
                       "Comma-separated list of hostname:port pairs")
tf.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                       "Comma-separated list of hostname:port pairs")
tf.flags.DEFINE_string("job_name", "worker", "job name: worker or ps")
tf.flags.DEFINE_string("log_dir", "../logs", "run log ouput and checkpoint dir")


def _get_worker_info():
    worker_spec = FLAGS.worker_hosts.split(",")
    num_workers = len(worker_spec)
    return worker_spec, num_workers

def _get_worker_device():
    worker_spec, num_workers = _get_worker_info()
    if FLAGS.num_gpus > 0:
        if FLAGS.num_gpus < num_workers:
            raise ValueError("number of gpus is less than number of workers")
        # Avoid gpu allocation conflict: now allocate task_num -> #gpu
        # for each worker in the corresponding machine
        #gpu = (FLAGS.task_index % FLAGS.num_gpus)
        gpu = 0

        # worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
        worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
        print("gpu worker device: ", worker_device)
    elif FLAGS.num_gpus == 0:
        # Just allocate the CPU to worker server
        cpu = 0
        worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)
    return worker_device

def get_cluster_device_info():
    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")
    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)
    worker_spec, num_workers = _get_worker_info()
    # Construct the cluster and start the server
    ps_spec = FLAGS.ps_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
    worker_device = _get_worker_device()
    device_info = tf.train.replica_device_setter(worker_device=worker_device, cluster=cluster)
    return device_info


def get_sess(opt, loss, global_step, checkpoint_dir):
    worker_spec, num_workers = _get_worker_info()
    is_chief = (FLAGS.task_index == 0)
    init_op = tf.global_variables_initializer()
    if checkpoint_dir:
        train_dir = checkpoint_dir
    elif FLAGS.log_dir:
        train_dir = FLAGS.log_dir
    else:
        raise ValueError("Must set checkpoint dir!")
    worker_device = _get_worker_device()
    # sess_config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=True,
    #                              device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

    sess_config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=True,
                                 device_filters=["/job:ps", worker_device])

    if FLAGS.sync_replicas:
        if FLAGS.replicas_to_aggregate is None:
            replicas_to_aggregate = num_workers
        else:
            replicas_to_aggregate = FLAGS.replicas_to_aggregate
        # 同步，聚合梯度并optimizer
        opt = tf.train.SyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=replicas_to_aggregate,
            total_num_replicas=num_workers,
            name="sync_replicas_opt")
        train_step = opt.minimize(loss, global_step=global_step)
        # 当前worker的初始化操作operation. 将全局的global step步数赋值给本地
        local_init_op = opt.local_step_init_op
        if is_chief:
            local_init_op = opt.chief_init_op
        # 未初始化的变量
        ready_for_local_init_op = opt.ready_for_local_init_op
        chief_queue_runner = opt.get_chief_queue_runner()
        sync_init_op = opt.get_init_tokens_op()
        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=train_dir,
            init_op=init_op,
            local_init_op=local_init_op,
            ready_for_local_init_op=ready_for_local_init_op,
            recovery_wait_secs=1,
            global_step=global_step)
    else:
        """
        A training helper that checkpoints models and computes summaries.
        The Supervisor is a small wrapper around a `Coordinator`, a `Saver`,
          and a `SessionManager` that takes care of common needs of TensorFlow
            training programs.

        Supervisor模型保存
         *chief*: the task that handles  initialization, checkpoints, summaries, and recovery.  The other tasks
           depend on the *chief* for these services.
        """
        train_step = opt.minimize(loss, global_step=global_step)
        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=train_dir,
            init_op=init_op,
            recovery_wait_secs=1,
            global_step=global_step)

    if is_chief:
        print("Worker %d: Initializing session..." % FLAGS.task_index)
    else:
        print("Worker %d: Waiting for session to be initialized..." % FLAGS.task_index)

    server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
    print("Using existing server at: %s" % server_grpc_url)
    sess = sv.prepare_or_wait_for_session(server_grpc_url,
                                          config=sess_config)

    print("Worker %d: Session initialization complete." % FLAGS.task_index)
    if FLAGS.sync_replicas and is_chief:
        sess.run(sync_init_op)
        sv.start_queue_runners(sess, [chief_queue_runner])
    return sess, train_step