# dist_tf_train
a simple distribute tensorflow framework template

1. create a simple local cluster

    ```
    cd create_cluster
    ./create_local_test_cluster.sh
    ```

2. run an distributed example
 
     ```
     ./start_dist_train.sh
     ```
 
 3. stop the local cluster 
  
      ```
      cd create_cluster
      kill_local_test_cluster.sh
      ```
      
## how to use template

  use the cluster_helper to get device info and then create session: 
   
      ```
      import cluster_helper
      with tf.device(cluster_helper.get_cluster_device_info()):
            ...
            train_step, optimizer = build_graph()
            sess, train_step = get_sess(worker_spec, num_workers, opt, loss, global_step)
            
      ```
      
## FAQ

1. Graph finalized error

    ```
        raise RuntimeError("Graph is finalized and cannot be modified.")
    RuntimeError: Graph is finalized and cannot be modified.
    ```
   Fix:
   comment original session init code
   ```python
   #     sess.run(tf.global_variables_initializer())
   #     sess.run(tf.initialize_all_variables())
   ```
 


Ref:
[tensorflow dist_test](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/dist_test)
