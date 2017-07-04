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

dist_train_example_model.py is  nearly a normal tf model 
  but two litte exceptions,
  
  - a. use the cluster device info: 
   
      ```
      device_info, worker_spec, num_workers = get_cluster_device_info()
      ```
  - b. use a well configured session with get_sess from ```cluster_helper.py```:
  
    ```
    sess, train_step = get_sess(worker_spec, num_workers, opt, loss, global_step)
    ```
