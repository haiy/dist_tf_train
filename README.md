## 如何使用分布式tensorflow


#### 首先,创建一个k2上的tf集群

1. 生成k2集群配置文件
    
    ```bash
    #单个资源的可选项有单个资源的配置大小,n核mG内存, 1Core2G,2Core4G,4Core8G,8Core16G,16Core32G,24Core64G
    
    python -m k2_cluster.create_cluster_conf --user_name LordStarkAya --numb_workers 2 --numb_ps 1 --single_resourse 1Core2G 
    
    ```
    上面程序执行完后会在本地生成一个tf-cluster.txt的集群配置文件，同时也会打印出生成的集群信息。
    
2. 导入集群配置文件
    
    打开http://k2.wac-ai.wacai.info/applications/tf-cluster,
    点击**应用导入**,导入刚才生成的tf-cluster.txt。
    
完了。这就完啦！是不是很简单！

#### 然后,使用创建好的集群来训练模型

1. 改写我们的现有的模型为分布式
    
    a. 在我们的模型代码中导入cluster.cluster_helper来获取并使用分布式设备
    b. 创建分布式需要的session 
   
      ```python
      from cluster import cluster_helper
      
      #a. 获取并使用分布式设备
      device_info = cluster_helper.get_cluster_device_info()
      with tf.device(device_info):
            ...
            train_step, optimizer = build_graph()
            
            #b.创建分布式需要的session
            sess, train_step = get_sess(worker_spec, num_workers, opt, loss, global_step)
            
      ```
      
      一个完整的例子,参照**```dist_train_example_model.py```**
      
2. 执行训练

    ```bash
    
    WORKER_HOSTS=demo-worker-0.tf-cluster.svc.intra.wac-ai.com,demo-worker-1.tf-cluster.svc.intra.wac-ai.com
    PS_HOSTS=demo-ps-0.tf-cluster.svc.intra.wac-ai.com
    LOG_DIR=./logs
 
    #start worker 0
    python3 dist_train_example_model.py \
        --data_dir="example_train_data/part_1" \
        --job_name=worker \
        --train_steps=2500 \
        --sync_replas=True \
        --log_dir=$LOG_DIR \
        --ps_hosts ${PS_HOSTS} \
        --worker_hosts ${WORKER_HOSTS} \
        --task_index=0    
    
    #start worker 1
    python3 dist_train_example_model.py \
        --data_dir="example_train_data/part_2" \
        --job_name=worker \
        --train_steps=2500 \
        --sync_replas=True \
        --log_dir=$LOG_DIR \
        --ps_hosts ${PS_HOSTS} \
        --worker_hosts ${WORKER_HOSTS} \
        --task_index=1 
       
    ```


#### 最后

有问题，找 @deben！


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
 



