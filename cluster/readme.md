### 创建自己的模型训练集群

#### 创建一个k2上的tf集群来训练模型

1. 生成k2集群配置文件
    
    ```bash
    #单个资源的可选项有单个资源的配置大小,n核mG内存, 1Core2G,2Core4G,4Core8G,8Core16G,16Core32G,24Core64G
    
    python -m k2_cluster.create_cluster_conf --user_name LordStarkAya --numb_workers 2 --numb_ps 1 --single_resourse 1Core2G 
    
    ```
    上面程序执行完后会在本地生成一个tf-cluster.txt的集群配置文件
    
2. 导入集群配置文件
    
    打开http://k2.wac-ai.wacai.info/applications/tf-cluster,
    点击**应用导入**,导入刚才生成的tf-cluster.txt。
    
完了。这就完啦！是不是很简单！

### 不用的集群资源要及时释放啊！

