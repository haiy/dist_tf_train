#encoding: utf-8
import requests
import json
import argparse
import re
import time
import sys
import os

cur_dir = os.path.dirname(__file__)
template_file=os.path.join(cur_dir,"template_pod.json")
def get_app_dict(namespace, app_name, core_numb="1", mem_g="2G", env_dict={}, replica_num=1,
                 img_name="dockerhub.test.wacai.info/wac.ai/tf_grpc_server:latest"):
    time_stamp = time.strftime("%Y%m%d%H%M%S")
    json_dict = json.load(open(template_file, 'r', encoding="utf-8"))
    json_dict["name"] = app_name #+ "-" + time_stamp
    json_dict["namespace"] = namespace
    json_dict["deployment"]["spec"]["replicas"] = replica_num
    for k in ["service", "deployment", "pod"]:
        json_dict[k]["namespace"] = namespace
        json_dict[k]["name"] = app_name
        json_dict[k]["spec"]["labels"] = []
        json_dict[k]["spec"]["labels"].append({"key": "app", "value": namespace + "-" + app_name})
    json_dict["service"]["legacy_domain"] = app_name + ".wacai.info"
    json_dict["pod"]["spec"]["envs"] = [{"key": k, "value": v} for k, v in env_dict.items()]
    json_dict["pod"]["spec"]["cpu"] = str(core_numb)
    json_dict["pod"]["spec"]["memory"] = str(mem_g) + "G"
    json_dict["pod"]["spec"]["image"] = img_name
    json_dict["pod"]["spec"]["node_selector"] = ["k2.wacai.com/cpu"]
    return json_dict

def get_bash_vars(ps_names, worker_names):
    """
    HOST=".tf-cluster.svc.intra.wac-ai.com"
PS_HOSTS=deben-ps-0$HOST:2222
WORKER_HOSTS=deben-worker-0$HOST:2222,deben-worker-1$HOST:2222
    :param ps_names:
    :param worker_names:
    :return:
    """
    host_str = ".tf-cluster.svc.intra.wac-ai.com"
    workers = []
    ps = []
    for single_worker_name in worker_names:
        workers.append(single_worker_name.split(":")[0] + host_str)
    for single_ps_name in ps_names:
        ps.append(single_ps_name.split(":")[0] + host_str)
    print("\nWORKER_HOSTS="+",".join(workers))
    print("PS_HOSTS=" + ",".join(ps))


def generate_cluster(name_prefix, namespace, number_workers, number_ps_servers, resourse_pattern = "1core2g"):
    env_dict = {}
    container_info_list = []
    ps_names = []
    for i in range(number_ps_servers):
        info_d = {"APP_NAME": name_prefix + "-ps-" + str(i),
                  "JOB_NAME": "ps",
                  "TASK_ID": str(i),
                  "POD_NAMESPACE": namespace
                  }
        ps_names.append(info_d["APP_NAME"] + ":2222")
        container_info_list.append(info_d)

    worker_names = []
    for i in range(number_workers):
        info_d = {"APP_NAME": name_prefix + "-worker-" + str(i),
                  "JOB_NAME": "worker",
                  "TASK_ID": str(i),
                  "POD_NAMESPACE": namespace
                  }
        worker_names.append(info_d["APP_NAME"] + ":2222")
        container_info_list.append(info_d)

    all_resourse_pattern = "1core2g,2core4g,4core8g,8core16g,16core32g,24core64g".split(",")
    if resourse_pattern.lower() not in all_resourse_pattern:
        raise Exception("输入的单个资源配置有错！支持的配置类型有：1Core2G,2Core4G,4Core8G,8Core16G,16Core32G,24Core64G！")

    res = re.search("(\d+)core(\d+)g", resourse_pattern.lower())
    core_num = int(res.group(1))
    mem_g = int(res.group(2))

    cluster_info = []
    cluster_str = ",".join(["worker|" + ";".join(worker_names), "ps|" + ";".join(ps_names)])
    for i in container_info_list:
        i["CLUSTER_INFO"] = cluster_str
        cluster_info.append(get_app_dict(namespace, i["APP_NAME"], core_numb=core_num, mem_g=mem_g, env_dict=i))
    #print(cluster_str)
    get_bash_vars(ps_names, worker_names)
    return cluster_info, cluster_str

def save_cluster_file(cluster_info, fname = os.path.join(cur_dir,"tf-cluster.txt")):
    json.dump(cluster_info, open(fname, 'w'))
    print("\ncluster config file saved to :",fname)


def _test():
    namespace = "tf-cluster"
    user_name = "demo"
    cluster_info, cluster_str = generate_cluster(user_name,namespace, 2, 1)
    #create_k2_cluster(user_name, namespace, cluster_info)
    save_cluster_file(cluster_info)

def run(input_args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-N",
        "--user_name",
        type=str,
        default="demo",
        help="""创建者名字"""
    )
    parser.add_argument(
        "-W",
        "--numb_workers",
        type=int,
        default=2,
        help="worker的数量"
    )
    parser.add_argument(
        "-P",
        "--numb_ps",
        type=int,
        default=1,
        help="parameter server的数量"
    )
    parser.add_argument(
        "-R",
        "--single_resourse",
        type=str,
        default="1Core2G",
        help="单个资源的配置大小,n核mG内存, 1Core2G,2Core4G,4Core8G,8Core16G,16Core32G,24Core64G"
    )
    parsed, unparsed = parser.parse_known_args(input_args)
    namespace = "tf-cluster"
    print("parsed input:",parsed.user_name,parsed.numb_workers,parsed.numb_ps, parsed.single_resourse)
    cluster_info, cluster_str = generate_cluster(parsed.user_name, namespace, parsed.numb_workers, parsed.numb_ps, parsed.single_resourse)
    save_cluster_file(cluster_info)

    #print(cluster_str)

if __name__ == "__main__":
    run(sys.argv)