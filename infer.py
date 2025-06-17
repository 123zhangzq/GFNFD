import torch
import torch.optim as optim
import numpy as np
import wandb
from data_loader import generate_train_data
from config import CONFIG
from utils import init_wandb, aspect_ratio_normalize,log_metrics, sample_uniform_per_bin, \
    non_uniform_thermometer_encode, get_epsilon_exp, compute_1order_distance, compute_2order_min_distance, compute_multiorder_min_distance
from gnn_model import OrderCourierHeteroGNN, NodeEmbedGNN
from env_instance import HeteroOrderDispatchEnv
import os
from lkh3_solver import LKH_solve_rider_with_retry
from data_loader import gen_pyg_data_nodes, gen_pyg_hetero_bigraph
import platform
import math
from utils import output_pareto_plot


'''
--infer
--model_path
./pretrained/10_3/checkpoint_epoch20.pt
'''

def infer(model_path):
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)

    BINS = CONFIG["preference_bins"]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # run name config
    run_name = f"[{CONFIG['run_name']}]" if CONFIG.get('run_name') else ""
    run_name += f"_orders{CONFIG['num_orders']}_riders{CONFIG['num_riders']}_sd{CONFIG['seed']}"
    pretrained_name = (
        CONFIG["pretrained"].replace("../pretrained/order_model/", "").replace("/", "_").replace(".pt", "")
        if CONFIG.get("pretrained") else None
    )
    run_name += f"{'' if pretrained_name is None else '_fromckpt-' + pretrained_name}"

    # unify the config
    config = CONFIG

    # saving model path
    output_dir = config["output"]
    savepath = os.path.join(output_dir, f'{config["num_orders"]}_{config["num_riders"]}', run_name)
    os.makedirs(savepath, exist_ok=True)

    # Init. GNN 和 GFlowNet
    gnn_node_emb = NodeEmbedGNN(feats = 3).to(DEVICE)
    gnn_order_dispatch = OrderCourierHeteroGNN(order_input_dim = 5, rider_input_dim = 3, edge_attr_dim= 1,
                                                hidden_dim = 64, omega_dim=6, flg_gfn=True).to(DEVICE)

    # Init. optimizer and schheduler
    optimizer = torch.optim.AdamW(list(gnn_order_dispatch.parameters()) + list(gnn_node_emb.parameters()), lr=config["lr"])  # lr可从config拿
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.1)

    # === Lodad checkpoint，if it is not None ===
    checkpoint = torch.load(model_path, map_location=DEVICE)
    gnn_node_emb.load_state_dict(checkpoint['emb_model_state_dict'])
    gnn_order_dispatch.load_state_dict(checkpoint['oc_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1  # 支持续训
    loaded_config = checkpoint.get('config', None)
    if loaded_config:
        # print("Loaded config from checkpoint")
        # config.update(loaded_config)  # 慎用，或者：
        pass


    # validation
    average_result = None

    average_result = inference(gnn_node_emb, gnn_order_dispatch, DEVICE)






def inference(gnn_node_emb, gnn_order_dispatch, DEVICE):
    gnn_node_emb.eval()
    gnn_order_dispatch.eval()
    sample_results = []

    # validation dataset
    val_data = [
        {
            "num_orders" : 10,
            "num_riders" : 3,
            "seed" : 1,
            "omega": 0.01,
            "k_sparse_node_emb": 21,
        },
        {
            "num_orders": 10,
            "num_riders": 3,
            "seed": 1,
            "omega": 0.8,
            "k_sparse_node_emb": 21,
        },
        {
            "num_orders": 10,
            "num_riders": 3,
            "seed": 1,
            "omega": 0.9,
            "k_sparse_node_emb": 21,
        },
        {
            "num_orders": 10,
            "num_riders": 3,
            "seed": 1,
            "omega": 0.93,
            "k_sparse_node_emb": 21,
        },
        {
            "num_orders": 10,
            "num_riders": 3,
            "seed": 1,
            "omega": 0.99,
            "k_sparse_node_emb": 21,
        }
    ]

    # sol
    opt_pareto_x = [0.7, 2.3, 3.3, 6.7]
    opt_pareto_y = [4.22, 3.68, 3.62, 3.57]
    sample_x = []
    sample_y = []


    for item in val_data:
        num_orders = item["num_orders"]
        num_riders = item["num_riders"]
        seed = item["seed"]
        omega = item["omega"]
        k_sparse = item["k_sparse_node_emb"]

        val_instance = generate_train_data(num_orders, num_riders, device=DEVICE, seed=seed)
        loss_all = 0.0
        num = 0
        for orders, riders in val_instance:
            # preference (weight of multo-obj) therm encoding
            encoded_omega = non_uniform_thermometer_encode(omega, DEVICE)

            # prepare data and pre-processing
            pickup_coor = np.column_stack((orders['pickup_lng'], orders['pickup_lat']))
            delivery_coor = np.column_stack((orders['delivery_lng'], orders['delivery_lat']))
            rider_coor = np.column_stack((riders['courier_lng'], riders['courier_lat']))
            all_coor = np.vstack((pickup_coor, delivery_coor, rider_coor))
            all_coor = aspect_ratio_normalize(all_coor)
            all_order_coor_tensor = torch.tensor(all_coor, dtype=torch.float32).to(DEVICE)

            # embed the nodes
            pyg_node_emb = gen_pyg_data_nodes(all_order_coor_tensor, num_orders=num_orders,
                                              k_sparse=k_sparse)
            x_order_node, edge_index_order_node, edge_attr_order_node = pyg_node_emb.x, pyg_node_emb.edge_index, pyg_node_emb.edge_attr
            node_emb = gnn_node_emb(x_order_node, edge_index_order_node, edge_attr_order_node)
            orders_emb = node_emb[:2 * num_orders, :]
            riders_emb = node_emb[2 * num_orders:, :]

            # sample n times
            for i in range(100):
                # generate pyg data for order-courier dispatching
                pyg_order_courier = gen_pyg_hetero_bigraph(num_orders=num_orders,
                                                           num_riders=num_riders, order_emb=orders_emb,
                                                           rider_emb=riders_emb)
                # order dispatching env init.
                env_dispatch = HeteroOrderDispatchEnv(pyg_order_courier, gnn_order_dispatch, encoded_omega)

                # dispatching
                pyg_order_courier = env_dispatch.run_all(flg_train=False, flg_infer=True)
                dispatch_result = pyg_order_courier.edge_index_dict['order', 'assigns_to', 'rider'].cpu().numpy()

                # calculate the routing cost by calling LKH3
                rider_dict = {}
                for order_idx, rider_idx in zip(dispatch_result[0], dispatch_result[1]):
                    pickup_coor = all_coor[order_idx]
                    delivery_coor = all_coor[num_orders + order_idx]
                    if rider_idx not in rider_dict:
                        rider_dict[rider_idx] = {
                            'start': all_coor[2 * num_orders + rider_idx],
                            'tasks': []
                        }
                    rider_dict[rider_idx]['tasks'].append(
                        [pickup_coor[0], pickup_coor[1], delivery_coor[0], delivery_coor[1]]
                    )
                for rider_idx in rider_dict:
                    rider_dict[rider_idx]['num_tasks'] = len(rider_dict[rider_idx]['tasks'])
                # 转为 numpy array
                for rider_idx in rider_dict:
                    rider_dict[rider_idx]['tasks'] = np.array(rider_dict[rider_idx]['tasks'])

                # call LKH3
                work_dir = os.path.join(".", "lkh_work_dir")
                os.makedirs(work_dir, exist_ok=True)
                exec_name = "LKH-3.exe" if platform.system() == "Windows" else "LKH"
                lkh_exec = os.path.abspath(os.path.join(work_dir, exec_name))

                # results
                results = {}
                total_routing_cost = 0
                for rider_idx, rider_data in rider_dict.items():
                    if rider_data['num_tasks'] == 1:
                        cost = compute_1order_distance(rider_data)
                    elif rider_data['num_tasks'] == 2:
                        cost = compute_2order_min_distance(rider_data)
                    elif rider_data['num_tasks'] == 3:
                        cost = compute_multiorder_min_distance(rider_data)
                    else:
                        cost = LKH_solve_rider_with_retry(rider_idx, rider_data, lkh_exec, work_dir) / 1000
                    results[rider_idx] = {
                        "routing_cost": cost,
                        "num_tasks": rider_data['num_tasks']
                    }
                    total_routing_cost += cost

                f2 = total_routing_cost

                task_counts = [r['num_tasks'] for r in results.values()]
                max_tasks = max(task_counts)
                min_tasks = min(task_counts)
                avg_num_order = num_orders / num_riders
                delta = max(abs(max_tasks - avg_num_order), abs(avg_num_order - min_tasks))
                f1 = delta

                final_f = f1 * (1 - omega) + f2 * omega

                print(f"F1: {f1:.4f}, F2: {f2:.4f}, omega_f1: {omega:.4f}")

                sample_x.append(f1)
                sample_y.append(f2)
                sample_results.append((final_f,f1,f2))

    # plot
    output_pareto_plot((opt_pareto_x, opt_pareto_y), (sample_x,sample_y), flag_save=True)

    return sample_results