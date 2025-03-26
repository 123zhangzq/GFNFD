import torch
import torch.optim as optim
import numpy as np
import wandb
from data_loader import generate_train_data
from config import CONFIG
from utils import init_wandb, aspect_ratio_normalize,log_metrics, sample_uniform_per_bin, non_uniform_thermometer_encode, get_epsilon_exp
from gnn_model import OrderCourierHeteroGNN, NodeEmbedGNN
from env_instance import HeteroOrderDispatchEnv
import os
from lkh3_solver import solve_rider_with_LKH
from data_loader import gen_pyg_data_nodes, gen_pyg_hetero_bigraph
import platform
import math



def train():
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)

    BINS = CONFIG["preference_bins"]
    USE_WANDB = True
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


    # === init wandb ===
    if USE_WANDB:
        wandb.init(project="Order-Courier-Assignment", name=run_name)
        wandb.config.update(CONFIG)
        wandb.config.update({"model": "OrderCourierHeteroGNN"})

    # unify the config
    config = wandb.config if USE_WANDB else CONFIG

    # saving model path
    output_dir = config["output"]
    savepath = os.path.join(output_dir, f'{config["num_orders"]}_{config["num_riders"]}', run_name)
    os.makedirs(savepath, exist_ok=True)

    # Init. GNN 和 GFlowNet
    gnn_node_emb = NodeEmbedGNN(feats = 3).to(DEVICE)
    gnn_order_dispatch = OrderCourierHeteroGNN(order_input_dim = 65, rider_input_dim = 33, edge_attr_dim= 1,
                                                hidden_dim = 64, omega_dim=6, flg_gfn=True).to(DEVICE)

    # Init. optimizer and schheduler
    optimizer = torch.optim.AdamW(list(gnn_order_dispatch.parameters()) + list(gnn_node_emb.parameters()), lr=config["lr"])  # lr可从config拿
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.1)

    # === Lodad checkpoint，if it is not None ===
    start_epoch = 0
    if config.get("pretrained"):
        checkpoint = torch.load(config["pretrained"], map_location=DEVICE)
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



    # start training
    # best_reward = float('-inf')
    # best_solution = None
    for epoch in range(start_epoch, config["epochs"]):
        seed = epoch + config["seed"]
        train_data = generate_train_data(
            config["num_orders"],
            config["num_riders"],
            device=DEVICE,
            seed=seed,
            num_run=config["batch_size"]
        )

        # train this epoch
        gnn_node_emb.train()
        gnn_order_dispatch.train()
        loss_all = 0.0
        num = 0
        for orders, riders in train_data:
            # 采样偏好向量 ω
            omega, bin_idx = sample_uniform_per_bin(BINS, DEVICE)
            encoded_omega = non_uniform_thermometer_encode(omega[0].item(), DEVICE)

            # prepare data and pre-processing
            pickup_coor = np.column_stack((orders['pickup_lng'], orders['pickup_lat']))
            delivery_coor = np.column_stack((orders['delivery_lng'], orders['delivery_lat']))
            rider_coor = np.column_stack((riders['courier_lng'], riders['courier_lat']))
            all_coor = np.vstack((pickup_coor, delivery_coor, rider_coor))
            all_coor = aspect_ratio_normalize(all_coor)
            all_order_coor_tensor = torch.tensor(all_coor, dtype=torch.float32).to(DEVICE)

            # embed the nodes
            pyg_node_emb = gen_pyg_data_nodes(all_order_coor_tensor, num_orders=config["num_orders"], k_sparse=config["k_sparse"])
            x_order_node, edge_index_order_node, edge_attr_order_node = pyg_node_emb.x, pyg_node_emb.edge_index, pyg_node_emb.edge_attr
            node_emb = gnn_node_emb(x_order_node, edge_index_order_node, edge_attr_order_node)
            orders_emb = node_emb[:2*config["num_orders"], :]
            riders_emb = node_emb[2*config["num_orders"]:, :]

            # sample k times for this instance
            loss_list = []
            flow_Z = None
            for i in range(config["sample_k"]):
                # generate pyg data for order-courier dispatching
                pyg_order_courier = gen_pyg_hetero_bigraph(num_orders=config["num_orders"],
                                                           num_riders=config["num_riders"], order_emb=orders_emb,
                                                           rider_emb=riders_emb)

                # order dispatching env init.
                env_dispatch = HeteroOrderDispatchEnv(pyg_order_courier, gnn_order_dispatch, encoded_omega)

                if i == 0:
                    flow_Z = env_dispatch.get_logz()

                # dispatching
                epsilon_epoch = get_epsilon_exp(epoch, config["epochs"])
                pyg_order_courier, Log_PF = env_dispatch.run_all(epsilon = epsilon_epoch, flg_train=True)
                dispatch_result = pyg_order_courier.edge_index_dict['order', 'assigns_to', 'rider'].cpu().numpy()

                # calculate the routing cost by calling LKH3
                rider_dict = {}
                for order_idx, rider_idx in zip(dispatch_result[0], dispatch_result[1]):
                    pickup_coor = all_coor[order_idx]
                    delivery_coor = all_coor[config["num_orders"] + order_idx]
                    if rider_idx not in rider_dict:
                        rider_dict[rider_idx] = {
                            'start': all_coor[2 * config["num_orders"] + rider_idx],
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
                    cost = solve_rider_with_LKH(rider_idx, rider_data, lkh_exec, work_dir)
                    results[rider_idx] = {
                        "routing_cost": cost,
                        "num_tasks": rider_data['num_tasks']
                    }
                    total_routing_cost += cost

                f2 = total_routing_cost / 1000

                task_counts = [r['num_tasks'] for r in results.values()]
                max_tasks = max(task_counts)
                min_tasks = min(task_counts)
                avg_num_order = config["num_orders"] / config["num_riders"]
                delta = max(abs(max_tasks - avg_num_order), abs(avg_num_order - min_tasks))
                f1 = delta

                final_f = f1 * omega[1] + f2 * omega[0]

                # print("Current reward: ", final_f)

                # calculate tb_loss_k
                reward_k = torch.tensor(1.0 / (final_f + 1e-8), device=DEVICE)
                Log_R = torch.log(reward_k)
                Log_Z_theta = torch.log(flow_Z + 1e-8)
                forward_flow = Log_PF.sum(0) + Log_Z_theta
                backward_flow = math.log(1 / config["num_orders"]) + Log_R
                tb_loss_k = torch.pow(forward_flow - backward_flow, 2)
                loss_list.append(tb_loss_k)

            tb_loss = torch.stack(loss_list).mean()
            loss_all = loss_all + tb_loss
            num += 1


        # GFN reward
        loss = loss_all / num
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=gnn_node_emb.parameters(), max_norm=3.0, norm_type=2)
        torch.nn.utils.clip_grad_norm_(parameters=gnn_order_dispatch.parameters(), max_norm=3.0, norm_type=2)
        optimizer.step()

        # Update the schedular
        scheduler.step()

        if epoch % 20 == 0:
            average_result = validate(gnn_node_emb, gnn_order_dispatch, DEVICE)

        # 保存完整checkpoint
        if epoch >= 20 and epoch % 20 == 0:
            checkpoint_path = os.path.join(savepath, f"checkpoint_epoch{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'emb_model_state_dict': gnn_node_emb.state_dict(),
                'oc_model_state_dict': gnn_order_dispatch.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
            }, checkpoint_path)


        # wandb日志
        if USE_WANDB:
            wandb.log({"epoch": epoch, "lr": scheduler.get_last_lr()[0], "loss":loss})




        # # 记录最优解
        # if total_reward > best_reward:
        #     best_reward = total_reward
        #     best_solution = actions.detach().cpu().numpy()
        #


        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {loss.item():.4f}")


def validate(gnn_node_emb, gnn_order_dispatch, DEVICE):
    gnn_node_emb.eval()
    gnn_order_dispatch.eval()

    # validation dataset
    val_data = [
        {
            "num_orders" : 10,
            "num_riders" : 3,
            "seed" : 1,
            "emega": 0.01,
            "k_sparse_node_emb": 21,
        },
        {
            "num_orders": 10,
            "num_riders": 3,
            "seed": 1,
            "emega": 0.75,
            "k_sparse_node_emb": 21,
        }
        ,
        {
            "num_orders": 10,
            "num_riders": 3,
            "seed": 1,
            "emega": 0.85,
            "k_sparse_node_emb": 21,
        },
        {
            "num_orders": 10,
            "num_riders": 3,
            "seed": 1,
            "emega": 0.95,
            "k_sparse_node_emb": 21,
        },
        {
            "num_orders": 10,
            "num_riders": 3,
            "seed": 2,
            "emega": 0.79,
            "k_sparse_node_emb": 21,
        },
        {
            "num_orders": 10,
            "num_riders": 3,
            "seed": 3,
            "emega": 0.79,
            "k_sparse_node_emb": 21,
        }
        ,
        {
            "num_orders": 10,
            "num_riders": 3,
            "seed": 3,
            "emega": 0.89,
            "k_sparse_node_emb": 21,
        },
        {
            "num_orders": 10,
            "num_riders": 3,
            "seed": 4,
            "emega": 0.69,
            "k_sparse_node_emb": 21,
        },
        {
            "num_orders": 10,
            "num_riders": 3,
            "seed": 4,
            "emega": 0.79,
            "k_sparse_node_emb": 21,
        },
        {
            "num_orders": 10,
            "num_riders": 3,
            "seed": 5,
            "emega": 0.89,
            "k_sparse_node_emb": 21,
        }
    ]

    for item in val_data:
        num_orders = item["num_orders"]
        num_riders = item["num_riders"]
        seed = item["seed"]
        omega = item["emega"]
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

            # keep the same structure as train
            for i in range(1):
                # generate pyg data for order-courier dispatching
                pyg_order_courier = gen_pyg_hetero_bigraph(num_orders=num_orders,
                                                           num_riders=num_riders, order_emb=orders_emb,
                                                           rider_emb=riders_emb)
                # order dispatching env init.
                env_dispatch = HeteroOrderDispatchEnv(pyg_order_courier, gnn_order_dispatch, encoded_omega)

                # dispatching
                pyg_order_courier = env_dispatch.run_all(flg_train=False)
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
                    cost = solve_rider_with_LKH(rider_idx, rider_data, lkh_exec, work_dir)
                    results[rider_idx] = {
                        "routing_cost": cost,
                        "num_tasks": rider_data['num_tasks']
                    }
                    total_routing_cost += cost

                f2 = total_routing_cost / 1000

                task_counts = [r['num_tasks'] for r in results.values()]
                max_tasks = max(task_counts)
                min_tasks = min(task_counts)
                avg_num_order = num_orders / num_riders
                delta = max(abs(max_tasks - avg_num_order), abs(avg_num_order - min_tasks))
                f1 = delta

                final_f = f1 * (1 - omega) + f2 * omega

                item["result"] = final_f

    results = [item["result"] for item in val_data if "result" in item]
    average_result = sum(results) / len(results) if results else None

    print("Average result of the validation dataset:", average_result)

    return average_result