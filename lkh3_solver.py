import subprocess
import re
import os


def generate_problem_and_par(rider_idx, rider_data, save_path):
    start = rider_data['start'] * 1000
    tasks = rider_data['tasks'] * 1000
    num_orders = tasks.shape[0]
    N = 1 + num_orders * 2  # depot + pickups + deliveries
    lines = []
    lines.append(f"NAME : rider_{rider_idx}")
    lines.append("TYPE : PDTSP")
    lines.append(f"DIMENSION : {N}")
    lines.append("EDGE_WEIGHT_TYPE : EUC_2D")
    lines.append("NODE_COORD_SECTION")
    lines.append(f"1 {start[0]} {start[1]}")

    # 先写pickup点
    for idx, task in enumerate(tasks):
        pickup_x, pickup_y, _, _ = task
        lines.append(f"{2 + idx} {pickup_x} {pickup_y}")

    # 再写delivery点
    for idx, task in enumerate(tasks):
        _, _, delivery_x, delivery_y = task
        lines.append(f"{2 + num_orders + idx} {delivery_x} {delivery_y}")


    lines.append("PICKUP_AND_DELIVERY_SECTION")

    # depot 行，按照你的格式 1 0 0 0 0 0 0
    lines.append("1 0 0 0 0 0 0")

    # pickup和delivery对应关系
    for idx in range(num_orders):
        pickup_node = 2 + idx
        delivery_node = 2 + num_orders + idx
        # pickup 行：pickup点编号，最后一列是对应delivery点
        lines.append(f"{pickup_node} 0 0 0 0 0 {delivery_node}")
    for idx in range(num_orders):
        pickup_node = 2 + idx
        delivery_node = 2 + num_orders + idx
        # delivery 行：delivery点编号，倒数第二列是对应pickup点
        lines.append(f"{delivery_node} 0 0 0 0 {pickup_node} 0")

    # DEPOT_SECTION 最后
    lines.append("DEPOT_SECTION")
    lines.append("1")
    lines.append("-1")
    lines.append("EOF")

    # 写入 .pdtsp 文件
    pdtsp_file = os.path.join(save_path, f"rider_{rider_idx}.pdtsp")
    with open(pdtsp_file, 'w') as f:
        f.write("\n".join(lines))

    # 写入 .par 文件
    par_file = os.path.join(save_path, f"rider_{rider_idx}.par")
    with open(par_file, 'w') as f:
        f.write(f"PROBLEM_FILE = rider_{rider_idx}.pdtsp\n")
        f.write(f"TOUR_FILE = rider_{rider_idx}.tour\n")

    return par_file


def solve_rider_with_LKH(rider_idx, rider_data, lkh_exec, work_dir):
    # 生成 .tsp 和 .par 文件
    par_file = generate_problem_and_par(rider_idx, rider_data, work_dir)

    # TODO:1, how to get the results. 2, del all the files after run
    while True:
        print('finish writing lkh file!')
        pass


    # 正确执行 LKH3：传入 .par 文件
    result = subprocess.run([lkh_exec, par_file], cwd=work_dir, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    print(f"Return code: {result.returncode}")

    # 读取tour文件
    tour_file = os.path.join(work_dir, f"rider_{rider_idx}.tour")
    with open(tour_file, 'r') as f:
        lines = f.readlines()
    tour = []
    start = False
    for line in lines:
        if "TOUR_SECTION" in line:
            start = True
            continue
        if "-1" in line:
            break
        if start:
            tour.append(int(line.strip()))

    # 读取log获取cost
    log_file = os.path.join(work_dir, f"rider_{rider_idx}.log")
    with open(log_file, 'r') as f:
        log_content = f.read()
    match = re.search(r"Cost\.min = ([\d\.]+)", log_content)
    cost = float(match.group(1)) if match else None

    return tour, cost