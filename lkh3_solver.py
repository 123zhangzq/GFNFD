import subprocess
import re
import os


def generate_problem_and_par(rider_idx, rider_data, save_path, flg_2runs = False):
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
        # if N < 6:
        #     f.write(f"RUNS = 2\n")
        #     f.write("TRACE_LEVEL = 1\n")
        #     f.write("MOVE_TYPE = 2\n")
        #     f.write("PATCHING_C = 0\n")
        #     f.write("PATCHING_A = 0\n")
        #     f.write("GAIN23 = NO\n")
        #     f.write("BACKTRACKING = NO\n")
        #     f.write("MAX_CANDIDATES = 0\n")
        if flg_2runs:
            f.write(f"SUBGRADIENT = NO\n")
            f.write(f"RUNS = 2\n")
        else:
            f.write(f"SUBGRADIENT = NO\n")
            f.write(f"RUNS = 5\n")

    return par_file, pdtsp_file


def solve_rider_with_LKH(rider_idx, rider_data, lkh_exec, work_dir, flg_cannot_run=None):
    # 生成 .tsp 和 .par 文件
    if not flg_cannot_run[0]:
        par_file, pdtsp_file = generate_problem_and_par(rider_idx, rider_data, work_dir)
    else:
        par_file, pdtsp_file = generate_problem_and_par(rider_idx, rider_data, work_dir, flg_2runs = True)

    # 正确执行 LKH3：传入 .par 文件
    assert os.path.exists(lkh_exec), "LKH 执行文件不存在"
    assert os.path.exists(work_dir), "工作目录不存在"

    try:
        result = subprocess.run(
        [lkh_exec, os.path.basename(par_file)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=30
        )
    except subprocess.TimeoutExpired as e:
        print(f"Warning: LKH run timed out. {e}")
        flg_cannot_run[0] = True
        return None


    # print(result.stdout)
    # print(result.stderr)
    # print(f"Return code: {result.returncode}")

    # 匹配 Best PDTSP solution 里的 cost

    stdout_text = result.stdout
    match = re.search(r'Best PDTSP solution:\s+Cost = -?\d+_(\d+)', stdout_text)

    if match:
        cost = int(match.group(1))
        # print(f"Finish the LKH3, Cost = {cost}")
    else:
        flg_cannot_run[0] = True
        return None

    os.remove(par_file)
    os.remove(pdtsp_file)


    return cost


def LKH_solve_rider_with_retry(rider_idx, rider_data, lkh_exec, work_dir):
    for attempt in range(2):  # 最多两次尝试
        flg_cannot_run = [False]
        cost = solve_rider_with_LKH(rider_idx, rider_data, lkh_exec, work_dir, flg_cannot_run)
        if not flg_cannot_run[0]:
            return cost  # 2nd try success
        else:
            print(f"Try LKH {attempt + 1} times，retry..." if attempt == 0 else "LKH fail!!!")

    # 如果两次都失败
    raise RuntimeError(f"LKH3 failed twice for rider {rider_idx}")