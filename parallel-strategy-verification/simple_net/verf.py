import os
import argparse
import hashlib
import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.nn import Layer, ReLU
from paddle.distributed.fleet.layers.mpu import (
    ColumnParallelLinear,
    RowParallelLinear,
)

# 定义一个简单的、使用张量并行的MLP模型
class SimpleMLP(Layer):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.linear_1 = ColumnParallelLinear(
            hidden_size, hidden_size * 2, has_bias=True, gather_output=False
        )
        self.relu = ReLU()
        self.linear_2 = RowParallelLinear(
            hidden_size * 2, hidden_size, has_bias=True, input_is_parallel=True
        )

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x

# 封装分布式环境初始化
def setup_dist_env():
    fleet.init(is_collective=True)


# [第1步] 初始化模型并保存初始的 checkpoint
def run_step1_save_initial(args):
    setup_dist_env()
    tp_rank = dist.get_rank()
    model = SimpleMLP()
    print(f"[Step 1] Rank {tp_rank}/2: Saving initial TP=2 model checkpoint.")
    sharded_state_dict = model.sharded_state_dict()
    dist.save_state_dict(sharded_state_dict, args.initial_ckpt_path)
    print(f"[Step 1] Rank {tp_rank}/2: Initial checkpoint saved to '{args.initial_ckpt_path}'.")

# [第2步] 加载初始 checkpoint，并立即重新保存
def run_step2_load_and_resave(args):
    setup_dist_env()
    tp_rank = dist.get_rank()
    model = SimpleMLP()
    print(f"[Step 2] Rank {tp_rank}/2: Loading initial checkpoint.")
    sharded_state_dict = model.sharded_state_dict()
    dist.load_state_dict(sharded_state_dict, args.initial_ckpt_path)
    print(f"[Step 2] Rank {tp_rank}/2: Resaving the loaded model to a new path.")
    dist.save_state_dict(sharded_state_dict, args.resaved_ckpt_path)
    print(f"[Step 2] Rank {tp_rank}/2: Resaved checkpoint saved to '{args.resaved_ckpt_path}'.")

# [第3步] 逐一对比每个参数的 MD5 哈希值
def run_step3_verify_per_parameter_md5(args):
    # 第3步需要在单卡上执行，以加载完整的逻辑权重进行对比
    setup_dist_env()
    
    # 辅助函数：计算单个 paddle.Tensor 的 MD5
    def get_tensor_md5(tensor):
        # 使用 .numpy().tobytes() 获取确定性的字节表示
        tensor_bytes = tensor.numpy().tobytes()
        md5_hash = hashlib.md5()
        md5_hash.update(tensor_bytes)
        return md5_hash.hexdigest()

    print("[Step 3] Starting per-parameter MD5 verification on a single GPU...")

    # 1. 加载初始 checkpoint 到单卡模型中
    # 注意：这里我们用单卡配置去加载一个2卡TP的checkpoint，
    # 这本身也利用了FlexCheckpoint的自动聚合能力。
    print(f"  Loading initial checkpoint from '{args.initial_ckpt_path}'...")
    model_initial = SimpleMLP() # 模型在单卡上初始化
    sharded_state_dict_initial = model_initial.sharded_state_dict()
    dist.load_state_dict(sharded_state_dict_initial, args.initial_ckpt_path)
    state_dict_initial = model_initial.state_dict()
    print("  Initial checkpoint loaded.")

    # 2. 加载重保存的 checkpoint 到另一个单卡模型中
    print(f"  Loading resaved checkpoint from '{args.resaved_ckpt_path}'...")
    model_resaved = SimpleMLP() # 模型在单卡上初始化
    sharded_state_dict_resaved = model_resaved.sharded_state_dict()
    dist.load_state_dict(sharded_state_dict_resaved, args.resaved_ckpt_path)
    state_dict_resaved = model_resaved.state_dict()
    print("  Resaved checkpoint loaded.")

    # 3. 逐一计算并对比每个参数的 MD5
    print("\n--- Verification Result ---")
    all_passed = True
    # 以初始 state_dict 的 keys 为准
    for key, initial_tensor in state_dict_initial.items():
        print(f"Verifying parameter: {key}")
        if key not in state_dict_resaved:
            print(f"  ❌ FAILED: Key '{key}' not found in resaved state_dict!")
            all_passed = False
            continue

        resaved_tensor = state_dict_resaved[key]
        
        initial_md5 = get_tensor_md5(initial_tensor)
        resaved_md5 = get_tensor_md5(resaved_tensor)
        
        print(f"  - Initial MD5 : {initial_md5}")
        print(f"  - Resaved MD5 : {resaved_md5}")
        
        if initial_md5 == resaved_md5:
            print("  - ✅ PASSED: MD5 hashes match.")
        else:
            print("  - ❌ FAILED: MD5 hashes DO NOT match!")
            all_passed = False
            
    if all_passed:
        print("\n✅ Per-Parameter MD5 Alignment Verification Successful! All parameters are bit-for-bit identical.\n")
    else:
        print("\n❌ Per-Parameter MD5 Alignment Verification FAILED! Some parameters are different.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3], required=True, help="Which step of the test to run."
    )
    parser.add_argument(
        "--initial_ckpt_path", type=str, default="./flex_ckpt_md5/tp2", help="Path for the initial checkpoint."
    )
    parser.add_argument(
        "--resaved_ckpt_path", type=str, default="./flex_ckpt_md5/tp4", help="Path for the resaved checkpoint."
    )
    args = parser.parse_args()

    # 只需要在第3步创建目录，前两步由 save_state_dict 自动创建
    if args.step == 1:
        run_step1_save_initial(args)
    elif args.step == 2:
        run_step2_load_and_resave(args)
    elif args.step == 3:
        run_step3_verify_per_parameter_md5(args)