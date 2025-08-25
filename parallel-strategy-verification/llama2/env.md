# 环境配置说明

## 1.vpn准备

由于AIstudio下载速度太慢，需要安装梯子，因此先下载群里的 clash-main，并且用以下命令给予文件夹权限：

```bash
chmod -777 clash-main
```

启动方式：

```
./clash -d ./
```

保持 clash 进程运行，使用如下命令切换到代理模式：

```
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
```

## 2.paddle环境准备

拉取自己的paddle仓库，并更新到最新：

```bash
git clone https://github.com/xxxx/Paddle.git
git pull upstream develop
```

拉取临时pr:
```bash
git fetch upstream pull/74785/head:pr-Support_AOA_for_load_state_dict
git switch pr-Support_AOA_for_load_state_dict
```


下载cuda=12.6最新版本的paddle：

```bash
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu126/ --force-reinstall --no-deps
```

用以下脚本替换 flexckpt 更新的代码，因为 aistdio 中没法编译只能手动拷贝需要修改的文件（按需修改/添加）

```bash
# 原文件目录和目标文件目录
source_dir="/home/aistudio/Paddle/python/paddle"
target_dir="/home/aistudio/external-libraries/lib/python3.10/site-packages/paddle"

# 要替换的文件列表
files=(
    "distributed/fleet/meta_optimizers/dygraph_optimizer/dygraph_sharding_optimizer.py"
    "distributed/flex_checkpoint/aoa/aoa_engine.py"
    "distributed/flex_checkpoint/aoa/lexer.py"
    "distributed/flex_checkpoint/aoa/macros.py"
    "distributed/flex_checkpoint/dcp/load_state_dict.py"
    "distributed/flex_checkpoint/dcp/metadata.py"
    "distributed/flex_checkpoint/dcp/reshard.py"
    "distributed/flex_checkpoint/dcp/save_state_dict.py"
    "distributed/flex_checkpoint/dcp/sharded_weight.py"
    "distributed/flex_checkpoint/dcp/utils.py"
    "optimizer/adamw.py"
)

# 替换文件
for file in "${files[@]}"; do
    # 拼接源文件和目标文件的完整路径
    source_file="$source_dir/$file"
    target_file="$target_dir/$file"

    # 检查源文件是否存在
    if [ -f "$source_file" ]; then
        echo "Replacing $target_file"
        # 确保目标文件的目录存在
        mkdir -p "$(dirname "$target_file")"
        # 替换文件
        cp "$source_file" "$target_file"
    else
        echo "Source file $source_file not found!"
        echo "Creating empty file at $target_file"
        # 确保目标文件的目录存在
        mkdir -p "$(dirname "$target_file")"
        # 创建一个空文件并复制
        touch "$target_file"
    fi
done

echo "Replacement completed!"

```

## 3. Qwen3-Moe 模型依赖 PaddleNLP

首先拉取 PaddleNLP 的代码仓库（fork后，拉取自己本地仓库的paddleNLP）

一个示例

```bash
git clone https://github.com/zty-king/PaddleNLP.git
cd PaddleNLP
```

然后安装相关依赖包：

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

添加远程分支

```bash
git remote add upstream https://github.com/PaddlePaddle/PaddleNLP.git
```

拉取临时pr
```bash
git fetch upstream pull/10996/head:pr-adapt_flex_checkpoint
git switch pr-adapt_flex_checkpoint
```

切换到对应的分支
```bash
git switch pr-adapt_flex_checkpoint
```

## 4. 运行代码

进入 paddlenlp/llm/

```bash
cd paddlenlp/llm/
```

数据准备

项目提供了预先处理好的数据方便用户测试模型，下载到 `data` 目录下：

```shell
mkdir -p data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.{bin,idx}
```

```bash
bash run_pretrain_llm.sh
```



