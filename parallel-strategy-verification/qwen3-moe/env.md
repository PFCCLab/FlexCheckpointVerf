# 环境配置说明

Qwen3-Moe 模型依赖 PaddleNLP

首先拉取 PaddleNLP 的代码仓库

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
```

然后安装相关依赖包：

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

然后 cd 到 llm 路径下，将本目录下 `scripts/run_qwen3.sh` 的脚本复制到 `PaddleNLP/llm` 目录下

