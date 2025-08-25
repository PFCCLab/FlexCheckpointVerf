# 简单 MLP 网络

第1步：保存初始 Checkpoint (使用2张GPU)

```
python -m paddle.distributed.launch --gpus="0,1" verf.py --step 1
```

第2步：加载并重保存 Checkpoint (使用2张GPU)

```
python -m paddle.distributed.launch --gpus="0,1,2,3" verf.py --step 2
```

第3步：执行逐参数MD5对齐验证 (使用1张GPU)

```
python -m paddle.distributed.launch --gpus="0" verf.py --step 3
```
