# Ernie4.5 纯文

验证方式：

1. ckpt1变换到ckpt2，再变换回ckpt1，md5与原始ckpt对齐，且均可正常训练，loss收敛或逐位对齐
2. ckpt1变换到ckpt2，再分别合成开源格式权重，两份开源权重md5对齐，且均可正常推理，loss逐位对齐

loss 收敛趋势图需要用 ckpt1 训 20 个 step， load 成 ckpt 2 继续训 100 个 step，loss 图用不同颜色区分。


| 模型                 |  dp2    | dp4  | sharding4（v1） | sharding4（v2） | tp4  | pp4  | ep4  |
| -------------------- | ---- | ---- | --------------- | --------------- | ---- | ---- | ---- |
| EB4.5 纯文 |   |      |                 |                 |      |      |      |
| sharding2（v1）      |      |      |                 |                 |      |      |      |
| sharding2（v2）      |      |      |                 |                 |      |      |      |
| tp2                  |      |      |                 |                 |      |      |      |
| pp2                  |      |      |                 |                 |      |      |      |
| ep2                  |      |      |                 |                 |      |      |      |

## 1. dp2->dp4

ckpt 存储路径：

验证脚本路径：

loss 收敛趋势图：


## 2. dp2->sharding4（v1）

ckpt 存储路径：

验证脚本路径：

loss 收敛趋势图：

## 3. dp2->sharding4（v2）

ckpt 存储路径：

验证脚本路径：

loss 收敛趋势图：