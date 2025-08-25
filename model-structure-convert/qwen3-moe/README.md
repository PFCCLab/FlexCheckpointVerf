# Qwen3 Moe 结构转换验证

1. ckpt1变换到ckpt2，再变换回ckpt1，md5与原始ckpt对齐，且均可正常训练，loss收敛或逐位对齐
2. ckpt1变换到ckpt2，再分别合成开源格式权重，两份开源权重md5对齐，且均可正常推理，loss逐位对齐

目前先验第一点即可：

loss 收敛趋势图需要用 ckpt1 训 50 个 step， load 成 ckpt 2 继续训 200 个 step，loss 图用不同颜色区分。

| 验证策略 | 验证结果 |
| :--- | :--- |
| fused＿qkv $\leftrightarrow$ 非 fused／fused ffn $\leftrightarrow$ 非 fused |  |
| 专家合并 |  |
| 专家拆分 |  |
| 参数重新命名 |  |
| 层删除 |  |
| 层添加 |  |

