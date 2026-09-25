# AutoJev

**Automated Jev-style decision model training based on Laya.**

把 JSONL/CSV 数据集自动转换为 Laya typed-decision 数据，完成分组切分、微调、温度校准、测试评估和可直接加载的模型导出。

AutoJev 是一个社区开源项目，并非 TypeSafe 或 Laya 的官方项目。它基于 Laya
公开能力构建自动化训练流水线，不声称复现 Jev 的私有训练配方。

## 特性

- 支持 Jev/System One 风格 JSONL，以及扁平 JSONL/CSV。
- 同一会话通过 `group_id` 分组切分，避免问题级随机切分造成数据泄漏。
- 支持 `choice`、`score`、`noul` 和硬标签/软概率标签。
- 支持普通监督训练，以及与 Laya 官方 Notebook 同类的 RLCD 目标。
- 校准集不参与训练，分别为三种问题类型拟合 temperature。
- 输出 Accuracy、NLL、Brier、ECE，并保存数据指纹和完整训练配置。
- 导出的目录可由 `laya.load("/path/to/model")` 直接加载。

## 安装

建议使用带 NVIDIA GPU 的 Linux 环境。CPU/MPS 可用于小规模验证，但训练会慢很多。

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

首次运行会从 Hugging Face 下载基础模型。生产环境可将 `model` 配置为已下载的本地目录。

## 数据格式

推荐一个 JSONL 行代表一段会话，多个问题共享同一个 `id`：

```json
{
  "id": "ticket-001",
  "state": {"conversation": [{"role": "customer", "text": "钱扣了两次"}]},
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "用户的主要诉求是什么？",
      "criteria": {
        "duplicate_charge": "同一笔交易被重复扣款",
        "refund": "明确要求退款"
      }
    },
    "urgent": {"type": "noul", "instructions": "是否需要立即升级人工？"}
  },
  "answers": {
    "intent": {"choice": "duplicate_charge"},
    "urgent": {"noul": true}
  }
}
```

软标签写法：

```json
{"probabilities": {"duplicate_charge": 0.7, "refund": 0.3}}
```

扁平 CSV/JSONL 支持这些列：

```text
group_id, question_id, state/text, question_type/type,
instructions/question, criteria, label/answer/probabilities
```

CSV中的 `criteria` 和 `probabilities` 使用JSON字符串。

## 使用

先校验数据：

```bash
laya-autofinetune validate --input data/customer_service.jsonl
```

一条命令完成全部流程：

```bash
laya-autofinetune run \
  --input data/customer_service.jsonl \
  --output runs/customer-service-v1 \
  --config examples/config.json
```

也可以分两步运行：

```bash
laya-autofinetune prepare \
  --input data/customer_service.jsonl \
  --output runs/prepared \
  --config examples/config.json

laya-autofinetune train \
  --prepared runs/prepared \
  --output runs/model \
  --config examples/config.json
```

加载导出的模型：

```python
import laya

agent = laya.load("runs/customer-service-v1/model", device="cuda")
result = agent.system_one(
    "钱扣了两次，订单还没生成",
    {
        "intent": {
            "type": "choice",
            "instructions": "用户的主要诉求是什么？",
            "criteria": {
                "duplicate_charge": "同一笔交易被重复扣款",
                "missing_order": "付款后没有生成订单"
            }
        }
    },
)
print(result)
```

## 训练目标

默认使用 `supervised`，即软标签交叉熵。它稳定、成本低，建议作为第一版。

要启用与上游 Notebook 同类的探索噪声和 proper-scoring-rule 奖励：

```json
{
  "objective": "rlcd",
  "rlcd_weight": 1.0,
  "group_size": 4,
  "sigma_start": 0.4,
  "sigma_end": 0.1
}
```

这并不声称复现 TypeSafe Jev 的私有 RLCD 配方；它实现的是 Laya 已公开的训练方法。

## 输出目录

```text
output/
├── prepared/
│   ├── train.jsonl
│   ├── calibration.jsonl
│   ├── test.jsonl
│   └── manifest.json
└── model/
    ├── model.safetensors
    ├── rl_agent_config.json
    ├── encoder/
    ├── tokenizer/
    ├── dataset_manifest.json
    └── training_report.json
```

`training_report.json` 包含训练损失、校准温度、最终指标、跳过样本和可复现配置。

## 生产注意事项

- 至少准备三个不同 `group_id`；真实训练应有数百至数千段会话。
- 不要将同一对话的不同问题分散到训练集和测试集。
- `calibration` 集只用于拟合概率，不能参与梯度训练。
- 多语言基础模型的原始置信度不能直接用作自动化阈值。
- 脱敏姓名、手机号、证件号、银行卡和地址后再构造训练集。
- 超过20个相近标签时优先采用分层分类或候选召回。
