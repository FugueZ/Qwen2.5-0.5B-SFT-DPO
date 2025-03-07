import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline,
)


if __name__ == "__main__":
    # 加载数据集与预训练模型
    fine_tuing_dataset = load_dataset(
        "trl-lib/Capybara", split="train"
    )  # 多轮对话数据集
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # 分词器
    pretrain_model = AutoModelForCausalLM.from_pretrained(
        model_name,
    )  # 预训练模型

    # 使用lora微调
    lora_config = LoraConfig(
        peft_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    lora_model = get_peft_model(pretrain_model, lora_config)

    # 训练参数
    training_args = SFTConfig(
        output_dir="./result/Qwen2.5-0.5B-SFT",
        group_by_length=True,
        per_device_train_batch_size=2, # 训练时每个GPU加载的batch大小
        gradient_accumulation_steps=4, # 梯度更新的间隔步数
        logging_steps=100,   # 记录日志的步数
        learning_rate=1e-5,  # 初始学习率
        weight_decay=0.001,  # 权重衰减率
        max_grad_norm=0.3,  # 裁剪梯度
        warmup_ratio=0.03,  # 训练开始时的预热样本比例
        lr_scheduler_type="cosine",  # 学习率调度器将使用常数衰减策略
    )

    # 加载训练器
    trainer = SFTTrainer(
        args=training_args,
        model=lora_model,
        processing_class=tokenizer,
        train_dataset=fine_tuing_dataset,
    )

    # 训练
    trainer.train(resume_from_checkpoint="last-checkpoint")
