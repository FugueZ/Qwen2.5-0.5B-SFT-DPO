import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__=="__main__":
    # 加载数据集和微调后的模型
    instruct_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    instruct_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    dpo_train = load_dataset("trl-lib/ultrafeedback_binarized",split='train')
    dpo_test = load_dataset("trl-lib/ultrafeedback_binarized",split='test')

    # 使用lora微调
    lora_config=LoraConfig(
        peft_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj','v_proj']
    )  # lora参数

    dpo_lora_model=get_peft_model(instruct_model,lora_config) # lora模型

    # 训练参数
    training_args = DPOConfig(
        output_dir="./result/Qwen2.5-0.5B-DPO",
        per_device_train_batch_size=2, # 训练时每个GPU加载的batch大小
        per_device_eval_batch_size=2, # 评价时每个GPU加载的batch大小
        gradient_accumulation_steps=2,  # 梯度更新的间隔步数
        eval_strategy='steps', # 评估的策略
        logging_steps=10, # 记录日志的间隔
        eval_steps=10, # 评估的间隔
        weight_decay=0.001 # 权重衰减
    )

    # 加载训练器
    trainer = DPOTrainer(
        model=dpo_lora_model,
        args=training_args,
        train_dataset=dpo_train,
        eval_dataset=dpo_test,
        processing_class=instruct_tokenizer
    )

    # 训练
    trainer.train()


