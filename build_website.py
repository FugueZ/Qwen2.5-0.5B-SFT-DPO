import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import gradio as gr
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def respond(message, chat_history):
    """定义多轮对话的响应函数"""

    # 构建格式
    messages=[]
    for usr_msg,bot_msg in chat_history:
        messages.append({
            'content':usr_msg,
            'role':'user'
        })
        messages.append({
            'content':bot_msg,
            'role':'assistant'
        })
    messages.append({
            'content':message,
            'role':'user'
        })

    # 模型推理
    inputs=tokenizer.apply_chat_template(messages,tokenize=True,return_tensors='pt',
                                         return_dict=True,add_generation_prompt=True).to(model.device)
    outputs = model.generate(**inputs, max_length=1024, pad_token_id=tokenizer.eos_token_id)
    sentence_len=len(inputs['input_ids'][0])
    response_ids=outputs[0][sentence_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    # 当前对话加入历史
    chat_history.append((message,response))

    return "",chat_history


if __name__=='__main__':
    
    # 加载模型和分词器
    instruct_model_path=r"./result/Qwen2.5-0.5B-SFT" # 微调权重
    dpo_model_path=r"./result/Qwen2.5-0.5B-DPO" # DPO权重
    tokenizer = AutoTokenizer.from_pretrained(instruct_model_path)
    model=PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(instruct_model_path,torch_dtype="auto",device_map="auto"),
        dpo_model_path)

    # 创建 Gradio 界面
    with gr.Blocks() as demo:
        gr.Markdown("# Qwen3.5-0.5B SFT+DPO")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="输入")
        clear = gr.ClearButton([msg, chatbot])

        # 提交按钮触发响应函数
        msg.submit(respond, [msg, chatbot], [msg, chatbot])

    # 启动应用
    demo.launch()
