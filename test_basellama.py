from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from safetensors.torch import load_file
import torch


# 文件路径
input_tensor_path_0 = "D:/compare/test_case_cpu_0_input.pt"
logits_tensor_path_0 = "D:/compare/test_case_cpu_0_logits.pt"
input_tensor_path_1 = "D:/compare/test_case_cpu_1_input.pt"
logits_tensor_path_1 = "D:/compare/test_case_cpu_1_logits.pt"

# 加载输入和输出张量
input_tensor_0 = torch.load(input_tensor_path_0)
expected_logits_0 = torch.load(logits_tensor_path_0)
input_tensor_1 = torch.load(input_tensor_path_1)
expected_logits_1 = torch.load(logits_tensor_path_1)

# 定义模型路径和权重文件路径
model_localpath = "E:/weightfile/llama7b"
safetensors_files = [
    f"{model_localpath}/model-00001-of-00002.safetensors",
    f"{model_localpath}/model-00002-of-00002.safetensors"
]

# 加载所有分片文件中的权重
state_dict = {}
for file_path in safetensors_files:
    state_dict.update(load_file(file_path))

# 提取和加载 scale_blockwise 数据
scale_blockwise_dict = {k: v for k, v in state_dict.items() if 'scale_blockwise' in k}
filtered_state_dict = {k: v for k, v in state_dict.items() if 'scale_blockwise' not in k}

# 创建自定义配置
llamaconfig = LlamaConfig(attn_implementation="eager",attention_bias=True)

# 初始化模型
model = LlamaForCausalLM.from_pretrained(model_localpath, config=llamaconfig, state_dict=filtered_state_dict, use_safetensors=True)
model.half()
# 将 scale_blockwise 数据加载到模型中
for name, param in model.named_parameters():
    if name in scale_blockwise_dict:
        param.data.copy_(scale_blockwise_dict[name])

# Manually initialize missing bias terms
for name, param in model.named_parameters():
    if "bias" in name and param.requires_grad:
        if param.data.numel() == 0:  # Check if bias is uninitialized
            param.data = torch.zeros_like(param.data, dtype=torch.float16)


# 将所有偏置项初始化为 0.1
for name, param in model.named_parameters():
    if "bias" in name and param.requires_grad:
        param.data.fill_(0.1)  # 将偏置项的值设为0.1
        print(f"偏置项 '{name}' 已初始化为0.1")



# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 生成输出
inputs = input_tensor_0
generate_ids = model.generate(input_ids=inputs, max_length=50,do_sample=True)

# 打印输出
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])