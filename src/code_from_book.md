
# 第一部分：算法实现和数据清洗

## 算法 2.1  LoRA算法实现示例
```python
class LoRA(nn.Linear):
    def __init__(self, in_features: int,  out_features: int,
        lora_rank: int,      # LoRA的秩，即低秩分解中使用的秩（r）
        lora_alpha: int = 1, # LoRA的缩放因子alpha，调整低秩矩阵的影响力
        lora_dropout: float = 0.0,   # 在应用LoRA时使用的dropout的概率
        **kwargs ):
        super().__init__(in_features, out_features, **kwargs)
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.scaling = float(lora_alpha) / float(lora_rank)

        # 创建可训练的参数：LoRA的A、B矩阵
        self.lora_A = nn.Parameter(torch.zeros((in_features, lora_rank)))
        self.lora_B = nn.Parameter(torch.zeros((lora_rank, out_features)))
        
        # （1）用kaiming均匀分布随机初始化A； （2）用0初始化B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))   
        nn.init.zeros_(self.lora_B)                             
        
        self.weight.requires_grad = False   # 冻结预训练的权重矩阵
        self.if_weights_merged = False      # 本类中忽略权重合并的相关代码

    def forward(self, x: torch.Tensor):
        if not self.if_weights_merged:
            # （1）计算原始模型参数的输出(X*W):
            result = F.linear(x, self.weight, bias=self.bias)
            # （2）计算LoRA模块的输出(X*A*B*scaling):
            lora_output = self.lora_dropout(x) @ self.lora_A
            lora_output = lora_output @ self.lora_B
            result += lora_output * self.scaling # 缩放LoRA模块的输出
            return result
        else:   # 如果已经合并权重（W + A*B*scaling），直接返回线性变换的结果
            return F.linear(x, self.weight, bias=self.bias)
```

## 代码 2.1  根据隐藏状态生成Logits

```python

# 定义lm_head，线性层将隐藏状态映射到词表大小
lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

# 模型进行前向传播，获取模型输出
outputs = model( input_ids=input_ids,              # 输入的Token ID序列
                 attention_mask=attention_mask,    # 注意力掩码
                 position_ids=position_ids,        # 位置ID序列
                 **kwargs)                         # 其他额外参数

# 提取模型的隐藏状态（通常是最后一层的隐藏状态）
hidden_states = outputs[0]  # 输出的第一个张量通常是hidden states

# 使用lm_head将隐藏状态映射到词表空间，得到logits
logits = lm_head(hidden_states)

```



## 代码 2.2  基于jq处理SFT数据的示例
```bash
# 将json格式转换为jsonl格式（每行是一个完整的json结构）
jq -c '.[]' demo.json > demo_new.jsonl

# 数据清洗（去除无效或不完整的数据）
jq -c 'select(.input_text != null and .input_text != "" and .target_text != null and .target_text != "")' demo_new.jsonl > demo_cleaned.jsonl

# 重命名字段名（input_text->prompt, target_text->response）
jq -c '{prompt: .input_text, response: .target_text}' demo_cleaned.jsonl > demo_renamed.jsonl

# 追加字段char_cnt，以表示prompt和response两个字段的字符数总和
jq -c '. | .char_cnt = ((.prompt | length) + (.response | length) | floor)' demo_renamed.jsonl > demo_with_char_cnt.jsonl

# 过滤掉长度大于等于4096的数据
jq -c 'select(.char_cnt < 4096)' demo_with_char_cnt.jsonl > demo_filtered.jsonl

# 随机打散并采样10000条数据
shuf demo_filtered.jsonl | head -n 10000 > demo_sampled.jsonl

# 统计数据集的字符总数
jq -c ".char_cnt // 0" demo_sampled.jsonl | awk '{s+=$1} END {print s}'
```

## 算法 3.1  DPO算法实现示例

```python

# 【1】将三种文本转换为Token ID（数字）
prompt_ids        = tokenizer.encode( prompt_text )
win_response_ids  = tokenizer.encode( win_response_text )
lose_response_ids = tokenizer.encode( lose_response_text )
# 【2】策略模型推理并获取动作概率
policy_win_logprob, policy_lose_logprob = 
    get_logprob( policy_model, prompt_ids, win_response_ids, lose_response_ids )
# 【3】参考模型推理并获取动作概率
reference_win_logprob, reference_lose_logprob = 
    get_logprob( reference_model, prompt_ids, win_response_ids, lose_response_ids )
# 【4】计算隐式奖励值（beta是超参数β）
win_rewards  = beta * (policy_win_logprob  - reference_win_logprob)
lose_rewards = beta * (policy_lose_logprob - reference_lose_logprob)
# 【5】计算DPO loss
loss = - log_sigmoid( win_rewards - lose_rewards).mean()

# 动作概率（logprob）的详细计算过程
def get_logprob( model, prompt_ids, win_response_ids, lose_response_ids ):
    # 【拼接&推理】拼接prompt和优质回答 --> 模型推理出logits_win
    input_win  = Concat( prompt_ids, win_response_ids)
    logits_win = model( input_win ) 
    # 【拼接&推理】拼接prompt和劣质回答 --> 模型推理出logits_lose
    input_lose  = Concat( prompt_ids, lose_response_ids)
    logits_lose = model( input_lose ) 
    # 【抽取】从logits中抽取出一小部分，作为动作概率
    win_logprob = Gather(logits_win.log_softmax(),index = win_response_ids )
    lose_logprob = Gather(logits_lose.log_softmax(), index = lose_response_ids )
    return win_logprob, lose_logprob
```

## 算法 6.1  GAE算法核心实现
```python
def compute_gae(rewards, values, gamma=0.99, lambda_=0.95):
    """
    参数:
        rewards (list 或 np.ndarray): 每个时间步收集到的奖励r，形状为(T,)
        values (list 或 np.ndarray): 每个状态的价值估计V，形状为(T + 1,)
        gamma (float): 折扣因子γ
        lambda_ (float): GAE的衰减参数λ
    返回:
        np.ndarray: 优势估计A，形状为(T,)，例如对于T=5，A=[A0, A1, A2, A3, A4]
    """
    T = len(rewards)            # 时间步数T，终止时间步为t=T-1
    advantages = np.zeros(T)    # 优势估计数组，例如[A0, A1, A2, A3, A4]
    gae = 0                     # 初始化优势值为0

	# 反向从时间步t=T-1到t=0进行迭代计算，总共迭代T次
    for t in reversed(range(T)):
        # δt = rt + γ * V(st+1) - V(st)
        δt = rewards[t] + gamma * values[t + 1] - values[t]
        
        # At = δt + γ * λ * At+1
        gae = δt + gamma * lambda_ * gae
        
        advantages[t] = gae     # 追加存储计算得到的优势估计值

    return advantages
```

## 算法 7.1  基于PPO算法进行RLHF训练的代码示意
```python
# 简写 R:rewards, V:values, Adv:advantages, J:objective, P:probability
for iteration in range(num_iterations):  # 进行num_iterations个训练迭代
    #【1/2】收集样本(prompt, response_old, logP_old, Adv, V_target)
    prompt_batch, response_old_batch = [], []
    logP_old_batch, Adv_batch, V_target_batch = [], [], []
    for _ in range(num_examples):
        logP_old, response_old  = actor_model(prompt)
        V_old    = critic_model(prompt, response_old)
        R        = reward_model(prompt, response_old)[-1]
        logP_ref = ref_model(prompt, response_old)
        
        # KL距离惩罚。注意：上面的R只取了最后一个token对应的奖励分数
        KL = logP_old - logP_ref
        R_with_KL = R - scale_factor * KL

        # 通过GAE算法计算优势Adv
        Adv = GAE_Advantage(R_with_KL, V_old, gamma, λ)
        V_target = Adv + V_old

        prompt_batch        += prompt
        response_old_batch  += response_old
        logP_old_batch      += logP_old
        Adv_batch           += Adv
        V_target_batch      += V_target

    # 【2/2】多轮PPO训练，多次参数更新
    for _ in range(ppo_epochs):
        mini_batches = shuffle_split( (prompt_batch, response_old_batch, 
            logP_old_batch, Adv_batch, V_target_batch), mini_batch_size )
        
        for prompt, response_old, logP_old, Adv, V_target in mini_batches:
            logits, logP_new = actor_model(prompt, response_old)
            V_new            = critic_model(prompt, response_old)

            # 策略概率比: ratio(θ) = π_θ(a|s) / π_θ_old (a|s)
            ratios = exp(logP_new - logP_old)

            # 计算策略模型Loss
            L_clip = -mean( min( ratios * Adv,
                                clip(ratios, 1 - ε, 1 + ε) * Adv ) )
            
            S_entropy = mean( compute_entropy(logits) )  # 计算策略的熵

            Loss_V = mean((V_new - V_target) ** 2)   # 计算价值模型Loss

            Loss = L_clip + C1 * Loss_V - C2 * S_entropy # 总损失
            backward_update(Loss, L_clip, Loss_V) # 反向传播; 更新模型参数
```

# 第二部分：综合实践
## 环境搭建
训练环境的准备工作主要包含以下步骤：
（1）系统环境准备：Linux系统是模型训练的首选环境，具有良好的稳定性和兼容性。在Windows系统中，可以通过安装Git工具，并使用Git Bash命令行来模拟Linux环境执行大部分命令。如果使用macOS，可参考Linux进行相关配置，但可能会遇到例如不支持bf16（BFloat16）数据类型等问题，此时需要关闭相关选项。
（2）安装Anaconda：Anaconda是一个跨平台的包管理和环境管理工具，支持Windows、Linux和macOS操作系统。使用Anaconda可以快速搭建一致的模型训练环境。通过Anaconda的包管理功能，可以一键安装如PyTorch等常用框架和工具，大幅简化环境配置的复杂性。
（3）创建虚拟conda环境：为了避免不同项目环境之间的依赖冲突，建议通过Anaconda创建隔离的虚拟环境。隔离环境不仅能提高环境管理效率，还能确保项目间的依赖互不干扰。使用以下命令创建并进入虚拟环境：
```bash
# 创建虚拟环境，名为my_env_name，Python版本为3.10
conda create -n my_env_name python=3.10

# 进入名为my_env_name的虚拟环境
conda activate my_env_name
```

（4）下载安装LLaMA-Factory：从GitHub下载LLaMA-Factory代码仓库，并运行相关命令安装所需依赖。如果使用华为NPU，需安装如torch-npu等工具包以支持相应硬件。在执行pip install时，可以根据需求选择安装其他依赖项，例如，vllm、deepspeed、bitsandbytes、hqq、eetq、gptq、awq、galore、modelscope等，用于支持特定功能或加速。以下为具体安装命令：
```bash
# 下载LLaMA-Factory代码仓库
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git

# 进入LLaMA-Factory目录并安装所需依赖
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

（5）下载模型：根据可用GPU显存选择合适的模型大小，从ModelScope、Hugging Face等平台下载开源模型及其相关文件（具体文件信息详见数据 9.1）。例如，对于拥有6GB显存的个人电脑，可以选择阿里巴巴Qwen系列的0.5B版本（5亿参数）。下载完成后，记录模型存储路径（例如models/Qwen2-0.5B-Instruct），以便后续配置使用。
```bash
数据 9.1  模型文件及用途
model.safetensors       943M    # 主模型的权重文件，包含模型的所有参数。
config.json              686    # 模型的基本配置文件，包含配置参数和架构信息。
generation_config.json   256    # 模型推理生成时的参数配置。
tokenizer.json           7.0M   # Tokenizer的配置文件，保存词表和分词规则。
vocab.json               2.7M   # 词表文件，包含所有词汇和对应的Token ID。
merges.txt               1.8M   # Tokenizer的BPE合并规则：每次迭代合并的Token。
tokenizer_config.json    1.3K   # Tokenizer的额外配置，用于指定分词的详细选项。
```

## SFT训练
在完成相关准备工作后，可按照以下步骤进行SFT训练：
（1）配置数据集：参考LLaMA-Factory中的data/alpaca_en_demo.json数据集格式，新建自己的数据集文件（例如my_sft_data.json），并在data/dataset_info.json文件中注册该数据集（例如注册名为my_sft_data）。
（2）选择微调技术：参考2.1.7节内容，选择合适的微调技术，本文以LoRA为例。
（3）参数配置：参考LLaMA-Factory中examples/train_lora/*_lora_sft.yaml文件，新建自己的SFT配置文件（例如my_sft.yaml）。在配置文件中，根据上述数据注册名、模型路径和微调技术等，修改相关任务参数，主要参数见表 9.2。完整参数说明可通过执行命令llamafactory-cli train -h查看。
（4）启动训练：使用以下命令启动SFT训练（设置CUDA_VISIBLE_DEVICES来指定GPU设备编号；若使用华为NPU，请使用ASCEND_RT_VISIBLE_DEVICES设置）：
```bash
# 启动SFT训练
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/my_sft.yaml
```

## DPO训练
DPO训练的整体流程与SFT类似，但其使用的数据为偏好数据，每条数据需要包含优质回答和劣质回答，分别通过字段chosen和rejected表示。
创建数据集时，参考data/dpo_en_demo.json文件的字段格式，新建自己的数据集文件（例如my_dpo_data.json），并在data/dataset_info.json文件中注册该数据集（例如注册名为my_dpo_data）。
配置文件可参考LLaMA-Factory中的examples/train_lora/*_lora_dpo.yaml，并新建自己的DPO配置文件（例如my_dpo.yaml），在配置文件中，根据上述数据注册名、模型路径和微调技术等，设置相关任务参数。
使用以下命令启动DPO训练：
```bash
# 启动DPO训练
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/my_dpo.yaml
```

## 基于PPO的RLHF训练
如第7章所述，RLHF的训练包括两个阶段——奖励模型训练和基于PPO的强化学习训练。具体步骤如下：
（1）训练奖励模型：训练数据格式与DPO类似，可参考data/dpo_en_demo.json文件的字段格式封装数据集，新建自己的数据集文件（例如my_reward_data.json），并在data/dataset_info.json文件中注册数据集（例如注册名为my_reward_data）。配置文件可参考examples/train_lora/*_lora_reward.yaml，并新建自己的奖励模型训练配置文件（例如my_reward.yaml），在配置文件中，根据上述数据注册名、模型路径和微调技术等，设置相关任务参数。使用以下命令启动奖励模型训练：
```bash
### 启动奖励模型训练
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/my_reward.yaml
```

（2）基于PPO的强化学习训练：如第7.1.3节所述，PPO阶段的样本只需包含Prompt，无需包含回答（output字段不会被读取）。数据集参考data/dpo_en_demo.json格式，新建自己的数据集文件（例如my_ppo_data.json），并在data/dataset_info.json文件中注册数据集（例如注册名为my_ppo_data）。配置文件参考examples/train_lora/*_lora_ppo.yaml，并在配置中设置reward_model为已训练奖励模型的路径，并将dataset字段更新为自己的数据注册名。使用以下命令启动PPO训练：
```bash
### 启动基于PPO算法的强化学习训练
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/my_ppo.yaml
```

## 启动模型并进行推理
参考LLaMA-Factory的examples/inference/*_lora_sft.yaml文件，可以新建自定义的运行配置文件（例如my_lora_sft.yaml）。将model_name_or_path设置为原模型路径，将adapter_name_or_path设置为训练生成的adapter路径（例如LLaMA-Factory/saves/qwen/lora/sft），然后，启动训练后的模型，进行问答交互：
```bash
# 运行SFT训练后的模型
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/my_lora_sft.yaml
```

# 参考引用（Citation）

如果你在论文、著作或报告中使用了本仓库/本书中的图片等内容，请按以下格式引用（If you use any content from this project (including diagrams or concepts from the book), please cite it as follows）:

- 中文（Chinese）引用格式：
```
余昌叶. 大模型算法：强化学习、微调与对齐[M]. 北京: 电子工业出版社, 2025. https://github.com/changyeyu/LLM-RL-Visualized
```

- English：
```
Yu, Changye. Large Model Algorithms: Reinforcement Learning, Fine-Tuning, and Alignment. 
Beijing: Publishing House of Electronics Industry, 2025. https://github.com/changyeyu/LLM-RL-Visualized
```

