<p align="center">
  <img src="../src/assets/banner/幻灯片1.SVG" alt="Illustrated Large Model Algorithms: LLM, RL, VLM ..." />
</p>


<p align="center">
    <a href="./README_EN.md">
    <img
      alt="English Version"
      src="https://img.shields.io/badge/English-Version-blue?style=for-the-badge"
      width="250"
      height="50"
    />
  </a> 
  &nbsp; &nbsp;&nbsp;

  <a href="../README.md">
    <img
      alt="Chinese 中文版本"
      src="https://img.shields.io/badge/Chinese-中文版本-red?style=for-the-badge"
      width="250"
      height="50"
    />
  </a>    

</p>




---
## Description

🎉 **100+ original diagrams**visualizing core concepts of large model algorithms — from LLMs, VLMs, and training methods (RL, RLHF, GRPO, DPO, SFT, distillation) to RAG and performance optimization.

🎉 Originally based on the diagrams from the Chinese **book** [《大模型算法：强化学习、微调与对齐》](https://item.jd.com/15017130.html) (<em>Large Model Algorithms: Reinforcement Learning, Fine-Tuning, and Alignment</em>), this project has since been continuously expanded with new content and improvements.

🎉 **Continuously updated** and actively maintained — click **Star ⭐** to stay tuned!

🎉 Contributions welcome — see the <a href="#contributing" style="color:rgb(44, 46, 49); text-decoration: underline;">Contributing Guide</a> to get involved.

Click on the images to view high-resolution versions, or browse the `.svg` vector files in the repository for infinite zoom support.


## Table of Contents
- [Overall Architecture of Large Model Algorithms (Focusing on LLMs and VLMs)](#header-1)
- [【LLM basics】LLM overview](#header-2)
- [【LLM basics】LLM structure](#header-3)
- [【LLM basics】LLM generation and decoding](#header-4)
- [【LLM basics】LLM Input](#header-5)
- [【LLM basics】LLM output](#header-6)
- [【LLM basics】MLLM and VLM](#header-7)
- [【LLM basics】LLM training process](#header-8)
- [【SFT】Categories of fine-tuning techniques](#header-9)
- [【SFT】LoRA(1 of 2)](#header-10)
- [【SFT】LoRA(2 of 2)](#header-11)
- [【SFT】Prefix-Tuning](#header-12)
- [【SFT】Token ID and Token](#header-13)
- [【SFT】Loss of SFT(cross-entropy)](#header-14)
- [【SFT】Packing of multiple pieces of sample](#header-15)
- [【DPO】RLHF vs DPO](#header-16)
- [【DPO】DPO(Direct Preference Optimization)](#header-17)
- [【DPO】Overview of DPO training](#header-18)
- [【DPO】Impact of the β parameter on DPO](#header-19)
- [【DPO】Effect of implicit reward differences on the magnitude of parameter updates](#header-20)
- [【Optimization without training】Comparison of CoT and traditional Q&A](#header-21)
- [【Optimization without training】CoT、Self-consistency CoT、ToT、GoT <sup>[<a href="./references.md">87</a>]</sup>](#header-22)
- [【Optimization without training】Exhaustive Search](#header-23)
- [【Optimization without training】Greedy Search](#header-24)
- [【Optimization without training】Beam Search](#header-25)
- [【Optimization without training】Multinomial Sampling](#header-26)
- [【Optimization without training】Top-K Sampling](#header-27)
- [【Optimization without training】Top-P Sampling](#header-28)
- [【Optimization without training】RAG(Retrieval-Augmented Generation)](#header-29)
- [【Optimization without training】Function Calling](#header-30)
- [【RL basics】History of RL](#header-31)
- [【RL basics】Three major machine learning paradigms](#header-32)
- [【RL basics】Basic architecture of RL](#header-33)
- [【RL basics】Fundamental Concepts of RL](#header-34)
- [【RL basics】Markov Chain vs MDP](#header-35)
- [【RL basics】Using dynamic ε values under the ε-greedy strategy](#header-36)
- [【RL basics】Comparison of RL training paradigms](#header-37)
- [【RL basics】Classification of RL](#header-38)
- [【RL basics】Return(cumulative reward)](#header-39)
- [【RL basics】Backwards iteration and computation of return G](#header-40)
- [【RL basics】Reward, Return, and Value](#header-41)
- [【RL basics】Qπ and Vπ](#header-42)
- [【RL basics】Estimate the value through Monte Carlo(MC)](#header-43)
- [【RL basics】TD target and TD error](#header-44)
- [【RL basics】TD(0), n-step TD, and MC](#header-45)
- [【RL basics】Characteristics of MC and TD methods](#header-46)
- [【RL basics】MC, TD, DP, and exhaustive search <sup>[<a href="./references.md">32</a>]</sup>](#header-47)
- [【RL basics】DQN model with two input-output structures](#header-48)
- [【RL basics】How to use DQN](#header-49)
- [【RL basics】DQN's overestimation problem](#header-50)
- [【RL basics】Value-Based vs Policy-Based](#header-51)
- [【RL basics】Policy gradient](#header-52)
- [【RL basics】Multi-agent reinforcement learning(MARL)](#header-53)
- [【RL basics】Multi-agent DDPG <sup>[<a href="./references.md">41</a>]</sup>](#header-54)
- [【RL basics】Imitation learning(IL)](#header-55)
- [【RL basics】Behavior cloning(BC)](#header-56)
- [【RL basics】Inverse RL(IRL) and RL](#header-57)
- [【RL basics】Model-Based and Model-Free](#header-58)
- [【RL basics】Feudal RL](#header-59)
- [【RL basics】Distributional RL](#header-60)
- [【Policy Optimization & Variants】Actor-Critic](#header-61)
- [【Policy Optimization & Variants】Comparison of baseline and advantage](#header-62)
- [【Policy Optimization & Variants】GAE(Generalized Advantage Estimation)](#header-63)
- [【Policy Optimization & Variants】TRPO and its trust region](#header-64)
- [【Policy Optimization & Variants】Importance sampling](#header-65)
- [【Policy Optimization & Variants】PPO-Clip](#header-66)
- [【Policy Optimization & Variants】Policy model update process in PPO training](#header-67)
- [【Policy Optimization & Variants】PPO Pseudocode](#header-67-2)
- [【Policy Optimization & Variants】GRPO & PPO <sup>[<a href="./references.md">72</a>]</sup>](#header-68)
- [【Policy Optimization & Variants】Deterministic policy vs. Stochastic policy](#header-69)
- [【Policy Optimization & Variants】DPG](#header-70)
- [【Policy Optimization & Variants】DDPG（Deep Deterministic Policy Gradient）](#header-71)
- [【RLHF and RLAIF】RL modeling of language models](#header-72)
- [【RLHF and RLAIF】Two-stage training process of RLHF](#header-73)
- [【RLHF and RLAIF】Structure of the reward model](#header-74)
- [【RLHF and RLAIF】Input and output of the reward model](#header-75)
- [【RLHF and RLAIF】Reward deviation and loss](#header-76)
- [【RLHF and RLAIF】Training of the reward model](#header-77)
- [【RLHF and RLAIF】Relationship between the four models in PPO](#header-78)
- [【RLHF and RLAIF】The structure and init of the four models in PPO](#header-79)
- [【RLHF and RLAIF】A value model with a dual-head structure](#header-80)
- [【RLHF and RLAIF】Four models can share one base in RLHF](#header-81)
- [【RLHF and RLAIF】Inputs and Outputs of Each Model in PPO](#header-82)
- [【RLHF and RLAIF】The Process of Calculating KL in PPO](#header-83)
- [【RLHF and RLAIF】RLHF Training Based on PPO](#header-84)
- [【RLHF and RLAIF】Rejection Sampling Fine-tuning](#header-85)
- [【RLHF and RLAIF】RLAIF vs RLHF](#header-86)
- [【RLHF and RLAIF】CAI(Constitutional AI)](#header-87)
- [【RLHF and RLAIF】OpenAI RBR(Rule-Based Reward)](#header-88)
- [【Reasoning capacity optimization】Knowledge Distillation Based on CoT](#header-89)
- [【Reasoning capacity optimization】Distillation Based on DeepSeek](#header-90)
- [【Reasoning capacity optimization】ORM(Outcome Reward Model) & PRM (Process Reward Model)](#header-91)
- [【Reasoning capacity optimization】Four Key Steps of Each MCTS](#header-92)
- [【Reasoning capacity optimization】MCTS](#header-93)
- [【Reasoning capacity optimization】Search Tree Example in a Linguistic Context](#header-94)
- [【Reasoning capacity optimization】BoN(Best-of-N) Sampling](#header-95)
- [【Reasoning capacity optimization】Majority Vote](#header-96)
- [【Reasoning capacity optimization】Performance Growth of AlphaGo Zero <sup>[<a href="./references.md">179</a>]</sup>](#header-97)
- [【LLM basics extended】Performance Optimization Map for Large Models](#header-98)
- [【LLM basics extended】ALiBi positional encoding](#header-99)
- [【LLM basics extended】Traditional knowledge distillation](#header-100)
- [【LLM basics extended】Numerical representation, quantization](#header-101)
- [【LLM basics extended】Forward and backward](#header-102)
- [【LLM basics extended】Gradient Accumulation](#header-103)
- [【LLM basics extended】Gradient Checkpoint(gradient recomputation)](#header-104)
- [【LLM basics extended】Full recomputation ](#header-105)
- [【LLM basics extended】LLM Benchmark](#header-106)
- [【LLM basics extended】MHA、GQA、MQA、MLA](#header-107)
- [【LLM basics extended】RNN(Recurrent Neural Network)](#header-108)
- [【LLM basics extended】Pre-norm vs Post-norm](#header-109)
- [【LLM basics extended】BatchNorm & LayerNorm](#header-110)
- [【LLM basics extended】RMSNorm](#header-111)
- [【LLM basics extended】Prune](#header-112)
- [【LLM basics extended】Role of the temperature coefficient](#header-113)
- [【LLM basics extended】SwiGLU](#header-114)
- [【LLM basics extended】AUC、PR、F1、Precision、Recall](#header-115)
- [【LLM basics extended】RoPE positional encoding](#header-116)
- [【LLM basics extended】The effect of RoPE on each sequence position and each dim](#header-117)
- [📌 For Reference Section](#header-118)
- [📌 BibTeX Citation Format](#header-119)


### <a name="header-1"></a>Overall Architecture of Large Model Algorithms (Focusing on LLMs and VLMs)

<img src="./assets/LLM-RL-Algorithms-en.png" alt="Overall Architecture of Large Model Algorithms">


### <a name="header-2"></a>【LLM basics】LLM overview
[![【LLM basics】LLM overview](../images_english/source_svg/%E3%80%90LLM%20basics%E3%80%91LLM%20overview.svg)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/source_svg/%E3%80%90LLM%20basics%E3%80%91LLM%20overview.svg)

### <a name="header-3"></a>【LLM basics】LLM structure
[![【LLM basics】LLM structure](../images_english/png_small/%E3%80%90LLM%20basics%E3%80%91LLM%20structure.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%E3%80%91LLM%20structure.png)

### <a name="header-4"></a>【LLM basics】LLM generation and decoding
[![【LLM basics】LLM generation and decoding](../images_english/png_small/%E3%80%90LLM%20basics%E3%80%91LLM%20generation%20and%20decoding.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%E3%80%91LLM%20generation%20and%20decoding.png)

### <a name="header-5"></a>【LLM basics】LLM Input
[![【LLM basics】LLM Input](../images_english/png_small/%E3%80%90LLM%20basics%E3%80%91LLM%20Input.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%E3%80%91LLM%20Input.png)

### <a name="header-6"></a>【LLM basics】LLM output
[![【LLM basics】LLM output](../images_english/png_small/%E3%80%90LLM%20basics%E3%80%91LLM%20output.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%E3%80%91LLM%20output.png)

### <a name="header-7"></a>【LLM basics】MLLM and VLM
[![【LLM basics】MLLM and VLM](../images_english/png_small/%E3%80%90LLM%20basics%E3%80%91MLLM%20and%20VLM.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%E3%80%91MLLM%20and%20VLM.png)

### <a name="header-8"></a>【LLM basics】LLM training process
[![【LLM basics】LLM training process](../images_english/png_small/%E3%80%90LLM%20basics%E3%80%91LLM%20training%20process.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%E3%80%91LLM%20training%20process.png)

### <a name="header-9"></a>【SFT】Categories of fine-tuning techniques
[![【SFT】Categories of fine-tuning techniques](../images_english/png_small/%E3%80%90SFT%E3%80%91Categories%20of%20fine-tuning%20techniques.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90SFT%E3%80%91Categories%20of%20fine-tuning%20techniques.png)

### <a name="header-10"></a>【SFT】LoRA(1 of 2)
[![【SFT】LoRA(1 of 2)](../images_english/png_small/%E3%80%90SFT%E3%80%91LoRA%281%20of%202%29.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90SFT%E3%80%91LoRA%281%20of%202%29.png)

### <a name="header-11"></a>【SFT】LoRA(2 of 2)
[![【SFT】LoRA(2 of 2)](../images_english/png_small/%E3%80%90SFT%E3%80%91LoRA%282%20of%202%29.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90SFT%E3%80%91LoRA%282%20of%202%29.png)

### <a name="header-12"></a>【SFT】Prefix-Tuning
[![【SFT】Prefix-Tuning](../images_english/png_small/%E3%80%90SFT%E3%80%91Prefix-Tuning.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90SFT%E3%80%91Prefix-Tuning.png)

### <a name="header-13"></a>【SFT】Token ID and Token
[![【SFT】Token ID and Token](../images_english/png_small/%E3%80%90SFT%E3%80%91Token%20ID%20and%20Token.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90SFT%E3%80%91Token%20ID%20and%20Token.png)

### <a name="header-14"></a>【SFT】Loss of SFT(cross-entropy)
[![【SFT】Loss of SFT(cross-entropy)](../images_english/png_small/%E3%80%90SFT%E3%80%91Loss%20of%20SFT%28cross-entropy%29.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90SFT%E3%80%91Loss%20of%20SFT%28cross-entropy%29.png)

### <a name="header-15"></a>【SFT】Packing of multiple pieces of sample
[![【SFT】Packing of multiple pieces of sample](../images_english/png_small/%E3%80%90SFT%E3%80%91Packing%20of%20multiple%20pieces%20of%20sample.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90SFT%E3%80%91Packing%20of%20multiple%20pieces%20of%20sample.png)

### <a name="header-16"></a>【DPO】RLHF vs DPO
[![【DPO】RLHF vs DPO](../images_english/png_small/%E3%80%90DPO%E3%80%91RLHF%20vs%20DPO.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90DPO%E3%80%91RLHF%20vs%20DPO.png)

### <a name="header-17"></a>【DPO】DPO(Direct Preference Optimization)
[![【DPO】DPO(DirectPreferenceOptimization)](../images_english/png_small/%E3%80%90DPO%E3%80%91DPO%28DirectPreferenceOptimization%29.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90DPO%E3%80%91DPO%28DirectPreferenceOptimization%29.png)

### <a name="header-18"></a>【DPO】Overview of DPO training
[![【DPO】Overview of DPO training](../images_english/png_small/%E3%80%90DPO%E3%80%91Overview%20of%20DPO%20training.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90DPO%E3%80%91Overview%20of%20DPO%20training.png)

### <a name="header-19"></a>【DPO】Impact of the β parameter on DPO
[![【DPO】Impact of the β parameter on DPO](../images_english/png_small/%E3%80%90DPO%E3%80%91Impact%20of%20the%20%CE%B2%20parameter%20on%20DPO.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90DPO%E3%80%91Impact%20of%20the%20%CE%B2%20parameter%20on%20DPO.png)

### <a name="header-20"></a>【DPO】Effect of implicit reward differences on the magnitude of parameter updates
[![【DPO】Effect of implicit reward differences on the magnitude of parameter updates](../images_english/png_small/%E3%80%90DPO%E3%80%91Effect%20of%20implicit%20reward%20differences%20on%20the%20magnitude%20of%20parameter%20updates.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90DPO%E3%80%91Effect%20of%20implicit%20reward%20differences%20on%20the%20magnitude%20of%20parameter%20updates.png)

### <a name="header-21"></a>【Optimization without training】Comparison of CoT and traditional Q&A
[![【Optimization without training】Comparison of CoT and traditional Q&A](../images_english/png_small/%E3%80%90Optimization%20without%20training%E3%80%91Comparison%20of%20CoT%20and%20traditional%20Q%26A.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Optimization%20without%20training%E3%80%91Comparison%20of%20CoT%20and%20traditional%20Q%26A.png)

### <a name="header-22"></a>【Optimization without training】CoT、Self-consistency CoT、ToT、GoT <sup>[<a href="./references.md">87</a>]</sup>
[![【Optimization without training】CoT、Self-consistencyCoT、ToT、GoT](../images_english/png_small/%E3%80%90Optimization%20without%20training%E3%80%91CoT%E3%80%81Self-consistencyCoT%E3%80%81ToT%E3%80%81GoT.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Optimization%20without%20training%E3%80%91CoT%E3%80%81Self-consistencyCoT%E3%80%81ToT%E3%80%81GoT.png)

### <a name="header-23"></a>【Optimization without training】Exhaustive Search
[![【Optimization without training】Exhaustive Search](../images_english/png_small/%E3%80%90Optimization%20without%20training%E3%80%91Exhaustive%20Search.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Optimization%20without%20training%E3%80%91Exhaustive%20Search.png)

### <a name="header-24"></a>【Optimization without training】Greedy Search
[![【Optimization without training】Greedy Search](../images_english/png_small/%E3%80%90Optimization%20without%20training%E3%80%91Greedy%20Search.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Optimization%20without%20training%E3%80%91Greedy%20Search.png)

### <a name="header-25"></a>【Optimization without training】Beam Search
[![【Optimization without training】Beam Search](../images_english/png_small/%E3%80%90Optimization%20without%20training%E3%80%91Beam%20Search.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Optimization%20without%20training%E3%80%91Beam%20Search.png)

### <a name="header-26"></a>【Optimization without training】Multinomial Sampling
[![【Optimization without training】Multinomial Sampling](../images_english/png_small/%E3%80%90Optimization%20without%20training%E3%80%91Multinomial%20Sampling.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Optimization%20without%20training%E3%80%91Multinomial%20Sampling.png)

### <a name="header-27"></a>【Optimization without training】Top-K Sampling
[![【Optimization without training】Top-K Sampling](../images_english/png_small/%E3%80%90Optimization%20without%20training%E3%80%91Top-K%20Sampling.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Optimization%20without%20training%E3%80%91Top-K%20Sampling.png)

### <a name="header-28"></a>【Optimization without training】Top-P Sampling
[![【Optimization without training】Top-P Sampling](../images_english/png_small/%E3%80%90Optimization%20without%20training%E3%80%91Top-P%20Sampling.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Optimization%20without%20training%E3%80%91Top-P%20Sampling.png)

### <a name="header-29"></a>【Optimization without training】RAG(Retrieval-Augmented Generation)
[![【Optimization without training】RAG(Retrieval-Augmented Generation)](../images_english/png_small/%E3%80%90Optimization%20without%20training%E3%80%91RAG%28Retrieval-Augmented%20Generation%29.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Optimization%20without%20training%E3%80%91RAG%28Retrieval-Augmented%20Generation%29.png)

### <a name="header-30"></a>【Optimization without training】Function Calling
[![【Optimization without training】Function Calling](../images_english/png_small/%E3%80%90Optimization%20without%20training%E3%80%91Function%20Calling.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Optimization%20without%20training%E3%80%91Function%20Calling.png)

### <a name="header-31"></a>【RL basics】History of RL
[![【RL basics】History of RL](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91History%20of%20RL.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91History%20of%20RL.png)

### <a name="header-32"></a>【RL basics】Three major machine learning paradigms
[![【RL basics】Three major machine learning paradigms](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Three%20major%20machine%20learning%20paradigms.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Three%20major%20machine%20learning%20paradigms.png)

### <a name="header-33"></a>【RL basics】Basic architecture of RL
[![【RL basics】Basic architecture of RL](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Basic%20architecture%20of%20RL.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Basic%20architecture%20of%20RL.png)

### <a name="header-34"></a>【RL basics】Fundamental Concepts of RL
[![【RL basics】Fundamental Concepts of RL](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Fundamental%20Concepts%20of%20RL.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Fundamental%20Concepts%20of%20RL.png)

### <a name="header-35"></a>【RL basics】Markov Chain vs MDP
[![【RL basics】Markov Chain vs MDP](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Markov%20Chain%20vs%20MDP.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Markov%20Chain%20vs%20MDP.png)

### <a name="header-36"></a>【RL basics】Using dynamic ε values under the ε-greedy strategy
[![【RL basics】Using dynamic ε values under the ε-greedy strategy](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Using%20dynamic%20%CE%B5%20values%20under%20the%20%CE%B5-greedy%20strategy.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Using%20dynamic%20%CE%B5%20values%20under%20the%20%CE%B5-greedy%20strategy.png)

### <a name="header-37"></a>【RL basics】Comparison of RL training paradigms
- On-policy，Off-policy，Offline RL
[![【RL basics】Comparison of RL training paradigms](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Comparison%20of%20RL%20training%20paradigms.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Comparison%20of%20RL%20training%20paradigms.png)

### <a name="header-38"></a>【RL basics】Classification of RL
[![【RL basics】Classification of RL](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Classification%20of%20RL.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Classification%20of%20RL.png)

### <a name="header-39"></a>【RL basics】Return(cumulative reward)
[![【RL basics】Return(cumulative reward)](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Return%28cumulative%20reward%29.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Return%28cumulative%20reward%29.png)

### <a name="header-40"></a>【RL basics】Backwards iteration and computation of return G
[![【RL basics】Backwards iteration and computation of return G](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Backwards%20iteration%20and%20computation%20of%20return%20G.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Backwards%20iteration%20and%20computation%20of%20return%20G.png)

### <a name="header-41"></a>【RL basics】Reward, Return, and Value
[![【RL basics】Reward, Return, and Value](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Reward%2C%20Return%2C%20and%20Value.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Reward%2C%20Return%2C%20and%20Value.png)

### <a name="header-42"></a>【RL basics】Qπ and Vπ
[![【RL basics】Qπ and Vπ](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Q%CF%80%20and%20V%CF%80.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Q%CF%80%20and%20V%CF%80.png)

### <a name="header-43"></a>【RL basics】Estimate the value through Monte Carlo(MC)
[![【RL basics】Estimate the value through Monte Carlo(MC)](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Estimate%20the%20value%20through%20Monte%20Carlo%28MC%29.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Estimate%20the%20value%20through%20Monte%20Carlo%28MC%29.png)

### <a name="header-44"></a>【RL basics】TD target and TD error
[![【RL basics】TD target and TD error](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91TD%20target%20and%20TD%20error.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91TD%20target%20and%20TD%20error.png)

### <a name="header-45"></a>【RL basics】TD(0), n-step TD, and MC
[![【RL basics】TD(0), n-step TD, and MC](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91TD%280%29%2C%20n-step%20TD%2C%20and%20MC.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91TD%280%29%2C%20n-step%20TD%2C%20and%20MC.png)

### <a name="header-46"></a>【RL basics】Characteristics of MC and TD methods
[![【RL basics】Characteristics of MC and TD methods](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Characteristics%20of%20MC%20and%20TD%20methods.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Characteristics%20of%20MC%20and%20TD%20methods.png)

### <a name="header-47"></a>【RL basics】MC, TD, DP, and exhaustive search <sup>[<a href="./references.md">32</a>]</sup>
[![【RL basics】MC, TD, DP, and exhaustive search](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91MC%2C%20TD%2C%20DP%2C%20and%20exhaustive%20search.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91MC%2C%20TD%2C%20DP%2C%20and%20exhaustive%20search.png)

### <a name="header-48"></a>【RL basics】DQN model with two input-output structures
[![【RL basics】DQN model with two input-output structures](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91DQN%20model%20with%20two%20input-output%20structures.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91DQN%20model%20with%20two%20input-output%20structures.png)

### <a name="header-49"></a>【RL basics】How to use DQN
[![【RL basics】How to use DQN](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91How%20to%20use%20DQN.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91How%20to%20use%20DQN.png)

### <a name="header-50"></a>【RL basics】DQN's overestimation problem
[![【RL basics】DQN's overestimation problem](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91DQN%27s%20overestimation%20problem.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91DQN%27s%20overestimation%20problem.png)

### <a name="header-51"></a>【RL basics】Value-Based vs Policy-Based
[![【RL basics】Value-Based vs Policy-Based](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Value-Based%20vs%20Policy-Based.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Value-Based%20vs%20Policy-Based.png)

### <a name="header-52"></a>【RL basics】Policy gradient
[![【RL basics】Policy gradient](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Policy%20gradient.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Policy%20gradient.png)

### <a name="header-53"></a>【RL basics】Multi-agent reinforcement learning(MARL)
[![【RL basics】Multi-agent reinforcement learning(MARL)](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Multi-agent%20reinforcement%20learning%28MARL%29.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Multi-agent%20reinforcement%20learning%28MARL%29.png)

### <a name="header-54"></a>【RL basics】Multi-agent DDPG <sup>[<a href="./references.md">41</a>]</sup>
[![【RL basics】Multi-agent DDPG](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Multi-agent%20DDPG.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Multi-agent%20DDPG.png)

### <a name="header-55"></a>【RL basics】Imitation learning(IL)
[![【RL basics】Imitation learning(IL)](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Imitation%20learning%28IL%29.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Imitation%20learning%28IL%29.png)

### <a name="header-56"></a>【RL basics】Behavior cloning(BC)
[![【RL basics】Behavior cloning(BC)](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Behavior%20cloning%28BC%29.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Behavior%20cloning%28BC%29.png)

### <a name="header-57"></a>【RL basics】Inverse RL(IRL) and RL
[![【RL basics】Inverse RL(IRL) and RL](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Inverse%20RL%28IRL%29%20and%20RL.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Inverse%20RL%28IRL%29%20and%20RL.png)

### <a name="header-58"></a>【RL basics】Model-Based and Model-Free
[![【RL basics】Model-Based and Model-Free](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Model-Based%20and%20Model-Free.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Model-Based%20and%20Model-Free.png)

### <a name="header-59"></a>【RL basics】Feudal RL
[![【RL basics】Feudal RL](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Feudal%20RL.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Feudal%20RL.png)

### <a name="header-60"></a>【RL basics】Distributional RL
[![【RL basics】Distributional RL](../images_english/png_small/%E3%80%90RL%20basics%E3%80%91Distributional%20RL.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RL%20basics%E3%80%91Distributional%20RL.png)

### <a name="header-61"></a>【Policy Optimization & Variants】Actor-Critic
[![【Policy Optimization & Variants】Actor-Critic](../images_english/png_small/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91Actor-Critic.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91Actor-Critic.png)

### <a name="header-62"></a>【Policy Optimization & Variants】Comparison of baseline and advantage
[![【Policy Optimization & Variants】Comparison of baseline and advantage](../images_english/png_small/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91Comparison%20of%20baseline%20and%20advantage.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91Comparison%20of%20baseline%20and%20advantage.png)

### <a name="header-63"></a>【Policy Optimization & Variants】GAE(Generalized Advantage Estimation)
[![【Policy Optimization & Variants】GAE](../images_english/png_small/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91GAE.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91GAE.png)

### <a name="header-64"></a>【Policy Optimization & Variants】TRPO and its trust region
[![【Policy Optimization & Variants】TRPO and its trust region](../images_english/png_small/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91TRPO%20and%20its%20trust%20region.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91TRPO%20and%20its%20trust%20region.png)

### <a name="header-65"></a>【Policy Optimization & Variants】Importance sampling
[![【Policy Optimization & Variants】Importance sampling](../images_english/png_small/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91Importance%20sampling.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91Importance%20sampling.png)

### <a name="header-66"></a>【Policy Optimization & Variants】PPO-Clip
[![【Policy Optimization & Variants】PPO-Clip](../images_english/png_small/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91PPO-Clip.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91PPO-Clip.png)

### <a name="header-67"></a>【Policy Optimization & Variants】Policy model update process in PPO training
[![【Policy Optimization & Variants】Policy model update process in PPO training](../images_english/png_small/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91Policy%20model%20update%20process%20in%20PPO%20training.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91Policy%20model%20update%20process%20in%20PPO%20training.png)

### <a name="header-67-2"></a>【Policy Optimization & Variants】PPO Pseudocode
[![【Policy Optimization & Variants】Policy model update process in PPO training](../images_english/png_small/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91Policy%20model%20update%20process%20in%20PPO%20training.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91Policy%20model%20update%20process%20in%20PPO%20training.png)

```python
# Abbreviations: R = rewards, V = values, Adv = advantages, J = objective, P = probability
for iteration in range(num_iterations):  # Perform num_iterations training iterations
    # [1/2] Collect samples (prompt, response_old, logP_old, Adv, V_target)
    prompt_batch, response_old_batch = [], []
    logP_old_batch, Adv_batch, V_target_batch = [], [], []
    for _ in range(num_examples):
        logP_old, response_old  = actor_model(prompt)
        V_old    = critic_model(prompt, response_old)
        R        = reward_model(prompt, response_old)[-1]
        logP_ref = ref_model(prompt, response_old)
        
        # KL penalty. Note: R here is only the reward for the final token
        KL = logP_old - logP_ref
        R_with_KL = R - scale_factor * KL

        # Compute advantage Adv via GAE
        Adv = GAE_Advantage(R_with_KL, V_old, gamma, λ)
        V_target = Adv + V_old

        prompt_batch        += prompt
        response_old_batch  += response_old
        logP_old_batch      += logP_old
        Adv_batch           += Adv
        V_target_batch      += V_target

    # [2/2] PPO training loop: multiple parameter updates
    for _ in range(ppo_epochs):
        mini_batches = shuffle_split(
            (prompt_batch, response_old_batch, logP_old_batch, Adv_batch, V_target_batch),
            mini_batch_size
        )
        
        for prompt, response_old, logP_old, Adv, V_target in mini_batches:
            logits, logP_new = actor_model(prompt, response_old)
            V_new            = critic_model(prompt, response_old)

            # Probability ratio: ratio(θ) = π_θ(a|s) / π_{θ_old}(a|s)
            ratios = exp(logP_new - logP_old)

            # Compute clipped policy loss
            L_clip = -mean(
                min(ratios * Adv,
                    clip(ratios, 1 - ε, 1 + ε) * Adv)
            )
            
            S_entropy = mean(compute_entropy(logits))  # Compute policy entropy

            Loss_V = mean((V_new - V_target) ** 2)     # Compute value function loss

            # Total loss
            Loss = L_clip + C1 * Loss_V - C2 * S_entropy

            backward_update(Loss, L_clip, Loss_V)      # Backpropagate and update parameters
```

### <a name="header-68"></a>【Policy Optimization & Variants】GRPO & PPO <sup>[<a href="./references.md">72</a>]</sup>
[![【Policy Optimization & Variants】GRPO & PPO](../images_english/png_small/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91GRPO%20%26%20PPO.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91GRPO%20%26%20PPO.png)

### <a name="header-69"></a>【Policy Optimization & Variants】Deterministic policy vs. Stochastic policy
[![【Policy Optimization & Variants】Deterministic policy vs. Stochastic policy](../images_english/png_small/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91Deterministic%20policy%20vs.%20Stochastic%20policy.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91Deterministic%20policy%20vs.%20Stochastic%20policy.png)

### <a name="header-70"></a>【Policy Optimization & Variants】DPG
[![【Policy Optimization & Variants】DPG](../images_english/png_small/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91DPG.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91DPG.png)

### <a name="header-71"></a>【Policy Optimization & Variants】DDPG（Deep Deterministic Policy Gradient）
[![【Policy Optimization & Variants】DDPG](../images_english/png_small/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91DDPG.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Policy%20Optimization%20%26%20Variants%E3%80%91DDPG.png)

### <a name="header-72"></a>【RLHF and RLAIF】RL modeling of language models
[![【RLHF and RLAIF】RL modeling of language models](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91RL%20modeling%20of%20language%20models.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91RL%20modeling%20of%20language%20models.png)

### <a name="header-73"></a>【RLHF and RLAIF】Two-stage training process of RLHF
[![【RLHF and RLAIF】Two-stage training process of RLHF](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Two-stage%20training%20process%20of%20RLHF.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Two-stage%20training%20process%20of%20RLHF.png)

### <a name="header-74"></a>【RLHF and RLAIF】Structure of the reward model
[![【RLHF and RLAIF】Structure of the reward model](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Structure%20of%20the%20reward%20model.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Structure%20of%20the%20reward%20model.png)

### <a name="header-75"></a>【RLHF and RLAIF】Input and output of the reward model
[![【RLHF and RLAIF】Input and output of the reward model](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Input%20and%20output%20of%20the%20reward%20model.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Input%20and%20output%20of%20the%20reward%20model.png)

### <a name="header-76"></a>【RLHF and RLAIF】Reward deviation and loss
[![【RLHF and RLAIF】Reward deviation and loss](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Reward%20deviation%20and%20loss.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Reward%20deviation%20and%20loss.png)

### <a name="header-77"></a>【RLHF and RLAIF】Training of the reward model
[![【RLHF and RLAIF】Training of the reward model](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Training%20of%20the%20reward%20model.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Training%20of%20the%20reward%20model.png)

### <a name="header-78"></a>【RLHF and RLAIF】Relationship between the four models in PPO
[![【RLHF and RLAIF】Relationship between the four models in PPO](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Relationship%20between%20the%20four%20models%20in%20PPO.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Relationship%20between%20the%20four%20models%20in%20PPO.png)

### <a name="header-79"></a>【RLHF and RLAIF】The structure and init of the four models in PPO
[![【RLHF and RLAIF】The structure and init of the four models in PPO](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91The%20structure%20and%20init%20of%20the%20four%20models%20in%20PPO.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91The%20structure%20and%20init%20of%20the%20four%20models%20in%20PPO.png)

### <a name="header-80"></a>【RLHF and RLAIF】A value model with a dual-head structure
[![【RLHF and RLAIF】A value model with a dual-head structure](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91A%20value%20model%20with%20a%20dual-head%20structure.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91A%20value%20model%20with%20a%20dual-head%20structure.png)

### <a name="header-81"></a>【RLHF and RLAIF】Four models can share one base in RLHF
[![【RLHF and RLAIF】Four models can share one base in RLHF](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Four%20models%20can%20share%20one%20base%20in%20RLHF.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Four%20models%20can%20share%20one%20base%20in%20RLHF.png)

### <a name="header-82"></a>【RLHF and RLAIF】Inputs and Outputs of Each Model in PPO
[![【RLHF and RLAIF】Inputs and Outputs of Each Model in PPO](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Inputs%20and%20Outputs%20of%20Each%20Model%20in%20PPO.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Inputs%20and%20Outputs%20of%20Each%20Model%20in%20PPO.png)

### <a name="header-83"></a>【RLHF and RLAIF】The Process of Calculating KL in PPO
[![【RLHF and RLAIF】The Process of Calculating KL in PPO](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91The%20Process%20of%20Calculating%20KL%20in%20PPO.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91The%20Process%20of%20Calculating%20KL%20in%20PPO.png)

### <a name="header-84"></a>【RLHF and RLAIF】RLHF Training Based on PPO
[![【RLHF and RLAIF】RLHF Training Based on PPO](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91RLHF%20Training%20Based%20on%20PPO.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91RLHF%20Training%20Based%20on%20PPO.png)

### <a name="header-85"></a>【RLHF and RLAIF】Rejection Sampling Fine-tuning
[![【RLHF and RLAIF】Rejection Sampling Fine-tuning](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Rejection%20Sampling%20Fine-tuning.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91Rejection%20Sampling%20Fine-tuning.png)

### <a name="header-86"></a>【RLHF and RLAIF】RLAIF vs RLHF
[![【RLHF and RLAIF】RLAIF vs RLHF](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91RLAIF%20vs%20RLHF.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91RLAIF%20vs%20RLHF.png)

### <a name="header-87"></a>【RLHF and RLAIF】CAI(Constitutional AI)
[![【RLHF and RLAIF】CAI(Constitutional AI)](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91CAI%28Constitutional%20AI%29.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91CAI%28Constitutional%20AI%29.png)

### <a name="header-88"></a>【RLHF and RLAIF】OpenAI RBR(Rule-Based Reward)
[![【RLHF and RLAIF】OpenAI RBR(Rule-Based Reward)](../images_english/png_small/%E3%80%90RLHF%20and%20RLAIF%E3%80%91OpenAI%20RBR%28Rule-Based%20Reward%29.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90RLHF%20and%20RLAIF%E3%80%91OpenAI%20RBR%28Rule-Based%20Reward%29.png)

### <a name="header-89"></a>【Reasoning capacity optimization】Knowledge Distillation Based on CoT
[![【Reasoning capacity optimization】Knowledge Distillation Based on CoT](../images_english/png_small/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91Knowledge%20Distillation%20Based%20on%20CoT.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91Knowledge%20Distillation%20Based%20on%20CoT.png)

### <a name="header-90"></a>【Reasoning capacity optimization】Distillation Based on DeepSeek
[![【Reasoning capacity optimization】Distillation Based on DeepSeek](../images_english/png_small/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91Distillation%20Based%20on%20DeepSeek.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91Distillation%20Based%20on%20DeepSeek.png)

### <a name="header-91"></a>【Reasoning capacity optimization】ORM(Outcome Reward Model) & PRM (Process Reward Model)
[![【Reasoning capacity optimization】ORM & PRM](../images_english/png_small/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91ORM%C2%A0%26%C2%A0PRM.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91ORM%C2%A0%26%C2%A0PRM.png)

### <a name="header-92"></a>【Reasoning capacity optimization】Four Key Steps of Each MCTS
[![【Reasoning capacity optimization】Four Key Steps of Each MCTS](../images_english/png_small/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91Four%20Key%20Steps%20of%20Each%20MCTS.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91Four%20Key%20Steps%20of%20Each%20MCTS.png)

### <a name="header-93"></a>【Reasoning capacity optimization】MCTS
[![【Reasoning capacity optimization】MCTS](../images_english/png_small/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91MCTS.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91MCTS.png)

### <a name="header-94"></a>【Reasoning capacity optimization】Search Tree Example in a Linguistic Context
[![【Reasoning capacity optimization】Search Tree Example in a Linguistic Context](../images_english/png_small/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91Search%20Tree%20Example%20in%20a%20Linguistic%20Context.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91Search%20Tree%20Example%20in%20a%20Linguistic%20Context.png)

### <a name="header-95"></a>【Reasoning capacity optimization】BoN(Best-of-N) Sampling
[![【Reasoning capacity optimization】BoN(Best-of-N) Sampling](../images_english/png_small/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91BoN%28Best-of-N%29%20Sampling.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91BoN%28Best-of-N%29%20Sampling.png)

### <a name="header-96"></a>【Reasoning capacity optimization】Majority Vote
[![【Reasoning capacity optimization】Majority Vote](../images_english/png_small/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91Majority%20Vote.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91Majority%20Vote.png)

### <a name="header-97"></a>【Reasoning capacity optimization】Performance Growth of AlphaGo Zero <sup>[<a href="./references.md">179</a>]</sup>
[![【Reasoning capacity optimization】Performance Growth of AlphaGo Zero](../images_english/png_small/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91Performance%20Growth%20of%20AlphaGo%20Zero.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90Reasoning%20capacity%20optimization%E3%80%91Performance%20Growth%20of%20AlphaGo%20Zero.png)

### <a name="header-98"></a>【LLM basics extended】Performance Optimization Map for Large Models
[![【LLM basics extended】Performance Optimization Map for Large Models](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91Performance%20Optimization%20Map%20for%20Large%20Models.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91Performance%20Optimization%20Map%20for%20Large%20Models.png)

### <a name="header-99"></a>【LLM basics extended】ALiBi positional encoding
[![【LLM basics extended】ALiBi positional encoding](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91ALiBi%20positional%20encoding.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91ALiBi%20positional%20encoding.png)

### <a name="header-100"></a>【LLM basics extended】Traditional knowledge distillation
[![【LLM basics extended】Traditional knowledge distillation](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91Traditional%20knowledge%20distillation.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91Traditional%20knowledge%20distillation.png)

### <a name="header-101"></a>【LLM basics extended】Numerical representation, quantization
[![【LLM basics extended】Numerical representation, quantization](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91Numerical%20representation%2C%20quantization.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91Numerical%20representation%2C%20quantization.png)

### <a name="header-102"></a>【LLM basics extended】Forward and backward
[![【LLM basics extended】Forward and backward](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91Forward%20and%20backward.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91Forward%20and%20backward.png)

### <a name="header-103"></a>【LLM basics extended】Gradient Accumulation
[![【LLM basics extended】Gradient Accumulation](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91Gradient%20Accumulation.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91Gradient%20Accumulation.png)

### <a name="header-104"></a>【LLM basics extended】Gradient Checkpoint(gradient recomputation)
[![【LLM basics extended】Gradient Checkpoint(gradient recomputation)](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91Gradient%20Checkpoint%28gradient%20recomputation%29.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91Gradient%20Checkpoint%28gradient%20recomputation%29.png)

### <a name="header-105"></a>【LLM basics extended】Full recomputation 
[![【LLM basics extended】Full recomputation ](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91Full%20recomputation%20.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91Full%20recomputation%20.png)

### <a name="header-106"></a>【LLM basics extended】LLM Benchmark
[![【LLM basics extended】LLM Benchmark](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91LLM%20Benchmark.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91LLM%20Benchmark.png)

### <a name="header-107"></a>【LLM basics extended】MHA、GQA、MQA、MLA
[![【LLM basics extended】MHA、GQA、MQA、MLA](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91MHA%E3%80%81GQA%E3%80%81MQA%E3%80%81MLA.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91MHA%E3%80%81GQA%E3%80%81MQA%E3%80%81MLA.png)

### <a name="header-108"></a>【LLM basics extended】RNN(Recurrent Neural Network)
[![【LLM basics extended】RNN](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91RNN.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91RNN.png)

### <a name="header-109"></a>【LLM basics extended】Pre-norm vs Post-norm
[![【LLM basics extended】Pre-norm vs Post-norm](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91Pre-norm%20vs%20Post-norm.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91Pre-norm%20vs%20Post-norm.png)

### <a name="header-110"></a>【LLM basics extended】BatchNorm & LayerNorm
[![【LLM basics extended】BatchNorm & LayerNorm](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91BatchNorm%C2%A0%26%C2%A0LayerNorm.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91BatchNorm%C2%A0%26%C2%A0LayerNorm.png)

### <a name="header-111"></a>【LLM basics extended】RMSNorm
[![【LLM basics extended】RMSNorm](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91RMSNorm.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91RMSNorm.png)

### <a name="header-112"></a>【LLM basics extended】Prune
[![【LLM basics extended】Prune](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91Prune.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91Prune.png)

### <a name="header-113"></a>【LLM basics extended】Role of the temperature coefficient
[![【LLM basics extended】Role of the temperature coefficient](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91Role%20of%20the%20temperature%20coefficient.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91Role%20of%20the%20temperature%20coefficient.png)

### <a name="header-114"></a>【LLM basics extended】SwiGLU
[![【LLM basics extended】SwiGLU](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91SwiGLU.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91SwiGLU.png)

### <a name="header-115"></a>【LLM basics extended】AUC、PR、F1、Precision、Recall
[![【LLM basics extended】AUC、PR、F1、Precision、Recall](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91AUC%E3%80%81PR%E3%80%81F1%E3%80%81Precision%E3%80%81Recall.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91AUC%E3%80%81PR%E3%80%81F1%E3%80%81Precision%E3%80%81Recall.png)

### <a name="header-116"></a>【LLM basics extended】RoPE positional encoding
[![【LLM basics extended】RoPE positional encoding](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91RoPE%20positional%20encoding.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91RoPE%20positional%20encoding.png)

### <a name="header-117"></a>【LLM basics extended】The effect of RoPE on each sequence position and each dim
- For details on the principles of RoPE, the base and θ values, and how they work, see: [RoPE-theta-base.xlsx](./RoPE-theta-base.xlsx) 


[![【LLM basics extended】The effect of RoPE on each sequence position and each dim](../images_english/png_small/%E3%80%90LLM%20basics%20extended%E3%80%91The%20effect%20of%20RoPE%20on%20each%20sequence%20position%20and%20each%20dim.png)](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_english/png_big/%E3%80%90LLM%20basics%20extended%E3%80%91The%20effect%20of%20RoPE%20on%20each%20sequence%20position%20and%20each%20dim.png)



<br>

---


## Contributing
- **Contributions are welcome!** Whether it's new diagrams, documentation, error corrections, or other improvements. You're welcome to include your name or nickname in the diagrams. Your GitHub account will also be listed in the **Contributors** to help more people discover you and your work. Example diagram template: [images-template.pptx](./assets/images-template.pptx)


-  **How to contribute:**  
  (1) Fork: Click the "Fork" button to create a copy of the repo under your GitHub account →  
  (2) Clone: Clone the forked repo to your local environment →  
  (3) Create a new local branch →  
  (4) Make changes and commit →  
  (5) Push changes to your remote repo →  
  (6) Submit a PR: On GitHub, go to your forked repo and click "Compare & pull request" to submit a PR. The maintainer will review and merge it into the main repository.

-  **Suggested color** scheme for diagram design:  
  <span style="display:inline-block;width:12px;height:12px;background-color:#71CCF5;border-radius:2px;margin-right:8px;"></span>Light Blue (`#71CCF5`) ;  
  <span style="display:inline-block;width:12px;height:12px;background-color:#FFE699;border-radius:2px;margin-right:8px;"></span>Light Yellow (`#FFE699`) ;  
  <span style="display:inline-block;width:12px;height:12px;background-color:#C0BFDE;border-radius:2px;margin-right:8px;"></span>Blue-Purple (`#C0BFDE`) ;  
  <span style="display:inline-block;width:12px;height:12px;background-color:#F0ADB7;border-radius:2px;margin-right:8px;"></span>Pink (`#F0ADB7`)


## Terms of Use
All images in this repository are licensed under [LICENSE](../LICENSE). You are free to use, modify, and remix the materials under the following terms:  
- **Sharing** — You may copy and redistribute the material in any format.  
- **Adapting** — You may remix, transform, and build upon the material.

You must also comply with the following terms:  
- **For Web Use** — If using the materials in blog posts or online content, please retain the original author information embedded in the images.  
- **For Papers, Books, and Publications** — If using the materials in formal publications, please cite the source using the format below. In such cases, the embedded author info may be removed from the image.  
- **Non-commercial Use Only** — These materials may not be used for any direct commercial purposes.



## Citation

If you use any content from this project (including diagrams or concepts from the book), please cite it as follows:

#### <a name="header-118"></a>📌 For Reference Section
```
Yu, Changye. Large Model Algorithms: Reinforcement Learning, Fine-Tuning, and Alignment. 
Beijing: Publishing House of Electronics Industry, 2025. https://github.com/changyeyu/LLM-RL-Visualized
```

#### <a name="header-119"></a>📌 BibTeX Citation Format
```bibtex
@book{yu2025largemodel_en,
  title     = {Large Model Algorithms: Reinforcement Learning, Fine-Tuning, and Alignment},
  author    = {Yu, Changye},
  publisher = {Publishing House of Electronics Industry},
  year      = {2025},
  address   = {Beijing},
  isbn      = {9787121500725},
  url       = {https://github.com/changyeyu/LLM-RL-Visualized},
  language  = {en}
}
```

---

<div align="center"> Continuously <strong> updating</strong>...   Click ⭐<strong>Star</strong> at the top-right to follow! </div> 