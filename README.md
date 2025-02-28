# Harnessing the Inference Economy: A Survey of Efficient Reasoning for Large Language Models


- [Harnessing the Inference Economy: A Survey of Efficient Reasoning for Large Language Models](#harnessing-the-inference-economy-a-survey-of-efficient-reasoning-for-large-language-models)
    - [Introduction](#introduction)
    - [▶️ 1   Refineing Post-training Methods for Efficient Reasoning](#️-1-refineing-post-training-methods-for-efficient-reasoning)
      - [1.1     Fault Identification and Analysis](#11---fault-identification-and-analysis)
        - [1.1.1  For SFT stage](#111for-sft-stage)
        - [1.1.2  For RL stage](#112for-rl-stage)
      - [1.2     (Mitigating) Solutions](#12---mitigating-solutions)
        - [1.2.1  For SFT stage](#121for-sft-stage)
        - [1.2.2  For RL stage](#122for-rl-stage)
    - [▶️ 2    Refineing Test-time Methods for Efficient Reasoning](#️-2--refineing-test-time-methods-for-efficient-reasoning)
      - [2.1     Fault Identification and Analysis](#21---fault-identification-and-analysis)
        - [The cause of computation waste](#the-cause-of-computation-waste)
      - [2.2    (Mitigating) Solutions](#22--mitigating-solutions)
        - [2.2.1    Budget Prediction \& Allocation before Decoding](#221--budget-prediction--allocation-before-decoding)
        - [2.2.2     Adaptive Budget Allocation During Decoding](#222---adaptive-budget-allocation-during-decoding)
    - [▶️ 3     Post-training Calibrated with Inference Algorithm](#️-3---post-training-calibrated-with-inference-algorithm)


### Introduction
Investing in improving inference-time computation might prove more beneficial than increasing model pre-training compute.


---

### ▶️ 1&nbsp;&nbsp; Refineing Post-training Methods for Efficient Reasoning


目前，研究普遍认为，在Reinforcement Learning from Human Feedback, RLHF阶段，可能会出现reward hacking现象，从而导致大型语言模型LLMs的输出结果存在潜在问题。

因此一些工作尝试对这些现象进行分析。

- fake alignment / reward hacking
  - **length bias**: 算法所引发的**length-bias**，会导致模型输出文本的长度不断增加，而其中的有用信息含量却相对较低。
  - **Reasoning is only skin-deep/ shallow thinking** : good reasoning style, bad performance, Fake reasoning abilities alignment

(self-refinement相关工作也发现了这种情况，有refine行为，但是没有refine的实质)

最近，R1等以长推理（long reasoning）为重点的模型，推理能力很强，但是模型输出文本长度冗长。
无论是针对简单问题还是复杂问题，模型的回复相比于通用模型，均呈现出显著的冗长性。

因此，先前工作提出使用多种方法来尝试解决。

#### 1.1&nbsp;&nbsp;   Fault Identification and Analysis


##### 1.1.1&nbsp;&nbsp;For SFT stage

##### 1.1.2&nbsp;&nbsp;For RL stage


- [A Long Way to Go: Investigating Length Correlations in RLHF](https://arxiv.org/abs/2310.03716v2)
  - 发现PPO中length bias十分严重，现有RM只能识别浅层人类偏好，如长度
  - 如果限制PPO采样出的文本长度和SFT的数据集长度类似，那么PPO的优势消失了。
  - 只用长度作为优势，也能取得和PPO类似性能


（对于R1系列模型的RL，有哪些观察？）
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs]()
  - Long2short RL，model merge，DPO...
- (其他复现报告，增长的length)


#### 1.2&nbsp;&nbsp;   (Mitigating) Solutions

##### 1.2.1&nbsp;&nbsp;For SFT stage

##### 1.2.2&nbsp;&nbsp;For RL stage
- [A Long Way to Go: Investigating Length Correlations in RLHF](https://arxiv.org/abs/2310.03716v2)
  - 设置高的KL loss，cut掉超出限度的rollout，设置长度相关的reward
- [Disentangling Length from Quality in Direct Preference Optimization](http://arxiv.org/abs/2403.19159)
  - DPO中，loss中加一个长度正则
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
  - 主要是针对DPO的改进，去掉reference，加上长度正则
- [Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback]()
- [ODIN: Disentangled Reward Mitigates Hacking in RLHF](http://arxiv.org/abs/2402.07319)


（对于R1系列模型的RL，有什么解决办法？）
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs]()
  - Long2short RL，model merge，DPO...


---

### ▶️ 2&nbsp;&nbsp;  Refineing Test-time Methods for Efficient Reasoning

#### 2.1&nbsp;&nbsp;   Fault Identification and Analysis

##### The cause of computation waste
- [Mind Your Step (by Step): Chain-of-Thought can Reduce Performance on Tasks where Thinking Makes Humans Worse]()

#### 2.2&nbsp;&nbsp;  (Mitigating) Solutions

##### 2.2.1&nbsp;&nbsp;  Budget Prediction & Allocation before Decoding

**Direct Prediction**

- [Token-Budget-Aware LLM Reasoning]()


**Difficulty-aware Prediction**
- [Make Every Penny Count: Difficulty-Adaptive Self-Consistency for Cost-Efficient Reasoning](http://arxiv.org/abs/2408.13457)




##### 2.2.2 &nbsp;&nbsp;  Adaptive Budget Allocation During Decoding

随着推理进行，在搜索过程中进行剪枝、early stop等操作。

**Early Stopping**
- [Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even Mid-Generation](https://arxiv.org/abs/2410.02725)
  - self-evaluation判断early stop
  
- [Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning](http://arxiv.org/abs/2401.10480)
  - 答案的consistency判断early stop

**Pruning while Searching**
- [Enhancing LLM Reasoning with Reward-guided Tree Search](https://arxiv.org/abs/2411.11694)
  - RM剪枝


---

### ▶️ 3&nbsp;&nbsp;   Post-training Calibrated with Inference Algorithm

设计post-training方法，并有配合的解码算法，实现efficient reasoning。

- [Learning How Hard to Think: Input-Adaptive Allocation of LM Computation](http://arxiv.org/abs/2410.04707)
- [InfAlign: Inference-aware language model alignment](https://arxiv.org/abs/2412.19792)
- [Adaptive Decoding via Latent Preference Optimization](https://arxiv.org/abs/2411.09661)


