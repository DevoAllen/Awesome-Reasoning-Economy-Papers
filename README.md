# Harnessing the Inference Economy: A Survey of Efficient Reasoning for Large Language Models


- [Harnessing the Inference Economy: A Survey of Efficient Reasoning for Large Language Models](#harnessing-the-inference-economy-a-survey-of-efficient-reasoning-for-large-language-models)
    - [Introduction](#introduction)
    - [▶️ I: Refineing Post-training Methods for Efficient Reasoning](#️-i-refineing-post-training-methods-for-efficient-reasoning)
      - [I-i: Fault Identification and Analysis](#i-i-fault-identification-and-analysis)
        - [For SFT stage](#for-sft-stage)
        - [For RL stage](#for-rl-stage)
      - [I-ii: (Mitigating) Solutions](#i-ii-mitigating-solutions)
        - [For SFT stage](#for-sft-stage-1)
        - [For RL stage](#for-rl-stage-1)
    - [▶️ II: Refineing Test-time Methods for Efficient Reasoning](#️-ii-refineing-test-time-methods-for-efficient-reasoning)
      - [II-i: Fault Identification and Analysis](#ii-i-fault-identification-and-analysis)
        - [The cause of computation waste](#the-cause-of-computation-waste)
      - [II-ii: (Mitigating) Solutions](#ii-ii-mitigating-solutions)
        - [Budget Prediction \& Allocation before Decoding](#budget-prediction--allocation-before-decoding)
        - [Adaptive Budget Allocation During Decoding](#adaptive-budget-allocation-during-decoding)
    - [▶️ III: Post-training Calibrated with Inference Algorithm](#️-iii-post-training-calibrated-with-inference-algorithm)


### Introduction
Investing in improving inference-time computation might prove more beneficial than increasing model pre-training compute.


---

### ▶️ I: Refineing Post-training Methods for Efficient Reasoning


目前，研究普遍认为，在Reinforcement Learning from Human Feedback, RLHF阶段，可能会出现reward hacking现象，从而导致大型语言模型LLMs的输出结果存在潜在问题。其中，较为突出的算法所引发的长度偏差（length-bias），该问题会导致模型输出文本的长度不断增加，而其中的有用信息含量却相对较低。

因此一些工作尝试对这些现象进行分析。

最近，R1等以长推理（long reasoning）为重点的模型，推理能力很强，但是模型输出文本长度冗长。
无论是针对简单问题还是复杂问题，模型的回复相比于通用模型，均呈现出显著的冗长性。

因此，先前工作提出使用多种方法来尝试解决。

#### I-i: Fault Identification and Analysis


##### For SFT stage

##### For RL stage


- [A Long Way to Go: Investigating Length Correlations in RLHF]()

（对于R1系列模型的RL，有哪些观察？）



#### I-ii: (Mitigating) Solutions

##### For SFT stage

##### For RL stage
- [Disentangling Length from Quality in Direct Preference Optimization](http://arxiv.org/abs/2403.19159)
- [SimPO: Simple Preference Optimization with a Reference-Free Reward]()
- [Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback]()
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs]()
- [ODIN: Disentangled Reward Mitigates Hacking in RLHF](http://arxiv.org/abs/2402.07319)

（对于R1系列模型的RL，有什么解决办法？）

---

### ▶️ II: Refineing Test-time Methods for Efficient Reasoning

#### II-i: Fault Identification and Analysis

##### The cause of computation waste
- [Mind Your Step (by Step): Chain-of-Thought can Reduce Performance on Tasks where Thinking Makes Humans Worse]()

#### II-ii: (Mitigating) Solutions

##### Budget Prediction & Allocation before Decoding
- [Token-Budget-Aware LLM Reasoning]()
- [Make Every Penny Count: Difficulty-Adaptive Self-Consistency for Cost-Efficient Reasoning](http://arxiv.org/abs/2408.13457)

##### Adaptive Budget Allocation During Decoding
- [Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even Mid-Generation](https://arxiv.org/abs/2410.02725)
- [Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning](http://arxiv.org/abs/2401.10480)
- [Enhancing LLM Reasoning with Reward-guided Tree Search](https://arxiv.org/abs/2411.11694)

---

### ▶️ III: Post-training Calibrated with Inference Algorithm

- [Learning How Hard to Think: Input-Adaptive Allocation of LM Computation](http://arxiv.org/abs/2410.04707)
- [InfAlign: Inference-aware language model alignment](https://arxiv.org/abs/2412.19792)
- [Adaptive Decoding via Latent Preference Optimization](https://arxiv.org/abs/2411.09661)


