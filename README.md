# Harnessing the Inference Economy: A Survey of Efficient Reasoning for Large Language Models


- [Harnessing the Inference Economy: A Survey of Efficient Reasoning for Large Language Models](#harnessing-the-inference-economy-a-survey-of-efficient-reasoning-for-large-language-models)
    - [▶️ 1   Post-training Methods for Efficient Thinking reasoning LLMs](#️-1-post-training-methods-for-efficient-thinking-reasoning-llms)
      - [1.1     Post-training Induced Inefficiency](#11---post-training-induced-inefficiency)
        - [Length-bias](#length-bias)
        - [Deceptive Behaviors](#deceptive-behaviors)
      - [1.2  Mitigating Solutions](#12mitigating-solutions)
        - [Length-bias Alleviation](#length-bias-alleviation)
        - [Deceptive Behaviors Alleviation](#deceptive-behaviors-alleviation)
    - [▶️ 2    Refineing Test-time Methods for Efficient Reasoning](#️-2--refineing-test-time-methods-for-efficient-reasoning)
      - [2.1     Test-time Methods Induced Inefficiency](#21---test-time-methods-induced-inefficiency)
      - [2.2    (Mitigating) Solutions](#22--mitigating-solutions)
        - [2.2.1    Budget Prediction \& Allocation before Decoding](#221--budget-prediction--allocation-before-decoding)
        - [2.2.2     Adaptive Budget Allocation During Decoding](#222---adaptive-budget-allocation-during-decoding)
    - [▶️ 3     Post-training Calibrated with Inference Algorithm](#️-3---post-training-calibrated-with-inference-algorithm)
    - [▶️ 4     Emerging Frontiers in Efficient Reasoning](#️-4---emerging-frontiers-in-efficient-reasoning)
      - [4.1    Chain-of-Thought Compression](#41--chain-of-thought-compression)
      - [4.2    System-1 and System-2 Cooperation](#42--system-1-and-system-2-cooperation)
      - [4.3    Recurrent Depth Reasoning](#43--recurrent-depth-reasoning)



### ▶️ 1&nbsp;&nbsp; Post-training Methods for Efficient Thinking reasoning LLMs

RL optimization often relies on reward models (RMs) that are inherently imperfect, primarily due to unreliable human preference annotations. Additionally, Goodhart’s Law states, “When a measure becomes a target, it ceases to be a good measure.” Consequently, over-optimizing based on these flawed RMs can negatively impact the overall capabilities of LLMs.

This situation highlights the risk of reward hacking, where models exploit the reward function to achieve high scores without genuinely aligning with human preferences. 
As a result, LLMs may exhibit **Superficial Alignment**, appearing to meet human expectations while lacking true understanding.

#### 1.1&nbsp;&nbsp;   Post-training Induced Inefficiency

##### Length-bias
LLMs trained with RL tend to **produce longer responses** compared to those trained through SFT.

- **[A Long Way to Go: Investigating Length Correlations in RLHF](https://openreview.net/forum?id=G8LaO1P0xv#discussion)** 
- **[Fine-grained human feedback gives better rewards for language model training](https://dl.acm.org/doi/abs/10.5555/3666122.3668696)** 
- **[Learning to summarize from human feedback](https://dl.acm.org/doi/10.5555/3495724.3495977)** 


**❗️&nbsp;&nbsp;Findings  of Overly Cautious / Overthinking reasoning LLMs**

Reasoning LLMs exhibit excessive unnecessary verification and redundant reasoning on easy-to-handle questions, leading to inefficient token usage and increased computational costs.

- **[When More is Less: Understanding Chain-of-Thought Length in LLMs](https://arxiv.org/abs/2502.07266)**
- **[The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks](https://arxiv.org/abs/2502.08235)**
- **[Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs](https://arxiv.org/abs/2412.21187)**
- **[The Impact of Reasoning Step Length on Large Language Models](https://aclanthology.org/2024.findings-acl.108/)**
- **[Concise Thoughts: Impact of Output Length on LLM Reasoning and Cost](https://arxiv.org/abs/2407.19825)**



##### Deceptive Behaviors


- Fake Alignment: Are LLMs Really Aligned Well?


**❗️&nbsp;&nbsp;Findings of Fake Thinking reasoning LLMs**
Deceptive Behavior is even harder to detect than length bias.
Previous research has demonstrated that LLMs may display differential behaviors across various demographic groups \cite{fake-align-1}, thereby raising concerns regarding the authenticity and fairness of their alignment with human values.

- [When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs]
- 
#### 1.2&nbsp;&nbsp;Mitigating Solutions

##### Length-bias Alleviation
Disentangling the length and quality reward.

Length reward penalty (kl divergense, length penal).

Length preference optimization, (DPO, preference partial pair reconstruction).
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://openreview.net/forum?id=3Tzcot1LKb)
- 
**✨&nbsp;&nbsp;Overly Cautious / Overthinking reasoning LLMs**
Emerging research papers for reasoning LLMs

- [Self-Training Elicits Concise Reasoning in Large Language Models](https://arxiv.org/abs/2502.20122)
  - self-training for long2short
- [O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning](https://arxiv.org/abs/2501.12570)
- 
##### Deceptive Behaviors Alleviation


**✨&nbsp;&nbsp;Fake Thinking reasoning LLMs**



**✨&nbsp;&nbsp; For Both of Them**

- [Training Language Models to Reason Efficiently]
  - 训练模型根据难度分配算力
- Token-Budget-Aware LLM Reasoning


---

### ▶️ 2&nbsp;&nbsp;  Refineing Test-time Methods for Efficient Reasoning

#### 2.1&nbsp;&nbsp;   Test-time Methods Induced Inefficiency

**解析test-time算法，探究哪些因素导致test-time算法造成计算资源浪费。**

sequential inference: self-refine何时停止？
- [Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models](https://aclanthology.org/2024.acl-long.818/)

parallel inference: 采样窗口太大 / 太小，导致计算资源浪费；
- [Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems](https://proceedings.neurips.cc/paper_files/paper/2024/hash/51173cf34c5faac9796a47dc2fdd3a71-Abstract-Conference.html)
- [Cerberus: Efficient Inference with Adaptive Parallel Decoding and Sequential Knowledge Enhancement](https://arxiv.org/abs/2410.13344)

search 过程中：
- inter-sample
  1. 简单题过度探索
  2. 难题探索不足
- intra-sample
  1. 无效探索
  2. 过度探索
   
- [Mind Your Step (by Step): Chain-of-Thought can Reduce Performance on Tasks where Thinking Makes Humans Worse](https://arxiv.org/abs/2410.21333)
- [Enhancing LLM Reasoning with Reward-guided Tree Search](https://arxiv.org/abs/2411.11694)

#### 2.2&nbsp;&nbsp;  (Mitigating) Solutions

**针对以上对test-time算法造成的计算资源浪费，如何改进？**

##### 2.2.1&nbsp;&nbsp;  Budget Prediction & Allocation before Decoding

**Direct Prediction**

- [Token-Budget-Aware LLM Reasoning]()
- [s1: Simple test-time scaling]()
- [Following Length Constraints in Instructions]
- [Concise Thoughts: Impact of Output Length on LLM Reasoning and Cost]

**Difficulty-aware Prediction**
- [Make Every Penny Count: Difficulty-Adaptive Self-Consistency for Cost-Efficient Reasoning](http://arxiv.org/abs/2408.13457)




##### 2.2.2 &nbsp;&nbsp;  Adaptive Budget Allocation During Decoding

随着推理进行，在搜索过程中进行剪枝、early stop等操作。

**Early Stopping**
- [Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even Mid-Generation](https://arxiv.org/abs/2410.02725)
  - self-evaluation判断early stop
  
- [Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning](http://arxiv.org/abs/2401.10480)
  - 答案的consistency判断early stop

- [Let’s Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs](https://aclanthology.org/2023.emnlp-main.761/)


**Pruning while Searching**
- [Enhancing LLM Reasoning with Reward-guided Tree Search](https://arxiv.org/abs/2411.11694)
  - RM剪枝
- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://aclanthology.org/2024.acl-long.510/)
- [OVM, Outcome-supervised Value Models for Planning in Mathematical Reasoning](https://aclanthology.org/2024.findings-naacl.55/)


---

### ▶️ 3&nbsp;&nbsp;   Post-training Calibrated with Inference Algorithm

Design post-training methods, along with corresponding inference algorithms, to achieve efficient reasoning.


- [Learning How Hard to Think: Input-Adaptive Allocation of LM Computation](http://arxiv.org/abs/2410.04707)
- [InfAlign: Inference-aware language model alignment](https://arxiv.org/abs/2412.19792)
- [BOND: Aligning LLMs with Best-of-N Distillation](https://arxiv.org/abs/2407.14622)
  - 训练对齐BoN，不需要测试时使用完全版BoN，减少开销
- [Adaptive Decoding via Latent Preference Optimization](https://arxiv.org/abs/2411.09661)
  - 训练：预测每个token需要的温度
  - 推理：使用训练后的温度预测器确定每个token使用的温度
- [L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning](https://arxiv.org/abs/2503.04697)
  - 训练：LCPO，训练中两个奖励：accuracy，和满足prompt中长度
  - 推理：prompt指定推理长度


### ▶️ 4&nbsp;&nbsp;   Emerging Frontiers in Efficient Reasoning


#### 4.1&nbsp;&nbsp;  Chain-of-Thought Compression

压缩cot，缩短推理长度。


**Explicit Compression**
- [TokenSkip: Controllable Chain-of-Thought Compression in LLMs](https://arxiv.org/abs/2502.12067.pdf)
  - 跳过不关键token
- [Stepwise Perplexity-Guided Refinement for Efficient Chain-of-Thought Reasoning in Large Language Models](https://arxiv.org/abs/2502.13260.pdf)
  - 跳过不关键步骤，专注关键步骤

**Implicit Compression**
- [Think before you speak: Training Language Models With Pause Tokens] `do not think this is relevant`
- [SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs](https://arxiv.org/abs/2502.12134.pdf)
- [LightThinker: Thinking Step-by-Step Compression](https://arxiv.org/pdf/2502.15589.pdf)
- [Compressed chain of thought: Efficient reasoning through dense representations.](https://arxiv.org/pdf/2412.13171.pdf)
- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/pdf/2412.06769.pdf)
- [From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step](https://arxiv.org/abs/2405.14838.pdf)
- [Implicit Chain of Thought Reasoning via Knowledge Distillation](https://arxiv.org/abs/2311.01460.pdf)

#### 4.2&nbsp;&nbsp;  System-1 and System-2 Cooperation

尽管prover，verifer的模型也可以称作双模型合作，但此处我们特指双prover的合作

**同架构模型协作**

Speculative Decoding
- [JUDGE DECODING: FASTER SPECULATIVE SAMPLING REQUIRES GOING BEYOND MODEL ALIGNMENT](https://arxiv.org/pdf/2501.19309.pdf)
- [Reward-Guided Speculative Decoding for Efficient LLM Reasoning](https://arxiv.org/pdf/2501.19324.pdf)
- [MAgICoRe: Multi-Agent, Iterative, Coarse-to-Fine Refinement for Reasoning](https://arxiv.org/abs/2409.12147.pdf) `also not very suitable here`
- [Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding](https://aclanthology.org/2024.findings-acl.456.pdf)
- [Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation](https://aclanthology.org/2023.findings-emnlp.257/)
- [Speculative Decoding with Big Little Decoder](https://arxiv.org/pdf/2302.07863.pdf)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192.pdf)

Model Routing
1. sample level routing
     - [System-1.x: Learning to Balance Fast and Slow Planning with Language Models]
     - [Synergy-of-Thoughts: Eliciting Efficient Reasoning in Hybrid Language Models]
     - [DynaThink: Fast or Slow? A Dynamic Decision-Making Framework for Large Language Models]
  
2. token level routing
     - [CITER: Collaborative Inference for Efficient Large Language Model Decoding with Token-Level Routing]
  

**异架构模型协作**
too few


**Distill System 2 to System 1**

同构（大transformer蒸馏到小transformer）
- [deepseek-r1]()
  - deepseekdistilled qwen
- [Distilling System 2 into System 1](https://arxiv.org/abs/2407.06023)

异构（高复杂度的transformer，蒸馏到低复杂度模型mamba）
transformer蒸馏到mamba
- [Thinking Slow, Fast: Scaling Inference Compute with Distilled Reasoners](https://arxiv.org/abs/2502.20339)

#### 4.3&nbsp;&nbsp;  Recurrent Depth Reasoning
1. 推理中模型深度很重要
   - [Physics of Language Models: Part 2.1]
     - Language model depth is crucial for mathematical reasoning.
   - [What Can Transformer Learn with Varying Depth? Case Studies on Sequence Learning Tasks]
     - 模型不同深度学习能力不同，即使参数相同
2. 小模型重复使用其中某些层，实现深度推理
   - [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach]
   - [Enhancing Auto-regressive Chain-of-Thought through Loop-Aligned Reasoning]

3. 这样训练、推理开销都小
4. 也可以跳过transformer某些不重要的层
   - [Not All Layers of LLMs Are Necessary During Inference](https://arxiv.org/abs/2403.02181)


---






