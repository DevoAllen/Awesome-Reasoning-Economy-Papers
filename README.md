# A survey of Efficient Thinking for Long Thought Large Language Models： From Decoding

## 备选题目方案

### candidates
1. Towards Efficient Reasoning in Large Language Models: A Comprehensive Survey of Adaptive Inference Strategies
   
2. Thinking Smart, Not Hard: A Review of Computation-Aware Reasoning in Large Language Models

3. Maximizing the Inference Economy: A Review of Computation-Aware Reasoning in Large Language Models

4. Maximizing the Inference Economy: A Survey of Computationally Efficient Reasoning in Large Language Models

5. The Inference Economy of Complex Reasoning: A Systematic Review of Efficient Thinking in Large Language Models
    
6. Harnessing the Inference Economy: Efficient Thinking for Large Language Models

### good parts  
1. Thinking Smart, Not Hard
2. Inference Economy
3. Computation-Aware or efficient
4. Efficient Thinking


## 1. Human Thinking Phenomena 
### 1.1 System 1 & System 2



## 2. Source of Token Waste

### 2.1 Length-bias during RL


## 3. Test-time Scaling


## 4. Why Efficient Thinking?

theory supports


## 5. Adaptive Computation Allocation


## 6. Architecture Methods
1. MLA
2. MOE
3. recurrent transformers
4. reasoning in continuous space
5. 非自回归解码策略

## 7. Pruning while Decoding


## 8. Early Stop

**主要思想是先做，做得好了就停，别浪费。**

<a href = "https://arxiv.org/abs/2410.02725"> Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even Mid-Generation</a>

训练模型自己确认每一次完成推理后，是否需要再来一次。


<a href = "https://arxiv.org/abs/2401.10480"> Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning</a>
early stop 的self-consistency，training-free。只要一个窗口里面所有答案都一致，那么后续就不再采样。



## 9. Budget Prediction & Allocation

**主要思想是根据先验确定模型推理该问题需要的最优预算，再根据该预算直接进行推理。**


<a href = "https://arxiv.org/abs/2408.03314">Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters</a>

想权衡下分配在test和post-train阶段的算力；并且提出模型**根据难度**不同分配其最优的算力。

<a href = "https://arxiv.org/abs/2410.04707">LEARNING HOW HARD TO THINK: INPUT-ADAPTIVE ALLOCATION OF LM COMPUTATION</a>

使用线性规划，指定算力和请求后，找出最大奖励的算力分配



<a href = "https://arxiv.org/abs/2408.13457">Make Every Penny Count: Difficulty-Adaptive Self-Consistency for Cost-Efficient Reasoning</a>

**根据难度**，调整不同问题的self-consistency窗口大小


<a href = "https://arxiv.org/abs/2412.18547">Token-Budget-Aware LLM Reasoning</a>

比较新的一个。prompt告诉token预算，但是精度有损失多。


## 10. Compound Inference Systems

**主要思想是大的LLM解决难问题，小的解决简单问题。system 1 & system2**


<a href = "https://arxiv.org/abs/2407.13692">Prover-Verifier Games improve legibility of LLM outputs</a>
主要针对可读性，但是文中提了length-bias问题。


<a href = "https://arxiv.org/abs/2410.04707">LEARNING HOW HARD TO THINK: INPUT-ADAPTIVE ALLOCATION OF LM COMPUTATION</a>



该工作也考虑了大小模型协作。


## 11. RL

Length-bias等系列算法

<a href = "https://arxiv.org/abs/2405.14734">SimPO: Simple Preference Optimization with a Reference-Free Reward</a>


Long2short

<a href = "https://arxiv.org/abs/2501.12599">KIMI K1.5: SCALING REINFORCEMENT LEARNING WITH LLMS</a>



## 12. 其他

<a href = "https://aclanthology.org/2024.emnlp-main.1112/">Reasoning in Token Economies: Budget-Aware Evaluation of LLM Reasoning Strategies</a>

提出评估prompt方法效果时，要将其放在同一算力成本下考虑。

<a href = "https://arxiv.org/abs/2403.02419">Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems</a>

考虑难度，不是所有的问题best-of-n都能带来更好的性能



## Future Directions



