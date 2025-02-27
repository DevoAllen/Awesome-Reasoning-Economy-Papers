# A survey of Efficient Thinking for Long Thought Large Language Models： From Decoding

## 备选题目方案

### candidates
1. Towards Efficient Reasoning in Large Language Models: A Comprehensive Survey of Adaptive Inference Strategies
   
2. Thinking Smart, Not Hard: A Review of Computation-Aware Reasoning in Large Language Models

3. The Inference Economy: A Review of Computation-Aware Reasoning in Large Language Models

4. Maximizing the Inference Economy: A Survey of Computationally Efficient Reasoning in Large Language Models

5. The Inference Economy of Complex Reasoning: A Systematic Review of Efficient Thinking in Large Language Models
    
6. Harnessing the Inference Economy: Efficient Thinking for Large Language Models

7. Efficient Reasoning in the Inference Economy: Adaptive Computation for Large Language Models

8. The Inference Economy: Adaptive Strategies for Efficient Reasoning in Large Language Models

### good parts  
1. Thinking Smart, Not Hard
2. Inference Economy
3. Computation-Aware or efficient
4. Efficient Thinking
5. optimal


## 1. Test-time Scaling

### 1.1 Benefits of TTS
test-time scaling 带来了明显的性能增长，无需训练。
Recent advances in inference-time techniques demonstrate the potential to enhance LLM reasoning without additional training by exploring intermediate steps during inference. 
test-time 方法
- Sequential Methods
    - critique and refine
    - Other prompting methods：
      - tot
      - got
      - 

- Parallel Methods
    - Beam Search
    - Best-of-N
    - Self-consistency

提高模型精度。
**paper（1）性能提升**

### 1.2 Why Efficient Thinking?
TTS方法要付出大量的算力，但很多时候付出的算力不一定带来性能提升。
**paper（2）算力开销计算**
**paper（4）模型能力分析，能力内，能力间，能力外**

算力浪费方式
（1）
（2）

尤其是，**paper（3）系列发现模型推理存在算力分配不合理**，即在固定算力下，简单题目，算力分配过多，造成算力浪费；困难题目，算力分配不足，导致模型无法充分探索解题可能性，影响性能。

**theory supports: Human Thinking Paradigm (Dual-Process Theory - System 1 & System 2)**

因此，一些paper希望动态分配算力，实现精度提升的同时，尽可能节省算力。
意义：
- 吞吐量
- 经济
- carbon footprint

## Adaptive Computation Methods分类

按照动机分类
是否用prior知识？

分小类别优势 / 劣势


- Mid/During decoding 在解码过程中动态调整，防止算力浪费。
  - Pruning while Decoding
  - Dynamically Adjusting Decoding Hypeparameters

- 优势： 测试时处理，训练完RM后，可以直接应用于模型；
- 劣势：RM的泛化性

---

- Before/after decoding， 在解码之前根据prompt分配算力，或者解码之后根据模型表现决定后续运算。
  - Early Stopping
  - Budget Prediction & Allocation

- 优势： 利用先验知识预先确定算力；使用历史经验（privious questions or current answers） 预测后续动作算力分配；
- 劣势： 往往需要大量资源采集先验知识；（博哥被喷的点）

---

- Architecture Methods
  - Implicit Inference
  - recurrent transformers
  - 非自回归解码策略


- 优势：
- 劣势：

*System-1/2 ？*

---
- RL based to alleviate length bias
  - SimPO


## Pruning while Decoding


## implicit inference
https://arxiv.org/pdf/2311.01460

https://arxiv.org/pdf/2405.14838v1

https://arxiv.org/pdf/2412.06769


## 6. Architecture Methods
1. MLA
2. MOE
3. recurrent transformers
4. reasoning in continuous space
5. 非自回归解码策略



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


## 10. System1/2协作框架 Dual-Process Theory (System 1/2)

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



