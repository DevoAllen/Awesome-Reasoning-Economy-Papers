# <img src="figures/productivity.png" alt="Example Figure" width="50" height="50" /> Harnessing the Inference Economy: A Survey of Efficient Reasoning for Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-Reasoning_Economy-b31b1b.svg)]()
[![Contribution Welcome](https://img.shields.io/badge/Contributions-welcome-blue)]()

<!-- omit in toc -->
## 📢 Updates

- **2025.03**: We released a github repo to record papers related with reasoning economy. Feel free to cite or open pull requests.

<!-- omit in toc -->

- [ Harnessing the Inference Economy: A Survey of Efficient Reasoning for Large Language Models](#-harnessing-the-inference-economy-a-survey-of-efficient-reasoning-for-large-language-models)
    - [▶️ 1   Foundation of Reasoning LLMs](#️-1-foundation-of-reasoning-llms)
      - [1.1  Post-training Methods for Reasoning LLMs](#11-post-training-methods-for-reasoning-llms)
      - [1.2  Test-time Methods for Reasoning LLMs](#12-test-time-methods-for-reasoning-llms)
    - [▶️ 2   Challenges towards Reasoning Economy](#️-2-challenges-towards-reasoning-economy)
      - [2.1     Inefficient Model Behaviors from Post-training](#21---inefficient-model-behaviors-from-post-training)
        - [2.1.1      Length-bias](#211----length-bias)
        - [2.1.2   Deceptive Behaviors](#212-deceptive-behaviors)
      - [2.2     Inefficient Model Usage in Test-time](#22---inefficient-model-usage-in-test-time)
        - [2.2.1   Unreasonable Algorithm Selection](#221-unreasonable-algorithm-selection)
        - [2.2.2   Unreasonable Computation Allocation](#222-unreasonable-computation-allocation)
    - [▶️ 3   Optimization for Reasoning Economy *part-1: Post-training*](#️-3-optimization-for-reasoning-economy-part-1-post-training)
      - [3.1    Data](#31--data)
      - [3.2    Algorithm](#32--algorithm)
        - [3.2.1    Long2short RL](#321--long2short-rl)
        - [3.2.2    Adaptive Budget-aware Tuning](#322--adaptive-budget-aware-tuning)
        - [3.2.3    CoT Compression](#323--cot-compression)
      - [3.3    Architecture](#33--architecture)
        - [3.3.1    System-1 and System-2 Cooperation](#331--system-1-and-system-2-cooperation)
        - [3.3.2    Adaptive Activated Parameters](#332--adaptive-activated-parameters)
    - [▶️ 4   Optimization for Reasoning Economy *part-2: Test-time Methods*](#️-4-optimization-for-reasoning-economy-part-2-test-time-methods)
      - [4.1    Input-side Optimization](#41--input-side-optimization)
        - [4.1.1    Adaptive Budget Allocation before Decoding](#411--adaptive-budget-allocation-before-decoding)
      - [4.2    Output-side Optimization](#42--output-side-optimization)
        - [4.2.1    Adaptive Algorithm Selection](#421--adaptive-algorithm-selection)
        - [4.2.2    Adaptive Budget Allocation During Decoding](#422--adaptive-budget-allocation-during-decoding)
    - [▶️ 5   Discussion](#️-5-discussion)
  - [Citation](#citation)


---

### ▶️ 1&nbsp;&nbsp; Foundation of Reasoning LLMs



#### 1.1 &nbsp;Post-training Methods for Reasoning LLMs
- [From System 1 to System 2: A Survey of Reasoning Large Language Models](http://arxiv.org/abs/2502.17419)
- [A Survey on Post-training of Large Language Models](http://arxiv.org/abs/2503.06072)
- [LLM Post-Training: A Deep Dive into Reasoning Large Language Models](https://arxiv.org/abs/2502.21321)


**Supervised Finetuning**
- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)
- [LIMA: Less Is More for Alignment](http://arxiv.org/abs/2305.11206)
- [LIMR: Less is More for RL Scaling](https://arxiv.org/abs/2502.11886)
- [s1: Simple test-time scaling](http://arxiv.org/abs/2501.19393)
- [Large Language Models Can Self-Improve](https://aclanthology.org/2023.emnlp-main.67/)
- [Improving Language Model Reasoning with Self-motivated Learning](https://aclanthology.org/2024.lrec-main.774/)
- [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)
- [Reinforced Self-Training (ReST) for Language Modeling](https://arxiv.org/abs/2308.08998)
- [Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains](https://openreview.net/forum?id=JtGPIZpOrz)

**Reinforcement Learning**
- [Proximal Policy Optimization Algorithms](http://arxiv.org/abs/1707.06347)
- [Group Robust Preference Optimization in Reward-free RLHF](https://openreview.net/forum?id=PRAsjrmXXK)
- [SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training](https://arxiv.org/abs/2501.17161)
- [The Lessons of Developing Process Reward Models in Mathematical Reasoning](https://arxiv.org/abs/2501.07301)
- [QwQ: Reflect Deeply on the Boundaries of the Unknown](https://qwenlm.github.io/blog/qwq-32b/)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) 
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)
- [ReFT: Reasoning with Reinforced Fine-Tuning](https://arxiv.org/abs/2401.08967)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)

#### 1.2 &nbsp;Test-time Methods for Reasoning LLMs

**Parallel Methods**
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://openreview.net/forum?id=1PL1NIMMrw)
- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787)
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](http://arxiv.org/abs/2408.03314)

**Sequential Methods**
- [Chain-of-thought Prompting Elicits Reasoning in Large Language Models](https://dl.acm.org/doi/10.5555/3600270.3602070)
- [Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/abs/2310.04406)
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
- [Self-Evaluation Guided Beam Search for Reasoning](https://arxiv.org/abs/2305.00633)
- [Don't throw away your value model! Generating more preferable text with Value-Guided Monte-Carlo Tree Search decoding](https://arxiv.org/abs/2309.15028)
- [OVM, Outcome-supervised Value Models for Planning in Mathematical Reasoning](https://aclanthology.org/2024.findings-naacl.55/)

---

### ▶️ 2&nbsp;&nbsp; Challenges towards Reasoning Economy

#### 2.1&nbsp;&nbsp;   Inefficient Model Behaviors from Post-training

##### 2.1.1 &nbsp;&nbsp;   Length-bias

- [A Long Way to Go: Investigating Length Correlations in RLHF](https://openreview.net/forum?id=G8LaO1P0xv#discussion)
- [Fine-grained human feedback gives better rewards for language model training](https://dl.acm.org/doi/abs/10.5555/3666122.3668696)
- [Learning to summarize from human feedback](https://dl.acm.org/doi/10.5555/3495724.3495977)
- [Scaling Laws for Reward Model Overoptimization](http://arxiv.org/abs/2210.10760)
- [Defining and Characterizing Reward Hacking](https://arxiv.org/abs/2209.13085)
  

**❗️&nbsp;&nbsp;Findings  of Overly Cautious reasoning LLMs**

- [When More is Less: Understanding Chain-of-Thought Length in LLMs](https://arxiv.org/abs/2502.07266)
- [The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks](https://arxiv.org/abs/2502.08235)
- [Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs](https://arxiv.org/abs/2412.21187)
- [The Impact of Reasoning Step Length on Large Language Models](https://aclanthology.org/2024.findings-acl.108/)
- [Concise Thoughts: Impact of Output Length on LLM Reasoning and Cost](https://arxiv.org/abs/2407.19825)
- [Over-Reasoning and Redundant Calculation of Large Language Models](https://arxiv.org/abs/2401.11467)
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)
- [The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks](https://arxiv.org/abs/2502.08235)
- [Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models](https://aclanthology.org/2024.acl-long.818/)
  
##### 2.1.2 &nbsp;&nbsp;Deceptive Behaviors


- [Fake Alignment: Are LLMs Really Aligned Well?](https://aclanthology.org/2024.naacl-long.263/)
- [Alignment faking in large language models](https://arxiv.org/abs/2412.14093)
- [Large Language Models Often Say One Thing and Do Another](https://openreview.net/forum?id=RTHbao4Mib)

**❗️&nbsp;&nbsp;Findings of Fake Thinking reasoning LLMs**

- [When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs](https://aclanthology.org/2024.tacl-1.78/)
- [Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs](http://arxiv.org/abs/2501.18585)
- [Large Language Models Cannot Self-Correct Reasoning Yet](https://openreview.net/forum?id=IkmD3fKBPQ)
- [PhD Knowledge Not Required: A Reasoning Challenge for Large Language Models](https://arxiv.org/abs/2502.01584)
- [Unlocking the Capabilities of Thought: A Reasoning Boundary Framework to Quantify and Optimize Chain-of-Thought](https://openreview.net/forum?id=pC44UMwy2v)


#### 2.2&nbsp;&nbsp;   Inefficient Model Usage in Test-time


##### 2.2.1 &nbsp;&nbsp;Unreasonable Algorithm Selection

- [Inference-Time Computations for LLM Reasoning and Planning: A Benchmark and Insights](https://arxiv.org/abs/2502.12521)
- [Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models](https://aclanthology.org/2024.acl-long.818/)
- [Scaling Test-Time Compute Without Verification or RL is Suboptimal](https://arxiv.org/abs/2502.12118)
- [Adaptive Decoding via Latent Preference Optimization](https://arxiv.org/abs/2411.09661)

##### 2.2.2 &nbsp;&nbsp;Unreasonable Computation Allocation
- [Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems](https://proceedings.neurips.cc/paper_files/paper/2024/hash/51173cf34c5faac9796a47dc2fdd3a71-Abstract-Conference.html)
- [Cerberus: Efficient Inference with Adaptive Parallel Decoding and Sequential Knowledge Enhancement](https://arxiv.org/abs/2410.13344)
- [When More is Less: Understanding Chain-of-Thought Length in LLMs](https://arxiv.org/abs/2502.07266)
- [Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](https://arxiv.org/abs/2502.18080)


---

### ▶️ 3&nbsp;&nbsp; Optimization for Reasoning Economy *part-1: Post-training*
#### 3.1 &nbsp;&nbsp; Data
- [s1: Simple test-time scaling](http://arxiv.org/abs/2501.19393)
- [O1 Replication Journey: A Strategic Progress Report -- Part 1](https://arxiv.org/abs/2410.18982)
- [LIMA: Less Is More for Alignment](http://arxiv.org/abs/2305.11206)
- [LIMR: Less is More for RL Scaling](https://arxiv.org/abs/2502.11886)
  

#### 3.2 &nbsp;&nbsp; Algorithm

##### 3.2.1 &nbsp;&nbsp; Long2short RL

- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://openreview.net/forum?id=3Tzcot1LKb)
- [A Long Way to Go: Investigating Length Correlations in RLHF](https://openreview.net/forum?id=G8LaO1P0xv#discussion)
- [Disentangling Length from Quality in Direct Preference Optimization](https://aclanthology.org/2024.findings-acl.297/)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.07572)
- [Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs](https://arxiv.org/abs/2412.21187)
- [Self-Training Elicits Concise Reasoning in Large Language Models](https://arxiv.org/abs/2502.20122)
- [O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning](https://arxiv.org/abs/2501.12570)
  

##### 3.2.2 &nbsp;&nbsp; Adaptive Budget-aware Tuning
- [Training Language Models to Reason Efficiently](https://arxiv.org/abs/2502.04463)
- [Token-Budget-Aware LLM Reasoning](https://arxiv.org/abs/2412.18547)
- [Mind Your Step (by Step): Chain-of-Thought can Reduce Performance on Tasks where Thinking Makes Humans Worse](https://arxiv.org/abs/2410.21333)
- [Enhancing LLM Reasoning with Reward-guided Tree Search](https://arxiv.org/abs/2411.11694)
- [L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning](https://arxiv.org/abs/2503.04697)
  

##### 3.2.3 &nbsp;&nbsp; CoT Compression
**Explicit Compression**
- [TokenSkip: Controllable Chain-of-Thought Compression in LLMs](https://arxiv.org/abs/2502.12067)
- [Stepwise Perplexity-Guided Refinement for Efficient Chain-of-Thought Reasoning in Large Language Models](https://arxiv.org/abs/2502.13260)
- [Can Language Models Learn to Skip Steps?](https://arxiv.org/abs/2411.01855)
  
**Implicit Compression**
- [Think before you speak: Training Language Models With Pause Tokens](https://openreview.net/forum?id=ph04CRkPdC)
- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)
- [From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step](https://arxiv.org/abs/2405.14838)
- [Implicit Chain of Thought Reasoning via Knowledge Distillation](https://arxiv.org/abs/2311.01460)
- [SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs](https://arxiv.org/abs/2502.12134)
- [LightThinker: Thinking Step-by-Step Compression](https://arxiv.org/abs/2502.15589)
- [Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning](https://arxiv.org/abs/2502.03275)


#### 3.3 &nbsp;&nbsp; Architecture
##### 3.3.1 &nbsp;&nbsp; System-1 and System-2 Cooperation

**Single Model Routing**
- [System-1.x: Learning to Balance Fast and Slow Planning with Language Models](https://openreview.net/forum?id=zd0iX5xBhA)
- [OpenAI o1](https://openai.com/o1/)

**Multi-model Cooperation**
- [JUDGE DECODING: FASTER SPECULATIVE SAMPLING REQUIRES GOING BEYOND MODEL ALIGNMENT](https://openreview.net/forum?id=mtSSFiqW6y)
- [Speculative Decoding with Big Little Decoder](https://dl.acm.org/doi/abs/10.5555/3666122.3667827)
- [MAgICoRe: Multi-Agent, Iterative, Coarse-to-Fine Refinement for Reasoning](https://arxiv.org/abs/2409.12147)
- [Reward-Guided Speculative Decoding for Efficient LLM Reasoning](https://arxiv.org/abs/2501.19324)
- [Agents Thinking Fast and Slow: A Talker-Reasoner Architecture](https://arxiv.org/abs/2410.08328)
- [Synergy-of-Thoughts: Eliciting Efficient Reasoning in Hybrid Language Models](https://arxiv.org/abs/2402.02563)
- [DynaThink: Fast or Slow? A Dynamic Decision-Making Framework for Large Language Models](https://aclanthology.org/2024.emnlp-main.814/)
- [CITER: Collaborative Inference for Efficient Large Language Model Decoding with Token-Level Routing](http://arxiv.org/abs/2502.01976)
- [Unlocking Efficient Long-to-Short LLM Reasoning with Model Merging](http://arxiv.org/abs/2503.20641)

**Knowledge Distillation**
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [Distilling System 2 into System 1](https://arxiv.org/abs/2407.06023)
- [Thinking Slow, Fast: Scaling Inference Compute with Distilled Reasoners](https://arxiv.org/abs/2502.20339)


##### 3.3.2 &nbsp;&nbsp; Adaptive Activated Parameters
- [Physics of Language Models: Part 2.1](https://arxiv.org/abs/2407.20311)
- [What Can Transformer Learn with Varying Depth? Case Studies on Sequence Learning Tasks](https://dl.acm.org/doi/10.5555/3692070.3692384)
- [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)
- [Enhancing Auto-regressive Chain-of-Thought through Loop-Aligned Reasoning](https://arxiv.org/abs/2502.08482)
- [Not All Layers of LLMs Are Necessary During Inference](https://arxiv.org/abs/2403.02181)
- [LaCo: Large Language Model Pruning via Layer Collapse](https://aclanthology.org/2024.findings-emnlp.372/)
- [Can Looped Transformers Learn to Implement Multi-step Gradient Descent for In-context Learning?](https://arxiv.org/abs/2410.08292)



---

### ▶️ 4&nbsp;&nbsp; Optimization for Reasoning Economy *part-2: Test-time Methods*


#### 4.1 &nbsp;&nbsp; Input-side Optimization
##### 4.1.1 &nbsp;&nbsp; Adaptive Budget Allocation before Decoding

- [Following Length Constraints in Instructions](https://arxiv.org/abs/2406.17744)
- [Make Every Penny Count: Difficulty-Adaptive Self-Consistency for Cost-Efficient Reasoning](http://arxiv.org/abs/2408.13457)
- [Token-Budget-Aware LLM Reasoning](https://arxiv.org/abs/2412.18547)
- [Concise Thoughts: Impact of Output Length on LLM Reasoning and Cost](https://openreview.net/forum?id=tg8okrv4Rz)
- [INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection](https://arxiv.org/abs/2402.03744)



#### 4.2 &nbsp;&nbsp; Output-side Optimization



##### 4.2.1 &nbsp;&nbsp; Adaptive Algorithm Selection
- [Adaptive Decoding via Latent Preference Optimization](https://arxiv.org/abs/2411.09661)
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](http://arxiv.org/abs/2408.03314)
- [Flaming-hot Initiation with Regular Execution Sampling for Large Language Models](https://doi.org/10.48550/arXiv.2410.21236)
##### 4.2.2 &nbsp;&nbsp; Adaptive Budget Allocation During Decoding

**Early Stopping**
- [Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even Mid-Generation](https://arxiv.org/abs/2410.02725)
- [Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning](http://arxiv.org/abs/2401.10480)
- [Let’s Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs](https://aclanthology.org/2023.emnlp-main.761/)
- [Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](https://arxiv.org/abs/2503.01422)
- [Reasoning Aware Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling](https://arxiv.org/abs/2408.17017)
- [Efficient Test-Time Scaling via Self-Calibration](https://arxiv.org/abs/2503.00031)

**Pruning while Searching**
- [Enhancing LLM Reasoning with Reward-guided Tree Search](https://arxiv.org/abs/2411.11694)
- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://aclanthology.org/2024.acl-long.510/)
- [Path-Consistency: Prefix Enhancement for Efficient Inference in LLM](https://arxiv.org/abs/2409.01281)
- [Self-Evaluation Guided Beam Search for Reasoning](https://arxiv.org/abs/2305.00633)
- [OVM, Outcome-supervised Value Models for Planning in Mathematical Reasoning](https://aclanthology.org/2024.findings-naacl.55/)
- [Fast Best-of-N Decoding via Speculative Rejection](https://arxiv.org/abs/2410.20290)
  
**Constrained Decoding**
- [s1: Simple test-time scaling](http://arxiv.org/abs/2501.19393)
- [Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs](http://arxiv.org/abs/2501.18585)
- [Large Language Models Cannot Self-Correct Reasoning Yet](https://openreview.net/forum?id=IkmD3fKBPQ)
- [CRITIC: Large Language Models Can Self-Correct with ToolInteractive Critiquing](https://arxiv.org/abs/2305.11738)

---
### ▶️ 5&nbsp;&nbsp; Discussion 

**Efficient Multi-modal Reasoning**


**Efficient Agentic Reasoning**

- [ATLAS: Agent Tuning via Learning Critical Steps](https://arxiv.org/pdf/2503.02197.pdf)
- [ReSo: A Reward-driven Self-organizing LLM-based Multi-Agent System for Reasoning Tasks](https://arxiv.org/abs/2503.02390.pdf)

**Evaluation Metrics and Benchmarks**
- [Scaling Evaluation-time Compute with Reasoning Models as Process Evaluators](https://arxiv.org/pdf/2503.19877.pdf)
- [PhD Knowledge Not Required: A Reasoning Challenge for Large Language Models](https://arxiv.org/abs/2502.01584)




**Explaniabilty of Reasoning LLMs**
- [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#related-work)
- [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)


---

## Citation
If you find this work useful, welcome to cite us.
```bib

```

<!-- omit in toc -->
## ⭐ Star History

<a href="https://www.star-history.com/#DevoAllen/EfficientThinking&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=DevoAllen/EfficientThinking&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=DevoAllen/EfficientThinking&type=Date&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=DevoAllen/EfficientThinking&type=Date&type=Date" />
 </picture>
</a>



