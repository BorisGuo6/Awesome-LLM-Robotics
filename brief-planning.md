# LLM+P: Empowering Large Language Models with Optimal Planning Proficiency

### 详细总结：

文章介绍了一种新的框架，LLM+P，该框架结合了大型语言模型（LLMs）和经典规划器的优势，以解决自然语言描述的规划问题。LLM+P首先将自然语言描述的规划问题转换为规划领域定义语言（PDDL）文件，然后利用经典规划器快速找到解决方案，并将找到的解决方案翻译回自然语言。

#### 主要内容和方法：

1. **背景** ：

* LLMs，如GPT-4和ChatGPT，已经展示了出色的零射泛化能力，但不能可靠地解决长期规划问题。
* 经典规划器可以使用高效的搜索算法快速识别正确或最优的计划。

1. **LLM+P框架** ：

* LLM+P使用大型语言模型（LLM）生成给定问题的PDDL描述，然后利用经典规划器找到最优计划，最后再次使用LLM将原始计划翻译回自然语言。
* LLM+P假设对于每个问题领域，人类专家可以提供一个固定的领域描述，该描述适用于该领域中的所有问题实例。
* LLM+P可以直接作为机器人系统的自然语言界面，用于解决复杂的规划任务。

1. **实验和应用** ：

* 文章通过一系列实验展示了LLM+P能够为许多规划问题生成正确的解决方案，而单独的LLMs则无法做到这一点。
* LLM+P可以应用于任何我们有声音和完整解决方案的问题类别，例如，通过利用计算器解决算术问题。

1. **局限性** ：

* 文章没有要求LLM识别它已经提出了一个适合使用LLM+P管道处理的提示。
* 未来的研究方向包括考虑识别何时应该使用LLM+P处理提示。

### 结论：

LLM+P框架结合了大型语言模型和经典规划器的优势，以自然语言为界面，解决了复杂的规划任务。通过将问题描述转换为PDDL，利用经典规划器找到解决方案，并将解决方案翻译回自然语言，LLM+P展示了在规划问题上的优越性和应用潜力。

# Foundation Models for Decision Making: Problems, Methods, and Opportunities