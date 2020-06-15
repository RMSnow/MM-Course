# Challenge1：跨话题迁移

## 问题分析

有研究表明：在一些话题的数据集（可被称为”通用话题“）上训练得到的模型，在一些特定话题的数据集上，泛化性能较差。例如：用社会生活话题下的数据训练模型，若直接对军事话题下的数据进行预测，则其各项分类指标表现均有很大的下降。

由此引出”跨话题迁移“的任务：

- 数据集中的新闻共包含社会生活、医药健康、文体娱乐、财经商业、教育考试、科技、军事、政治，共8种话题的新闻；
- 将社会生活、医药健康、文体娱乐、财经商业、教育考试、科技等6类话题下的数据作为训练集；将军事、政治这2类话题下的数据作为测试集；
- 通过迁移学习的相关方法，希望能够提高模型在测试集上的性能。

## 数据集划分

|        | 话题                                                         | 样本数 | 正负样本数量   | 有图片样本  |
| ------ | ------------------------------------------------------------ | ------ | -------------- | ----------- |
| 训练集 | 社会生活、医药健康、文体娱乐、财经商业、教育考试、科技 (共 6 类) | 32193  | 16119+，16704- | 19006 (59%) |
| 测试集 | 军事、政治 (共 2 类)                                         | 1613   | 891+，722-     | 1061 (66%)  |

## 数据分析和预处理

### 类别分布

#### 训练集

```
【样本数量】
社会生活    20616
医药健康     6320
文体娱乐     2620
财经商业     1455
教育考试      901
科技        281

【百分比】
社会生活    0.640388
医药健康    0.196316
文体娱乐    0.081384
财经商业    0.045196
教育考试    0.027987
科技      0.008729
```

#### 测试集

```
【样本数量】
政治    1241
军事     372

【百分比】
政治    0.769374
军事    0.230626
```

### 文本预处理

1. 去除url

2. 采用`jieba`进行分词，分词后，统计最大句子长度：

   ```
   The longest sentence has 1631 words. When WORDS = 100, the cover_rate = 0.93
   The longest sentence has 1631 words. When WORDS = 120, the cover_rate = 0.96
   The longest sentence has 1631 words. When WORDS = 150, the cover_rate = 0.97
   The longest sentence has 1631 words. When WORDS = 200, the cover_rate = 0.98
   ```

   在后续的文本特征提取器中，均选择最大句子长度为120（能够覆盖96%的数据样本）

## 尝试1: EANN‘s idea

> EANN: Event Adversarial Neural Networks for MultiModal Fake News Detection. KDD 2018: 849-857

### 模型架构

<img src="/Users/snow/Library/Application Support/typora-user-images/image-20200615153535866.png" style="zoom:40%" />

如图所示：在经过特征提取器的处理后，得到的特征向量分别输入到了Fake News判别器和Topic判别器，记二者的loss function分别为$L_d$、$L_t$。具体地：

- 特征提取器：为针对文本数据的BiGRU模型。
- Topic判别器：输入为8类，对应着8类话题。

整个模型的Loss Function、Objective Fuction、Optimization Strategy 如下图所示：

<img src="/Users/snow/Library/Application Support/typora-user-images/image-20200615174852612.png" style="zoom:40%" />

模型具有以下特点：

- 在迁移学习中，该方法属于对边缘分布的迁移。
- 模型属于端对端的模型，仅需一次训练。
- 该模型利用两个输出的loss，进行min-max game的对抗训练，即在“ Fake News 判得准” 和 “Topic 分得准” 之间做一个博弈。

### 性能评估

<img src="/Users/snow/Library/Application Support/typora-user-images/image-20200615175146070.png" style="zoom:40%" />

其中，BiGRU为Baseline模型，其没有进行迁移。可以看到，在利用EANN模型进行话题迁移后，性能反而有所下降。

![image-20200615175320411](/Users/snow/Library/Application Support/typora-user-images/image-20200615175320411.png)

用tsne可视化方法，将两个模型的源域（train）、目标域（test）的分布展示出来，可发现：

- 对于“BiGRU”模型，可看出：在不经过迁移时，源域、目标域的边缘分布没有对齐，这说明了跨话题迁移的必要性；
- 对于“BiGRU+EANN迁移”模型，可看出：虽然经过了EANN的迁移，但源域、目标域的边缘分布仍没有对齐，这表明EANN的迁移效果不如预期。

### 原因分析

由于训练集中只有 6 类 Topic 的数据，因此模型没有学到对军事、政治这两类 Topic 进行分类的能力。可能的解决方案：

> <img src="/Users/snow/Library/Application Support/typora-user-images/image-20200615181107105.png" style="zoom:40%" />
>
> Domain-Adversarial Training of Neural Networks. J. Mach. Learn. Res. 17: 59:1-59:35 (2016)

采用DANN论文中的处理，更改其loss function为：

<img src="/Users/snow/Library/Application Support/typora-user-images/image-20200615181312679.png" style="zoom:40%" />

其中，可以仅利用测试集的topic label（不使用fake label，因此不存在数据泄露问题），这样模型就可以学到对军事、政治这两类 Topic 进行分类的能力。

## 尝试2：Two Branches‘ Idea

> <img src="/Users/snow/Library/Application Support/typora-user-images/image-20200615181631897.png" style="zoom:40%" />
>
> Integrating Semantic and Structural Information with Graph Convolutional Network for Controversy Detection. ACL 2020 accepted.

### 模型架构

<img src="/Users/snow/Library/Application Support/typora-user-images/image-20200616002546480.png" style="zoom:40%" />

如图所示，模型的训练流程如下：

1. 训练 Related Branch 的Topic 分类器，分类损失最小化
2. 训练 Unrelated Branch 的 Topic 分类器，分类损失最大化
3. 固定前两步中两个分支的参数，在特征融合（此处为直接拼接）后，再进行 Fake News 判别器的训练

### 预训练语料

<img src="/Users/snow/Library/Application Support/typora-user-images/image-20200616002802103.png" style="zoom:40%" />

采用了上图中的 imbalanced 8 topics 与 balanced 3 topics 两种方式。

### 性能评估

![image-20200616002935141](/Users/snow/Library/Application Support/typora-user-images/image-20200616002935141.png)

可以看到：

- 经过Two Branches迁移后，模型的各项指标均有所上升
- 用平衡的3分类的预训练语料，能够取得更好的效果
- Related Branch、Unrelated Branch均有一定的迁移效果，且Unrelated Branch的迁移效果更好

### 迁移效果可视化

用tsne可视化方法，将各模型的源域（train）、目标域（test）的分布展示出来，可发现：

![image-20200616003208429](/Users/snow/Library/Application Support/typora-user-images/image-20200616003208429.png)

- EANN迁移过后，边缘分布没有对齐
- Unrelated Branch 迁移过后：1. 边缘分布对齐有适当提升；2. 目标域的类内分布更加紧凑（测试集数据，橙色点）

将各模型的测试集的决策面展示出来：

![image-20200616003348350](/Users/snow/Library/Application Support/typora-user-images/image-20200616003348350.png)

可发现：Unrelated Branch 迁移过后：测试集（目标域）的正负样本，类内分布都更加紧凑，因此分类效果有所提升。

## 关于跨主题迁移的思考

想要用 Topic 的迁移对 Fake News 的判别产生促进，究竟是要对 Topic “分得准” 还是 “分不准”？

- EANN、DANN：这两种模型是在 Fake News “判得准” 和 Topic“分得准” 之间做博弈

- Related Branch，Unrelated Branch：二者中 Unrelated Branch 略优，但Related Branch 仍有良好的效果

由此，似乎“分得准”和“分不准”都有一定的作用。

# Challenge 2: 情感信息挖掘

## 设计动机

Fake News的文本、图片，往往均具有强烈的煽动性，容易引发人们的情绪反应。

> [1] DEAN: Learning Dual Emotion for Fake News Detection on Social Media. arXiv 2019.
>
> [2] Exploiting Multi-domain Visual Information for Fake News Detection. ICDM 2019.

### 模型架构

<img src="/Users/snow/Library/Application Support/typora-user-images/image-20200616003935605.png" style="zoom:40%" />

设计情感引导的多模态融合模型，如上图所示。其能够动态地捕捉蕴含情感信息丰富，且情感信号有助于Fake News判别的模态。

### 文本模态的情感信息

#### 特征提取

![image-20200616004125699](/Users/snow/Library/Application Support/typora-user-images/image-20200616004125699.png)

共提取55维特征，如上表所示。

#### 情感特征分类

![image-20200616004228147](/Users/snow/Library/Application Support/typora-user-images/image-20200616004228147.png)

采用Random Forest模型进行分类，能够达到0.660的准确率，证明情感特征具有一定的可分性。

# 总结与收获

在本次课程设计中，主要在“跨主题迁移”上投入了很多精力，复现并尝试了Fake News及相关领域中，比较重要的两篇论文中的迁移方法，并对其迁移效果进行了评估，对表现出的性能进行了一定的分析。个人的论文阅读能力、代码实践能力、解决分析问题的能力，均得到了有效的锻炼！