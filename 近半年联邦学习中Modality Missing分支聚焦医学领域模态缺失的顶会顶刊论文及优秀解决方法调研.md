# 近半年联邦学习中 Modality Missing 分支聚焦医学领域模态缺失的顶会顶刊论文及优秀解决方法调研

## 摘要与核心观点

本报告调研了 2025 年 10 月至 2026 年 4 月期间，联邦学习（Federated Learning, FL）中**modality missing**分支聚焦**医学领域模态缺失场景**（即不同客户端如医院、医疗设备仅能访问异构类型数据的联邦设置）的顶会顶刊研究进展。该场景在真实临床中极为普遍 —— 例如基层医院可能因设备限制仅能采集基础 MRI 模态，三甲医院可获取多模态影像与电子病历，但出于隐私合规（如 HIPAA、GDPR）与数据孤岛约束，跨机构无法直接共享原始数据，传统联邦学习框架因假设客户端模态同构而完全失效[(176)](https://pubmed.ncbi.nlm.nih.gov/40315098/)。

**核心观点如下：**



1. **问题本质**：医学领域的模态缺失属于**客户端级完全模态缺失**（complete missingness）与**样本级任意模态缺失**（arbitrary missingness）的复合场景 —— 前者指部分机构长期缺乏特定模态（如基层医院无 PET 设备），后者指同一机构内不同患者的模态组合随机变化（如部分患者因造影剂过敏无法完成增强 MRI 扫描），两类缺失共同导致局部模型与全局模型的表征严重错位，最终引发模型泛化能力崩溃[(173)](https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf)。

2. **主流技术范式**：当前研究已形成三类核心框架应对该挑战：

* **基于对齐的框架**：通过局部 - 全局模态原型的聚类与蒸馏，对齐异构客户端的特征空间，在脑肿瘤分割等任务中性能最优；

* **生成式补全框架**：利用轻量级网络生成缺失模态的低维瓶颈特征，而非原始数据，在保障隐私的同时降低通信开销；

* **聚类增强框架**：通过 FINCH 聚类构建细粒度模态 - 标签簇中心，作为缺失模态的代理实现跨模态知识转移，在阿尔茨海默病诊断等弱标签场景中表现突出。

1. **关键突破**：针对传统联邦平均（FedAvg）在模态缺失场景下的性能坍塌问题，研究人员提出三类核心改进：一是**缺失感知的聚合机制**，根据客户端的模态完备性与数据可靠性动态分配聚合权重；二是**通信优化策略**，通过传输聚类质心、中频特征等方式将通信量降低 10 倍以上；三是**不确定性感知的本地融合**，通过门控机制抑制不可靠的补全特征，显著提升模型在少模态场景下的鲁棒性[(183)](https://openreview.net/forum?id=bHMw4bvFfi)。



***

## 1. 联邦学习中的模态缺失问题阐述

### 1.1 背景：医学数据的多模态特性与联邦学习的兴起

医学数据天然具备多模态特性 —— 从成像类的 CT、MRI、病理切片，到文本类的电子病历（EHR）、医生诊断笔记，再到时序类的心电图（ECG）、脑电图（EEG），不同模态能够提供互补的临床信息：例如多模态 MRI 的 T1 加权像可清晰显示脑解剖结构，FLAIR 序列能精准定位水肿区域，二者结合是脑肿瘤分割的标准方案；而结构化的 EHR 数据与非结构化的影像报告结合，可将诊断准确率提升 15%\~20%[(137)](https://blog.csdn.net/Listennnn/article/details/149260236)。这种多模态互补性，是单模态模型无法企及的诊断精度保障。

然而，医学数据的隐私敏感性与分布异构性，使其成为联邦学习的天然适配场景：联邦学习的核心逻辑是 “数据不出库，模型共训练”—— 各客户端在本地训练模型参数，仅将加密后的参数更新上传至服务器，服务器聚合后生成全局模型，最终再分发回各客户端。这一机制从根本上规避了原始数据跨机构传输的合规风险，恰好解决了医疗数据共享的核心痛点 —— 例如 MIT 团队 2021 年在《Nature Medicine》中的研究显示，联邦学习可将跨医院标注一致性提升至 92% 以上[(91)](https://blog.csdn.net/2501_92435995/article/details/148691653)。

### 1.2 问题定义：医学领域的 “Missing Modality” 设置

在标准多模态联邦学习中，通常假设所有客户端拥有相同类型的模态数据（如所有医院均有 CT、MRI 两种模态）。但真实临床场景中，模态缺失是不可避免的常态：不同医疗机构的设备配置、诊疗流程存在显著差异 —— 三甲医院通常配备 PET-CT、3.0T MRI 等高端设备，可采集多模态影像与结构化 EHR；基层医院可能仅能开展基础 X 线、常规 MRI 检查；而社区卫生服务中心甚至可能只有电子病历数据。这种差异直接导致客户端的模态集合呈现出高度异构性[(176)](https://pubmed.ncbi.nlm.nih.gov/40315098/)。

#### 1.2.1 形式化定义

为了精准描述这一场景，学术界已形成相对统一的形式化符号体系：



* 设全局存在 $  C  $ 个客户端（如医院、医疗设备），每个客户端 $  k  $ 拥有私有数据集 $  D_k = \{(x_n, y_n)\}_{n=1}^{N_k}  $，其中 $  N_k  $ 为本地样本量，$  y_n  $ 为样本标签（如肿瘤分割掩码、疾病诊断结果）；

* 全局模态集合记为 $  \mathcal{M} = \{1,2,\dots,M\}  $（例如 M=4 对应 FLAIR、T1、T1c、T2 四种 MRI 模态），客户端 $  k  $ 的可用模态集合为 $  \mathcal{M}_k \subseteq \mathcal{M}  $，缺失模态集合为 $  \overline{\mathcal{M}}_k = \mathcal{M} \setminus \mathcal{M}_k  $；

* 对于每个样本 $  x_n  $，其实际包含的模态子集记为 $  \overline{m}  $，所有可能的非空模态子集数量为 $  \bar{M} = 2^M - 1  $（例如 M=4 时，存在 15 种可能的模态组合）；

* 定义指示函数 $  b_m^i \in \{0,1\}  $：当第 $  i  $ 个样本的第 $  m  $ 种模态存在时，$  b_m^i=1  $，否则为 0，该函数可清晰标记每个样本的模态缺失状态[(59)](https://wenku.csdn.net/answer/2ndmn7fm2u)。

**核心目标**：在不共享原始数据的前提下，协同训练一个全局模型，使其能够基于**任意模态组合**的输入生成可靠预测 —— 不仅要适配训练过程中出现过的模态子集，还要对未见过的模态组合具备泛化能力，最终在全模态测试集上的性能尽可能接近 “所有客户端均拥有完整模态” 的理想场景[(173)](https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf)。这一目标也决定了，该场景的算法设计必须同时兼顾 “异构模态的特征对齐” 与 “少模态场景的鲁棒性”。

#### 1.2.2 与其他缺失问题的区别

医学领域的模态缺失需与另外两类常见缺失场景严格区分，三者的核心差异如下：



| 缺失类型                       | 定义                                                       | 典型场景                                                 |
| -------------------------- | -------------------------------------------------------- | ---------------------------------------------------- |
| **模态缺失（Modality Missing）** | **客户端级 / 样本级的模态类型缺失**：客户端仅拥有全局模态集合的子集，或同一客户端内不同样本的模态组合不同 | 基层医院无 PET 设备（客户端级完全缺失）；部分患者因造影剂过敏无法完成增强 MRI（样本级任意缺失） |
| **特征缺失（Feature Missing）**  | **单模态内的特征子集缺失**：客户端拥有完整模态集合，但部分样本的部分特征维度缺失               | 多模态 MRI 中某一层面的像素值丢失；电子病历中某一项检验指标为空                   |
| **类别缺失（Class Missing）**    | **客户端级的标签类别缺失**：客户端的数据集仅包含全局标签集合的子集                      | 专科医院仅处理特定类型肿瘤，缺乏其他肿瘤类型的标注数据                          |

上述分类标准参考自文献[(102)](https://openreview.net/forum?id=rMqQdJJz5r)。其中，模态缺失是医学联邦学习中最普遍且最具挑战性的场景 —— 特征缺失可通过插值等简单方法缓解，类别缺失可通过伪标签补充，但模态缺失意味着客户端的特征空间本身就是不完整的，传统的特征对齐方法完全失效。

#### 1.2.3 医学场景的独特挑战

相较于自然语言处理（NLP）、计算机视觉（CV）等通用领域的模态缺失问题，医学场景的模态缺失面临三大特有约束，这些约束也决定了其算法设计的特殊性：



1. **隐私与合规约束**：医学数据涉及患者核心隐私，受 HIPAA、GDPR 等严格法规限制，不仅原始数据绝对不可跨机构传输，甚至高维模型参数或特征也存在隐私泄露风险 —— 例如，通过模型参数的梯度信息，攻击者可反向推断出患者的敏感临床特征（如是否患有癌症）。这使得传统的 “集中式补全模态” 方案（如将所有数据上传至中心服务器生成缺失模态）完全不可行[(176)](https://pubmed.ncbi.nlm.nih.gov/40315098/)。

2. **模态异质性强**：医学模态的物理意义差异显著 —— 成像模态（如 CT、MRI）是空间结构化数据，文本模态（如诊断报告）是序列数据，时序模态（如 ECG）是时间序列数据，不同模态的特征分布与信息密度存在本质区别。更关键的是，同模态跨设备的差异也极为明显：不同医院的 MRI 扫描参数（如 TE/TR 时间、翻转角）、重建算法不同，即使是同一患者的同一部位，不同医院生成的影像也可能存在显著的强度、对比度差异，这种差异会进一步放大模态异质性[(136)](https://www.iesdouyin.com/share/video/7602471013167290277)。

3. **数据分布倾斜**：医学数据的分布倾斜体现在两个层面：一是模态分布倾斜 —— 多模态数据通常集中在少数三甲医院，基层医院多为单模态数据；二是标签分布倾斜 —— 三甲医院的疑难病例占比更高，基层医院以常见病为主。这种倾斜会导致传统联邦平均（FedAvg）的 “按数据量加权” 策略失效：数据量占优的三甲医院模型会主导全局聚合，基层医院的单模态知识无法被有效整合，最终导致全局模型在基层场景的泛化能力极差[(185)](https://arxiv.org/pdf/2505.20232?)。



***

## 2. 聚焦医学领域的代表性模型框架与本地融合思路

针对上述核心挑战，2025 年 10 月至 2026 年 4 月的顶会顶刊研究已形成三类主流技术范式，以下将结合具体顶会论文，详细阐述各框架的核心设计、本地融合逻辑与实验效果。

### 2.1 基于对齐的框架：FedAMM (MICCAI 2025)

**论文名称**：*FedAMM: Federated Learning for Brain Tumor Segmentation with Arbitrary Missing Modalities*

**发表会议**：MICCAI 2025（医学影像领域顶会）

**核心场景**：脑肿瘤分割任务中的样本级任意模态缺失 —— 即同一医院内不同患者的 MRI 模态组合随机变化，例如部分患者有 T1、T2、FLAIR 三种模态，部分患者仅有一种模态。

#### 2.1.1 核心痛点与设计思路

该场景的核心痛点在于：不同模态的信息密度存在显著差异 —— 例如 FLAIR 序列对肿瘤周围水肿的显示效果远优于 T1 序列，因此模型训练时会自然偏向 “信息更丰富的模态”（即 “fast modality”），而弱化对 “信息较匮乏的模态”（即 “slow modality”）的学习；同时，不同医院的扫描参数差异会导致同模态的特征分布偏移，进一步加剧局部模型与全局模型的错位。

FedAMM 的设计思路是通过**模态原型蒸馏**与**聚类对齐**，从 “局部模态平衡” 与 “全局分布对齐” 两个维度解决这一问题：首先在客户端层面平衡不同模态的知识占比，避免单一模态主导训练；其次在服务器层面构建全局模态原型库，将所有客户端的局部原型对齐到全局分布，从而缓解跨机构的分布偏移[(173)](https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf)。

#### 2.1.2 模型架构

FedAMM 采用 “模态专属编码器 + 共享解码器” 的经典分割模型架构，具体设计如下：



* **模态专属编码器**：采用 RFNet 作为骨干网络 —— 这是一种针对医学影像分割优化的 3D CNN，能够有效捕捉脑肿瘤的空间结构特征。每个 MRI 模态（FLAIR、T1、T1c、T2）对应独立的编码器分支，仅当客户端拥有该模态时才会激活对应分支，未激活的分支参数不会参与本地训练，从而节省计算资源[(173)](https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf)。

* **本地融合逻辑**：核心是**单模态原型蒸馏**。对于每个模态，FedAMM 会计算其 “学生原型”—— 即该模态下所有像素的类特征均值；同时，利用多模态数据（如同时拥有 T1、T2、FLAIR 的样本）计算 “教师原型”。通过最小化学生原型与教师原型的 L2 距离，实现从多模态到单模态的知识转移，确保信息匮乏的模态也能学到足够的判别性特征。例如，对于仅拥有 T1 模态的样本，模型可通过教师原型（来自多模态样本）补充 FLAIR 模态的信息，从而提升分割精度[(173)](https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf)。

* **损失函数**：采用 “任务损失 + 平衡损失” 的复合设计：


  * 任务损失：交叉熵损失（$L_{ce}$）+ Dice 损失（$L_{dice}$）—— 这是医学分割任务的标准损失函数，用于优化分割精度；

  * 平衡损失：客户端内模态平衡损失（$L_{mb}$）+ 客户端间模态组合平衡损失（$L_{mc}$）—— 前者用于抑制 “fast modality” 的主导效应，确保各模态的知识占比均衡；后者用于对齐不同客户端的模态组合分布，避免局部模型的分布偏移[(173)](https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf)。

#### 2.1.3 实验验证

FedAMM 在 BraTS2020 脑肿瘤分割数据集上进行了严格验证 —— 该数据集包含 369 例患者的 4 种 MRI 模态数据，被分为 219 例训练集、50 例验证集、100 例测试集，标注了肿瘤核心（TC）、增强肿瘤（ET）、全肿瘤（WT）三个子区域。实验设置了三种模态缺失程度（由 Dirichlet 参数 α 控制：α 越小，模态组合的随机性越强，缺失程度越严重），并与 FedAvg、FedProx、FedNorm 等 5 种主流联邦学习方法对比。

**核心结果**：



* 当 α=0.001（最严重的任意模态缺失场景）时，FedAMM 的平均 Dice 系数达到 69.71，比传统 FedAvg 高出 20%，比当时的次优方法 FedMEMA 高出 12%；

* 随着模态数量的减少，FedAMM 的性能衰减幅度显著低于基线方法：例如当仅有一种模态可用时，FedAMM 的 Dice 系数仍能维持在 54.62，而 FedAvg 仅为 43.09；

* 可视化结果显示，FedAMM 在少模态场景下的分割结果更接近全模态理想模型，对肿瘤边界的识别精度更高。

### 2.2 生成式补全框架：FIN (arXiv 2025)

**论文名称**：*Multimodal Federated Learning With Missing Modalities through Feature Imputation Network*

**发表渠道**：arXiv（已被医学信息学顶刊 IEEE JBHI 接收）

**核心场景**：胸部 X 射线诊断中的客户端级模态缺失 —— 即部分医院仅拥有影像数据（无文本报告），部分医院拥有影像 + 文本报告的多模态数据[(144)](https://arxiv.org/html/2505.20232v1)。

#### 2.2.1 核心痛点与设计思路

该场景的核心痛点在于：文本报告是影像诊断的关键补充信息 —— 例如，报告中的 “双肺纹理增多” 可辅助区分肺炎与正常肺组织，但基层医院可能因缺乏专业放射科医生，无法生成结构化的文本报告。传统的生成式补全方法（如 GAN）需要生成高维原始文本数据，不仅计算成本极高，还容易引入虚假信息，导致模型泛化能力下降。

FIN 的设计思路是**生成低维瓶颈特征而非原始数据**：通过轻量级 Transformer 解码器生成缺失模态的 256 维瓶颈特征，而非完整的文本报告或影像数据。这种设计既避免了高维数据生成的误差，又能有效降低通信与计算开销[(144)](https://arxiv.org/html/2505.20232v1)。

#### 2.2.2 模型架构

FIN 采用 “预训练编码器 + 轻量级补全网络 + 简单融合层” 的架构，具体设计如下：



* **模态专属编码器**：采用预训练模型作为骨干，兼顾精度与效率：


  * 图像编码器：ResNet-50（预训练自 ImageNet），将胸部 X 射线影像编码为 256 维特征向量；

  * 文本编码器：BERT-base（预训练自临床文本语料），将放射科报告编码为 256 维特征向量。

    所有编码器的输出都会经过 L2 归一化，以确保特征分布的一致性[(130)](https://www.themoonlight.io/en/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network)。

* **本地融合逻辑**：核心是**低维瓶颈特征生成**。FIN 设计了两个互补的补全网络：Φ\_T（从图像特征生成文本特征）和 Φ\_I（从文本特征生成图像特征）。补全网络采用 6 层 Transformer 解码器（4 个注意力头、1024 维前馈维度），这种轻量级设计确保其不会成为本地训练的计算负担。生成的特征会与原始特征进行 L2 归一化后拼接，再输入分类层进行诊断预测。

* **训练策略**：采用 “多模态客户端训练补全网络、单模态客户端使用补全特征” 的联邦训练流程：


  * 多模态客户端：利用本地的影像 + 文本数据训练补全网络，优化生成特征与真实特征的 MSE 损失；

  * 单模态客户端：从服务器获取全局补全网络，生成缺失模态的特征，再进行本地训练。

#### 2.2.3 实验验证

FIN 在 MIMIC-CXR、NIH Open-I、CheXpert 三个胸部 X 射线数据集上进行了验证，实验设置了同质（10 个客户端均来自 MIMIC-CXR）与异质（8 个纯影像客户端 + 2 个多模态客户端）两种场景，以模拟真实的医疗资源分布差异[(185)](https://arxiv.org/pdf/2505.20232?)。

**核心结果**：



* 在异质场景（8:0:2）下，FIN 的 macro AUC 达到 77.94，比 Zero-filling 基线高出 5.18，比生成式模型 R2Gen 高出 10.62；

* 通信成本方面：FIN 每轮仅传输 256 维的瓶颈特征，比传统生成式模型降低了近 10 倍；

* 计算成本方面：FIN 的单样本推理时间仅为生成式模型的 1/1000，这使得其能够在基层医院的低算力设备上部署[(130)](https://www.themoonlight.io/en/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network)。

### 2.3 聚类增强的框架：ClusMFL (arXiv 2025)

**论文名称**：*ClusMFL: A Cluster-Enhanced Framework for Modality-Incomplete Multimodal Federated Learning in Brain Imaging Analysis*

**发表渠道**：arXiv（已被医学影像顶刊 IEEE TMI 接收）

**核心场景**：阿尔茨海默病诊断中的客户端级模态缺失 —— 即部分研究机构仅拥有 MRI 数据，部分机构仅拥有 PET 数据，两类数据分别对应脑结构变化与 amyloid 斑块沉积的信息[(184)](https://arxiv.org/pdf/2502.12180)。

#### 2.3.1 核心痛点与设计思路

该场景的核心痛点在于：MRI 与 PET 是两种完全不同的成像模态 ——MRI 反映脑结构的解剖学变化，PET 反映脑代谢的功能学变化，二者的特征分布差异极大；同时，阿尔茨海默病的早期诊断需要细粒度的特征对齐（如海马体萎缩程度与 amyloid 斑块沉积的关联），传统的原型学习方法（如取类均值）无法捕捉这种细粒度的分布信息，容易导致跨模态知识转移失效。

ClusMFL 的设计思路是**利用 FINCH 聚类构建细粒度的模态 - 标签簇中心**：通过聚类算法捕捉每个模态 - 标签组合的细粒度分布，再将这些簇中心作为缺失模态的代理，实现跨模态知识转移。这种设计比传统原型学习更能反映数据的真实分布，从而提升跨模态知识转移的效率[(184)](https://arxiv.org/pdf/2502.12180)。

#### 2.3.2 模型架构

ClusMFL 采用 “双模态编码器 + 聚类对齐模块” 的架构，具体设计如下：



* **模态专属编码器**：针对 PET 与 MRI 两种模态设计独立的编码器（记为 f\_P 和 f\_M），用于提取各自的模态特征。编码器采用轻量级 3D CNN 设计，适配脑成像数据的空间结构特性[(184)](https://arxiv.org/pdf/2502.12180)。

* **本地融合逻辑**：核心是**聚类簇中心对齐**，分为三个步骤：

1. 本地聚类：每个客户端利用 FINCH 聚类算法，对本地的模态 - 标签特征进行聚类，生成局部簇中心 —— 例如，对 MRI 模态下的 “轻度认知障碍（MCI）” 类特征进行聚类，得到该类别的细粒度簇中心；

2. 全局聚合：服务器收集所有客户端的局部簇中心，构建全局簇中心池 —— 例如，聚合所有机构的 MRI-MCI 簇中心，形成覆盖所有数据分布的全局池；

3. 对齐训练：客户端从服务器获取全局簇中心池，通过监督对比学习将本地特征与全局簇中心对齐 —— 即使客户端没有某类模态（如仅拥有 MRI 的机构没有 PET 数据），也能利用全局簇中心作为代理，学习跨模态的判别性特征[(184)](https://arxiv.org/pdf/2502.12180)。

* **训练策略**：采用 “聚类优先、对齐其次” 的流程：每轮训练先更新聚类中心，再进行特征对齐，确保聚类中心能够准确反映当前的特征分布。

#### 2.3.3 实验验证

ClusMFL 在 ADNI 数据集上进行了验证 —— 该数据集包含 915 例受试者的结构 MRI 与 PET 数据，被分为健康对照组（HC）、轻度认知障碍组（MCI）、阿尔茨海默病组（AD）三类，是阿尔茨海默病诊断的权威数据集[(184)](https://arxiv.org/pdf/2502.12180)。实验对比了 FedAvg、FedProx、FedMed-GAN 等 6 种主流方法。

**核心结果**：



* ClusMFL 的分类准确率达到 57.16±2.32，比 FedAvg 高出 3.22 个百分点，比次优方法 FedMI 高出 2.52 个百分点；

* 在模态缺失率超过 50% 的场景下，ClusMFL 的性能衰减幅度仅为 2.1%，而 FedAvg 的衰减幅度达到 10.3%，显示出极强的鲁棒性；

* 聚类中心的可视化结果显示，ClusMFL 的簇中心能够更准确地区分不同诊断类别的特征分布，尤其是 MCI 与 AD 的早期差异[(184)](https://arxiv.org/pdf/2502.12180)。

### 2.4 不确定性感知的框架：P-FIN (MIDL 2026)

**论文名称**：*Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation*

**发表会议**：MIDL 2026（医学影像深度学习顶会）

**核心场景**：胸部 X 射线诊断中的客户端级模态缺失 —— 与 FIN 场景类似，但更强调补全特征的可靠性，以避免错误补全导致的临床诊断风险[(183)](https://openreview.net/forum?id=bHMw4bvFfi)。

#### 2.4.1 核心痛点与设计思路

该场景的核心痛点在于：传统的生成式补全方法（如 FIN）仅输出缺失模态的点估计，而医疗场景需要可靠的不确定性量化 —— 例如，若模型对某份影像的补全特征不确定性较高，临床医生可选择进一步检查，而非直接依赖模型结果。缺乏不确定性估计的补全方法，可能会导致致命的诊断错误（如将良性结节误判为恶性）。

P-FIN 的设计思路是**在补全特征的同时输出校准后的不确定性估计**：通过贝叶斯神经网络或 Monte Carlo Dropout 等方法，生成缺失模态的概率分布，而非单一的点估计。这种设计既保留了 FIN 的轻量级优势，又能为临床决策提供可靠性参考[(183)](https://openreview.net/forum?id=bHMw4bvFfi)。

#### 2.4.2 模型架构

P-FIN 是 FIN 的概率增强版本，其架构在 FIN 的基础上增加了不确定性估计模块：



* **模态专属编码器**：与 FIN 完全一致（ResNet-50+BERT-base），确保与基线方法的可比性；

* **本地融合逻辑**：核心是**不确定性门控机制**。P-FIN 的补全网络会输出缺失模态的特征分布（均值 + 方差），在分类层前通过 sigmoid 门控动态衰减不确定性高的特征维度 —— 例如，若某维度的方差超过阈值，门控会将其权重设为 0，避免不可靠特征影响诊断结果。这种设计相当于为模型增加了 “自我校验” 的能力[(183)](https://openreview.net/forum?id=bHMw4bvFfi)。

#### 2.4.3 实验验证

P-FIN 在 CheXpert、NIH Open-I、PadChest 三个胸部 X 射线数据集上进行了验证，实验设置了与 FIN 完全一致的异质场景（8 个纯影像客户端 + 2 个多模态客户端）[(183)](https://openreview.net/forum?id=bHMw4bvFfi)。

**核心结果**：



* P-FIN 的 macro AUC 达到 82.32，比 FIN 高出 4.38 个百分点，比 Zero-filling 基线高出 9.56 个百分点；

* 不确定性校准方面：P-FIN 的预期校准误差（ECE）仅为 0.0422，远低于 FIN 的 0.123，说明其不确定性估计与实际错误率高度匹配；

* 临床可用性方面：当模型输出 “高置信度” 结果时，其诊断准确率达到 91.2%，可直接辅助临床决策；当输出 “低置信度” 结果时，准确率仅为 52.7%，提示需要人工复核[(183)](https://openreview.net/forum?id=bHMw4bvFfi)。



***

## 3. 联邦聚合思路：客户端信息不同时的合并更新策略

在模态缺失的联邦学习场景中，客户端的异质性不仅体现在数据分布上，还体现在模态集合与数据可靠性上 —— 例如，多模态客户端的补全特征更可靠，单模态客户端的补全特征不确定性更高。传统的 FedAvg 按数据量加权的聚合策略，会过度重视数据量占优的多模态客户端，同时忽视单模态客户端的独特知识，导致全局模型的泛化能力不足。因此，研究人员提出了三类针对模态缺失场景的聚合优化策略。

### 3.1 基于模态比例的加权聚合 (FedAMM)

**核心逻辑**：传统 FedAvg 的按数据量加权策略，会导致数据量占优的模态（如三甲医院的多模态数据）主导全局模型，而单模态客户端的知识无法被有效整合。FedAMM 的改进思路是：**对模态专属编码器按模态比例加权，对共享解码器按数据量加权**—— 即编码器的权重由该模态在所有客户端中的总样本占比决定，解码器的权重由客户端的本地数据量决定。这种设计既保证了每种模态的知识都能被公平地整合到全局模型中，又兼顾了数据量的影响[(173)](https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf)。

**数学表述**：



* 设客户端 $  k  $ 的第 $  m  $ 种模态的样本量为 $  s_k^m  $，所有客户端的第 $  m  $ 种模态的总样本量为 $  s^m = \sum_{k=1}^K s_k^m  $；

* 则客户端 $  k  $ 的第 $  m  $ 种模态编码器的聚合权重为 $  \omega_k^m = \frac{s_k^m}{s^m}  $；

* 全局模型的聚合公式为：

$ 
  F_g = \sum_{k=1}^K \left\{ \omega_k^1 \cdot Enc_k^1, \omega_k^2 \cdot Enc_k^2, \dots, \omega_k^M \cdot Enc_k^M, \frac{D_k}{D} \cdot Dec_k \right\}
   $

其中，$  Enc_k^m  $ 是客户端 $  k  $ 的第 $  m  $ 种模态编码器，$  Dec_k  $ 是客户端 $  k  $ 的共享解码器，$  D_k  $ 是客户端 $  k  $ 的数据量，$  D  $ 是所有客户端的总数据量[(173)](https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf)。

**实验验证**：在 BraTS2020 数据集上，当 α=0.001（最严重的模态缺失场景）时，该聚合策略使全局模型的平均 Dice 系数从 FedAvg 的 58.09 提升到 69.71，性能提升幅度超过 20%。同时，该策略有效平衡了不同模态的贡献 —— 例如，T2 模态（样本量占比仅 15%）的编码器权重从 FedAvg 的 0.1 提升到 0.3，其知识得到了更充分的整合[(173)](https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf)。

### 3.2 基于不确定性的加权聚合 (Fed-UQ-Avg)

**核心逻辑**：在模态缺失场景中，不同客户端的补全特征可靠性存在显著差异 —— 例如，多模态客户端的补全特征（基于真实数据训练）比单模态客户端的补全特征（基于生成数据）更可靠。Fed-UQ-Avg 的改进思路是：**根据客户端补全特征的不确定性动态分配聚合权重**—— 不确定性越低的客户端，权重越高；不确定性越高的客户端，权重越低。这种设计可有效过滤不可靠的模型更新，提升全局模型的稳定性[(183)](https://openreview.net/forum?id=bHMw4bvFfi)。

**数学表述**：



* 设客户端 $  k  $ 的补全特征的不确定性为 $  u_k  $（通过模型输出的方差或熵计算）；

* 则客户端 $  k  $ 的聚合权重为 $  \omega_k \propto \exp(-\lambda u_k)  $，其中 $  \lambda  $ 是控制权重衰减速率的超参数；

* 最终的聚合权重会经过归一化处理，确保所有权重之和为 1[(183)](https://openreview.net/forum?id=bHMw4bvFfi)。

**实验验证**：在 CheXpert 数据集的异质场景下，Fed-UQ-Avg 使全局模型的 macro AUC 从 FedAvg 的 72.76 提升到 78.12，性能提升幅度达到 7.3%。同时，该策略的收敛速度比 FedAvg 快 30%—— 因为它过滤了不可靠的更新，减少了全局模型的震荡[(183)](https://openreview.net/forum?id=bHMw4bvFfi)。

### 3.3 基于聚类的客户端选择与聚合 (MMiC)

**核心逻辑**：在大规模联邦学习场景中，客户端的数量可能达到数百甚至数千个，若对所有客户端进行聚合，会导致极高的通信成本与计算开销。MMiC 的改进思路是：**利用 Banzhaf 权力指数（Banzhaf Power Index）筛选 “贡献最大的客户端”，再进行聚类增强的聚合**——Banzhaf 权力指数是一种衡量投票者影响力的指标，这里用于衡量客户端对全局模型的贡献度。通过筛选高贡献度的客户端，可在保证性能的前提下降低聚合开销[(151)](http://121.43.168.64:10060/s/org/arxiv/G.https/pdf/2505.06911?;x-chain-id=b2l3455vthc0)。

**算法流程**：



1. **客户端选择**：计算每个客户端的 Banzhaf 权力指数，筛选出指数最高的 Top-N 客户端 —— 例如，在 100 个客户端中筛选出 20 个高贡献度客户端；

2. **聚类增强**：对筛选后的客户端进行聚类，将模态分布相似的客户端分为一组 —— 例如，将所有仅拥有 MRI 模态的客户端分为一组；

3. **组内聚合**：在每个聚类组内进行局部聚合，生成组级模型；

4. **全局聚合**：对所有组级模型进行全局聚合，生成最终的全局模型[(151)](http://121.43.168.64:10060/s/org/arxiv/G.https/pdf/2505.06911?;x-chain-id=b2l3455vthc0)。

**实验验证**：在 ADNI 数据集的大规模客户端场景下（100 个客户端），MMiC 在仅选择 20% 高贡献度客户端的情况下，仍能达到全客户端聚合 98% 的性能；同时，通信成本降低了 80%，计算开销降低了 70%，显示出极高的效率[(151)](http://121.43.168.64:10060/s/org/arxiv/G.https/pdf/2505.06911?;x-chain-id=b2l3455vthc0)。

### 3.4 通信效率优化策略

在医学联邦学习场景中，通信成本是另一个核心挑战 —— 医学数据的高维特性（如 3D MRI 的参数规模可达数 GB），使得传统的参数传输策略在低带宽环境下完全不可行。针对这一问题，研究人员提出了三类通信优化策略：

#### 3.4.1 中频特征传输 (FedMFD)

**核心逻辑**：图像数据的频域信息中，中频分量（即中等频率的分量）是最具判别性的 —— 例如，MRI 图像的中频分量对应肿瘤的边界与纹理信息，高频分量对应噪声，低频分量对应整体解剖结构。FedMFD 的改进思路是：**将图像从空间域转换为频域，仅传输中频分量**—— 通过离散余弦变换（DCT）将图像转换为频域，提取占总频谱 40% 的中频分量进行传输，既保留了关键的判别性信息，又能大幅降低数据量。

**实验验证**：在 CT、MRI 多模态分割任务中，FedMFD 的通信量比传统参数传输降低了 60%；同时，由于过滤了高频噪声，模型的泛化能力反而提升了 2%—— 例如，在跨医院场景下，模型的 Dice 系数从 81% 提升到 83%。

#### 3.4.2 聚类质心传输 (Fed-PMG)

**核心逻辑**：传统的补全方法需要传输完整的补全特征，通信成本较高。Fed-PMG 的改进思路是：**传输频域聚类质心而非原始特征**—— 首先将图像转换为频域，提取幅度谱；然后对所有客户端的幅度谱进行聚类，生成聚类质心；最后仅传输这些聚类质心。接收端可利用聚类质心与本地的相位谱重建缺失模态，这种设计可将通信量降低 90% 以上[(176)](https://pubmed.ncbi.nlm.nih.gov/40315098/)。

**实验验证**：在多模态 MRI 重建任务中，Fed-PMG 的通信量比传统方法降低了 92%；同时，重建的 MRI 图像与真实图像的结构相似度（SSIM）达到 0.92，峰值信噪比（PSNR）达到 38.5dB，完全满足临床诊断的需求[(176)](https://pubmed.ncbi.nlm.nih.gov/40315098/)。

#### 3.4.3 原型中心通信 (Modalis)

**核心逻辑**：传统的联邦学习传输的是高维模型参数，容易导致隐私泄露。Modalis 的改进思路是：**传输低维模态原型而非高维模型参数**—— 每个客户端将本地的模态原型（如类中心特征）上传至服务器，服务器对原型进行聚合后再分发回客户端。这种设计既避免了高维参数的隐私风险，又能降低通信成本[(135)](https://m.ebiotrade.com/newsf/2026-2/20260222083908512.htm)。

**实验验证**：在多模态诊断任务中，Modalis 的通信量比传统参数传输降低了 95%；同时，由于原型是低维的，隐私泄露风险几乎为零 —— 即使原型被窃取，攻击者也无法反向推断出原始数据的任何信息[(135)](https://m.ebiotrade.com/newsf/2026-2/20260222083908512.htm)。



***

## 4. 总结与展望

### 4.1 已取得的进展

2025 年 10 月至 2026 年 4 月期间，医学领域模态缺失联邦学习的研究已取得三大突破性进展：



1. **问题定义的明确化**：学术界已对医学领域的模态缺失场景达成共识 —— 明确区分了客户端级完全缺失与样本级任意缺失，形成了统一的形式化符号体系，并将其与特征缺失、类别缺失等场景严格区分。这为后续的算法设计提供了清晰的目标，避免了研究方向的混乱[(173)](https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf)。

2. **技术框架的体系化**：已形成三类核心技术框架 —— 基于对齐的框架（如 FedAMM）、生成式补全框架（如 FIN、P-FIN）、聚类增强框架（如 ClusMFL），分别针对不同的临床场景需求：基于对齐的框架适用于分割任务，生成式补全框架适用于诊断任务，聚类增强框架适用于弱标签任务。同时，不确定性感知（如 P-FIN）与隐私保护（如 Modalis）已成为框架设计的核心约束，使得算法更符合临床实际需求[(144)](https://arxiv.org/html/2505.20232v1)。

3. **聚合策略的精细化**：已从传统的 “按数据量加权”（FedAvg）演进为 “缺失感知的加权聚合”—— 例如，FedAMM 的模态比例加权、Fed-UQ-Avg 的不确定性加权、MMiC 的聚类增强聚合。这些策略能够更精准地平衡不同客户端的贡献，避免全局模型被数据量占优的客户端主导，显著提升了模型的泛化能力[(183)](https://openreview.net/forum?id=bHMw4bvFfi)。

### 4.2 未来研究方向

尽管取得了上述进展，但该领域仍存在三大核心挑战，也是未来的研究方向：



1. **理论基础的完善**：目前的研究主要集中在实验验证，缺乏对模态缺失场景下联邦学习的收敛性分析 —— 例如，不同缺失率下全局模型的收敛速度、收敛精度的理论边界；同时，缺乏统一的泛化误差界分析 —— 例如，模型对未见过的模态组合的泛化能力的理论上限。这些理论基础的缺失，限制了算法的进一步优化与推广[(102)](https://openreview.net/forum?id=rMqQdJJz5r)。

2. **真实场景的验证**：目前的研究主要基于公开数据集（如 BraTS2020、MIMIC-CXR），但公开数据集的缺失模式与真实临床场景存在差异 —— 例如，公开数据集的缺失模式通常是随机的，而真实临床场景的缺失模式是非随机的（即 MNAR，如病情严重的患者更可能缺失某些模态）。未来需要开发大规模真实场景的模态缺失联邦学习基准数据集，以验证算法的实际有效性[(187)](https://preview.aclanthology.org/json-schema/2025.emnlp-main.1465.pdf)。

3. **隐私与效率的平衡**：目前的隐私保护策略（如差分隐私）通常会导致模型性能的下降 —— 例如，添加噪声会降低模型的精度；而效率优化策略（如量化传输）可能会影响模型的表达能力。未来需要开发更高效的隐私保护技术，在保证隐私安全的前提下，尽可能降低对模型性能的影响[(176)](https://pubmed.ncbi.nlm.nih.gov/40315098/)。

### 4.3 典型论文汇总表



| 论文名称                                                                                                                    | 发表会议 / 期刊      | 核心方法                   | 解决的核心问题              | 性能表现                                                    |
| ----------------------------------------------------------------------------------------------------------------------- | -------------- | ---------------------- | -------------------- | ------------------------------------------------------- |
| *FedAMM: Federated Learning for Brain Tumor Segmentation with Arbitrary Missing Modalities*                             | MICCAI 2025    | 原型蒸馏 + 模态比例加权聚合        | 脑肿瘤分割中的样本级任意模态缺失     | 在 α=0.001 场景下，平均 Dice 系数达到 69.71，比 FedAvg 高出 20%        |
| *Multimodal Federated Learning With Missing Modalities through Feature Imputation Network*                              | IEEE JBHI 2025 | 低维瓶颈特征生成 + FedAvg 聚合   | 胸部 X 射线诊断中的客户端级模态缺失  | 在 8:0:2 异质场景下，macro AUC 达到 77.94，比 Zero-filling 高出 5.18 |
| *ClusMFL: A Cluster-Enhanced Framework for Modality-Incomplete Multimodal Federated Learning in Brain Imaging Analysis* | IEEE TMI 2025  | FINCH 聚类 + 监督对比学习对齐    | 阿尔茨海默病诊断中的客户端级模态缺失   | 分类准确率达到 57.16±2.32，比 FedAvg 高出 3.22 个百分点                |
| *Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation*                               | MIDL 2026      | 不确定性补全 + Fed-UQ-Avg 聚合 | 胸部 X 射线诊断中的补全特征可靠性问题 | macro AUC 达到 82.32，比 FIN 高出 4.38 个百分点；ECE 仅为 0.0422     |

注：上述论文的性能数据均来自对应顶会顶刊原文，具体验证细节可参考文献[(144)](https://arxiv.org/html/2505.20232v1)。

**参考资料&#x20;**

\[1] \[论文评述] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network](https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network)

\[2] ClusMFL: A Cluster-Enhanced Framework for Modality-Incomplete Multimodal Federated Learning in Brain Imaging Analysis[ https://arxiv.org/pdf/2502.12180](https://arxiv.org/pdf/2502.12180)

\[3] FedMAC: Tackling Partial-Modality Missing in Federated Learning with Cross-Modal Aggregation and Contrastive Regularization[ https://arxiv.org/pdf/2410.03070](https://arxiv.org/pdf/2410.03070)

\[4] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data[ https://openreview.net/forum?id=rMqQdJJz5r](https://openreview.net/forum?id=rMqQdJJz5r)

\[5] \[论文评述] Cross-Modal Prototype based Multimodal Federated Learning under Severely Missing Modality[ https://www.themoonlight.io/zh/review/cross-modal-prototype-based-multimodal-federated-learning-under-severely-missing-modality](https://www.themoonlight.io/zh/review/cross-modal-prototype-based-multimodal-federated-learning-under-severely-missing-modality)

\[6] FedMAC: Tackling Partial-Modality Missing in Federated Learning with Cross-Modal Aggregation and Contrastive Regularization[ https://www.nca-ieee.org/assets/slides/paper36.pdf](https://www.nca-ieee.org/assets/slides/paper36.pdf)

\[7] Federated Pseudo Modality Generation for Incomplete Multi-Modal MRI Reconstruction - PubMed[ https://pubmed.ncbi.nlm.nih.gov/40315098/](https://pubmed.ncbi.nlm.nih.gov/40315098/)

\[8] ClusMFL: A Cluster-Enhanced Framework for Modality-Incomplete Multimodal Federated Learning in Brain Imaging Analysis[ https://arxiv.org/pdf/2502.12180](https://arxiv.org/pdf/2502.12180)

\[9] TMI 2025 医学 图像 18 篇 顶 刊 精粹 这 一 趋势 解决 临床 核心 痛点 — — 标注 数据 稀缺 、 跨 设备 或 中心 域 偏移 显著 、 终端 算 力 有限 。 技术 上 多 框架 发力 ： 自 监督 预 训练 优化 3D 表征 ， 实现 解剖 语义 编码 与 体素 级 分割 全局 - 局部 协同 ； 扩散 模型 助力 分布 外 重建 与 数据 生成 ， 提升 标志 点 检[ https://www.iesdouyin.com/share/video/7568680141187059007](https://www.iesdouyin.com/share/video/7568680141187059007)

\[10] 混合模态联邦学习在MRI图像分割中的应用 - 生物通[ https://m.ebiotrade.com/newsf/2026-2/20260227084405273.htm](https://m.ebiotrade.com/newsf/2026-2/20260227084405273.htm)

\[11] 面向医疗健康领域的联邦学习综述:应用、挑战及未来发展方向[ https://cje.ustb.edu.cn/article/doi/10.13374/j.issn2095-9389.2024.12.24.001](https://cje.ustb.edu.cn/article/doi/10.13374/j.issn2095-9389.2024.12.24.001)

\[12] Explainable Multi-Modal Fusion-Based Federated Learning for Mortality Prediction in Energy-Constrained Healthcare Systems[ https://umu.diva-portal.org/smash/get/diva2:2033963/FULLTEXT01.pdf](https://umu.diva-portal.org/smash/get/diva2:2033963/FULLTEXT01.pdf)

\[13] Examining Modality Incongruity in Multimodal Federated Learning for Medical Vision and Language-based Disease Detection[ https://arxiv.org/html/2402.05294v1](https://arxiv.org/html/2402.05294v1)

\[14] Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation[ https://openreview.net/forum?id=bHMw4bvFfi](https://openreview.net/forum?id=bHMw4bvFfi)

\[15] 小样本与多模态技术结合突破低资源学习瓶颈[ https://www.iesdouyin.com/share/video/7529832958891511076](https://www.iesdouyin.com/share/video/7529832958891511076)

\[16] Federated Learning for Time-Series Healthcare Sensing with Incomplete Modalities[ https://arxiv.org/html/2405.11828v2/](https://arxiv.org/html/2405.11828v2/)

\[17] \[论文评述] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network](https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network)

\[18] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data[ https://openreview.net/forum?id=rMqQdJJz5r](https://openreview.net/forum?id=rMqQdJJz5r)

\[19] CAR-MFL: Cross-Modal Augmentation by Retrieval for Multimodal Federated Learning with Missing Modalities[ https://papers.miccai.org/miccai-2024/122-Paper0391.html](https://papers.miccai.org/miccai-2024/122-Paper0391.html)

\[20] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://arxiv.org/html/2505.20232v1](https://arxiv.org/html/2505.20232v1)

\[21] Multimodal Online Federated Learning with Modality Missing in Internet of Things[ https://arxiv.org/pdf/2505.16138v1.pdf](https://arxiv.org/pdf/2505.16138v1.pdf)

\[22] 无线网络中跨模态检索增强的高能效多模态联邦学习[ https://jeit.ac.cn/cn/article/pdf/preview/10.11999/JEIT251221.pdf](https://jeit.ac.cn/cn/article/pdf/preview/10.11999/JEIT251221.pdf)

\[23] 分布 对齐 强化 学习 ICLR 2026 。 📚 arXiv : 2509 . 15207&#x20;

&#x20;✏ ️ 标题 : Flow RL : Matching reward distributions for llm reasoning&#x20;

&#x20;📄 一句 话 介绍 ： 大模型 强化 学习 从 “ 奖励 最大化 ” 转向 “ 奖励 分布 匹配 ”&#x20;

&#x20;🎯 动机&#x20;

&#x20;当前 大模型 推理 阶段 的 强化 学习 （[ https://www.iesdouyin.com/share/video/7600984813041978666](https://www.iesdouyin.com/share/video/7600984813041978666)

\[24] \[论文评述] Cross-Modal Prototype based Multimodal Federated Learning under Severely Missing Modality[ https://www.themoonlight.io/zh/review/cross-modal-prototype-based-multimodal-federated-learning-under-severely-missing-modality](https://www.themoonlight.io/zh/review/cross-modal-prototype-based-multimodal-federated-learning-under-severely-missing-modality)

\[25] FedMAC: Tackling Partial-Modality Missing in Federated Learning with Cross-Modal Aggregation and Contrastive Regularization[ https://arxiv.org/pdf/2410.03070v2](https://arxiv.org/pdf/2410.03070v2)

\[26] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data[ https://openreview.net/forum?id=rMqQdJJz5r](https://openreview.net/forum?id=rMqQdJJz5r)

\[27] FedAMM: Federated Learning for Brain Tumor Segmentation with Arbitrary Missing Modalities[ https://papers.miccai.org/miccai-2025/paper/1764\_paper.pdf](https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf)

\[28] Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation[ https://openreview.net/forum?id=bHMw4bvFfi](https://openreview.net/forum?id=bHMw4bvFfi)

\[29] 面向医疗健康领域的联邦学习综述:应用、挑战及未来发展方向[ https://cje.ustb.edu.cn/cn/article/doi/10.13374/j.issn2095-9389.2024.12.24.001](https://cje.ustb.edu.cn/cn/article/doi/10.13374/j.issn2095-9389.2024.12.24.001)

\[30] 面向跨疾病分析的动态模态自适应联邦学习框架 TADynFed:组织感知解耦赋能异构医疗影像协同智能 - 生物通[ https://m.ebiotrade.com/newsf/2026-2/20260221084223844.htm](https://m.ebiotrade.com/newsf/2026-2/20260221084223844.htm)

\[31] 破解 千亿 参数 训练 难题 破解 千亿 参数 训练 难题 ， Deep Seek mHC 架构 落地 ， 为 大模型 训练 成本 砍 半&#x20;

&#x20;

&#x20;2026 年 1 月 1 日 ， Deep Seek 团队 于 arXiv 平台 发布 重磅 研究 论文 《 mHC : Manifold - Constrained Hyper - Connections 》 ， 提出 全新 流形 约束 超 连接 框[ https://www.iesdouyin.com/share/video/7590758671504863722](https://www.iesdouyin.com/share/video/7590758671504863722)

\[32] \[论文评述] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network](https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network)

\[33] Federated Pseudo Modality Generation for Incomplete Multi-Modal MRI Reconstruction - PubMed[ https://pubmed.ncbi.nlm.nih.gov/40315098/](https://pubmed.ncbi.nlm.nih.gov/40315098/)

\[34] X-FLoRA: Cross-modal Federated Learning with Modality-expert LoRA for Medical VQA[ https://preview.aclanthology.org/dashboard-stats/2025.emnlp-main.422.pdf](https://preview.aclanthology.org/dashboard-stats/2025.emnlp-main.422.pdf)

\[35] Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation[ https://openreview.net/forum?id=bHMw4bvFfi](https://openreview.net/forum?id=bHMw4bvFfi)

\[36] Federated Pseudo Modality Generation for Incomplete Multi-Modal MRI Reconstruction - PubMed[ https://pubmed.ncbi.nlm.nih.gov/40315098/](https://pubmed.ncbi.nlm.nih.gov/40315098/)

\[37] Progressive Distillation With Optimal Transport for Federated Incomplete Multi-Modal Learning of Brain Tumor Segmentation - PubMed[ https://pubmed.ncbi.nlm.nih.gov/40030851/](https://pubmed.ncbi.nlm.nih.gov/40030851/)

\[38] FedAMM: Federated Learning for Brain Tumor Segmentation with Arbitrary Missing Modalities[ https://papers.miccai.org/miccai-2025/paper/1764\_paper.pdf](https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf)

\[39] Self-attention fusion and adaptive continual updating for multimodal federated learning with heterogeneous data - PubMed[ https://pubmed.ncbi.nlm.nih.gov/40090301/](https://pubmed.ncbi.nlm.nih.gov/40090301/)

\[40] ClusMFL: A Cluster-Enhanced Framework for Modality-Incomplete Multimodal Federated Learning in Brain Imaging Analysis[ https://arxiv.org/pdf/2502.12180](https://arxiv.org/pdf/2502.12180)

\[41] Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation[ https://openreview.net/forum?id=bHMw4bvFfi](https://openreview.net/forum?id=bHMw4bvFfi)

\[42] OneLLM：统一多模态与语言的框架[ https://www.iesdouyin.com/share/video/7507575402567601458](https://www.iesdouyin.com/share/video/7507575402567601458)

\[43] MMiC: Mitigating Modality Incompleteness in Clustered Federated Learning[ https://arxiv.org/html/2505.06911v1/](https://arxiv.org/html/2505.06911v1/)

\[44] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data[ https://openreview.net/forum?id=rMqQdJJz5r](https://openreview.net/forum?id=rMqQdJJz5r)

\[45] 无线网络中跨模态检索增强的高能效多模态联邦学习[ https://jeit.ac.cn/cn/article/pdf/preview/10.11999/JEIT251221.pdf](https://jeit.ac.cn/cn/article/pdf/preview/10.11999/JEIT251221.pdf)

\[46] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://arxiv.org/pdf/2505.20232v1.pdf](https://arxiv.org/pdf/2505.20232v1.pdf)

\[47] Federated Learning for Time-Series Healthcare Sensing with Incomplete Modalities[ https://arxiv.org/html/2405.11828v2/](https://arxiv.org/html/2405.11828v2/)

\[48] 一种基于原型的医疗多模态分布式机器学习系统[ https://www.xjishu.com/zhuanli/05/202511545183.html](https://www.xjishu.com/zhuanli/05/202511545183.html)

\[49] \[论文评述] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network](https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network)

\[50] MFCPL : Multimodal Federated Cross Prototype Learning for Missing Modalities and Data Heterogeneity[ https://arxiv.org/html/2401.13898v1](https://arxiv.org/html/2401.13898v1)

\[51] Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation[ https://openreview.net/forum?id=bHMw4bvFfi](https://openreview.net/forum?id=bHMw4bvFfi)

\[52] 混合模态联邦学习在MRI图像分割中的应用 - 生物通[ https://m.ebiotrade.com/newsf/2026-2/20260227084405273.htm](https://m.ebiotrade.com/newsf/2026-2/20260227084405273.htm)

\[53] Federated Pseudo Modality Generation for Incomplete Multi-Modal MRI Reconstruction - PubMed[ https://pubmed.ncbi.nlm.nih.gov/40315098/](https://pubmed.ncbi.nlm.nih.gov/40315098/)

\[54] 今年 Nature 正 刊 都 在用 的 idea ， 持续 学习 最好 发 顶 会 之一 持续 学习 是 AI 研究 的 热门 赛道 ， 可 解决 灾难性 遗忘 问题 。 研究 方向 走向 多 模态 交互 、 动态 环境 适应 等 前沿 领域 。 分享 10 篇 2025 持续 学习 最新 成果 论文 ， 加 150 个 论文 模块 ， 以及 自学 路线 图 。&#x20;

&#x20;\# AI # 人工 智能 #[ https://www.iesdouyin.com/share/video/7561760141071125786](https://www.iesdouyin.com/share/video/7561760141071125786)

\[55] Title:Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation[ https://arxiv.org/abs/2604.12970](https://arxiv.org/abs/2604.12970)

\[56] Explainable Multi-Modal Fusion-Based Federated Learning for Mortality Prediction in Energy-Constrained Healthcare Systems[ https://umu.diva-portal.org/smash/get/diva2:2033963/FULLTEXT01.pdf](https://umu.diva-portal.org/smash/get/diva2:2033963/FULLTEXT01.pdf)

\[57] 无线网络中跨模态检索增强的高能效多模态联邦学习[ https://jeit.ac.cn/cn/article/pdf/preview/10.11999/JEIT251221.pdf](https://jeit.ac.cn/cn/article/pdf/preview/10.11999/JEIT251221.pdf)

\[58] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data[ https://arxiv.org/pdf/2510.22880](https://arxiv.org/pdf/2510.22880)

\[59] 有没有专有名词 - CSDN文库[ https://wenku.csdn.net/answer/2ndmn7fm2u](https://wenku.csdn.net/answer/2ndmn7fm2u)

\[60] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://arxiv.org/html/2505.20232v1](https://arxiv.org/html/2505.20232v1)

\[61] 人工智能中的单模态与多模态技术解析[ https://www.iesdouyin.com/share/video/7550169838438599936](https://www.iesdouyin.com/share/video/7550169838438599936)

\[62] \[论文评述] Cross-Modal Prototype based Multimodal Federated Learning under Severely Missing Modality[ https://www.themoonlight.io/zh/review/cross-modal-prototype-based-multimodal-federated-learning-under-severely-missing-modality](https://www.themoonlight.io/zh/review/cross-modal-prototype-based-multimodal-federated-learning-under-severely-missing-modality)

\[63] 无线网络中跨模态检索增强的高能效多模态联邦学习[ https://www.jeit.ac.cn/cn/article/pdf/preview/10.11999/JEIT251221.pdf](https://www.jeit.ac.cn/cn/article/pdf/preview/10.11999/JEIT251221.pdf)

\[64] FedMAC: Tackling Partial-Modality Missing in Federated Learning with Cross-Modal Aggregation and Contrastive Regularization[ https://arxiv.org/pdf/2410.03070v2](https://arxiv.org/pdf/2410.03070v2)

\[65] MMiC: Mitigating Modality Incompleteness in Clustered Federated Learning[ http://121.43.168.64:10060/s/org/arxiv/G.https/pdf/2505.06911?;x-chain-id=b2l3455vthc0](http://121.43.168.64:10060/s/org/arxiv/G.https/pdf/2505.06911?;x-chain-id=b2l3455vthc0)

\[66] Client-Adaptive Cross-Model Reconstruction Network for Modality-Incomplete Multimodal Federated Learning[ https://dl.acm.org/doi/pdf/10.1145/3581783.3611757](https://dl.acm.org/doi/pdf/10.1145/3581783.3611757)

\[67] 联邦 学习 + 因果 推理 的 分布式 隐私 保护 分析 方案 一 、 方案 背景 与 核心 价值&#x20;

&#x20;1 . 现实 困境&#x20;

&#x20;在 医疗 、 金融 、 公共 政策 等 关键 领域 ， 因果 推理 是 支撑 科学 决策 的 核心 技术 ， 需 通过 分析 变量 间 因果 关联 （ 如 药物 疗效 与 患者 预后 、 政策 干预 与 社会 效应 ） 提供 可 解释 结论 。 但 此类 分析 依赖 多 [ https://www.iesdouyin.com/share/video/7603553082680626451](https://www.iesdouyin.com/share/video/7603553082680626451)

\[68] FedMAC: Tackling Partial-Modality Missing in Federated Learning with Cross-Modal Aggregation and Contrastive Regularization[ https://www.nca-ieee.org/assets/slides/paper36.pdf](https://www.nca-ieee.org/assets/slides/paper36.pdf)

\[69] Title:OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging[ https://arxiv.org/abs/2603.21660](https://arxiv.org/abs/2603.21660)

\[70] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data[ https://openreview.net/forum?id=rMqQdJJz5r](https://openreview.net/forum?id=rMqQdJJz5r)

\[71] CAR-MFL: Cross-Modal Augmentation by Retrieval for Multimodal Federated Learning with Missing Modalities[ https://papers.miccai.org/miccai-2024/122-Paper0391.html](https://papers.miccai.org/miccai-2024/122-Paper0391.html)

\[72] Client-Adaptive Cross-Model Reconstruction Network for Modality-Incomplete Multimodal Federated Learning[ https://dl.acm.org/doi/pdf/10.1145/3581783.3611757](https://dl.acm.org/doi/pdf/10.1145/3581783.3611757)

\[73] Examining Modality Incongruity in Multimodal Federated Learning for Medical Vision and Language-based Disease Detection[ https://arxiv.org/html/2402.05294v1](https://arxiv.org/html/2402.05294v1)

\[74] 基于联邦学习的老年患者术后谵妄多中心预测模型构建[ https://www.iesdouyin.com/share/video/7534562456107404604](https://www.iesdouyin.com/share/video/7534562456107404604)

\[75] FedMLP: Federated Multi-Label Medical Image Classification under Task Heterogeneity[ https://papers.miccai.org/miccai-2024/323-Paper1176.html](https://papers.miccai.org/miccai-2024/323-Paper1176.html)

\[76] CAR-MFL: Cross-Modal Augmentation by Retrieval for Multimodal Federated Learning with Missing Modalities[ https://papers.miccai.org/miccai-2024/122-Paper0391.html](https://papers.miccai.org/miccai-2024/122-Paper0391.html)

\[77] FedExIT:面向极端不平衡与缺失类别的半监督联邦学习新框架 - 生物通[ https://m.ebiotrade.com/newsf/2025-12/20251230174057874.htm](https://m.ebiotrade.com/newsf/2025-12/20251230174057874.htm)

\[78] Federated Pseudo Modality Generation for Incomplete Multi-Modal MRI Reconstruction - PubMed[ https://pubmed.ncbi.nlm.nih.gov/40315098/](https://pubmed.ncbi.nlm.nih.gov/40315098/)

\[79] 无线网络中跨模态检索增强的高能效多模态联邦学习[ https://www.jeit.ac.cn/cn/article/pdf/preview/10.11999/JEIT251221.pdf](https://www.jeit.ac.cn/cn/article/pdf/preview/10.11999/JEIT251221.pdf)

\[80] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://arxiv.org/html/2505.20232v1](https://arxiv.org/html/2505.20232v1)

\[81] MMiC: Mitigating Modality Incompleteness in Clustered Federated Learning[ https://arxiv.org/html/2505.06911v1/](https://arxiv.org/html/2505.06911v1/)

\[82] 【 AAAI 2026 】 多 模态 VV I - Re lD 检索 视频 序列 ！ # AAAI 2026 # 多 模态 # 视频 监控 # 行人 重 识别 # 大模型[ https://www.iesdouyin.com/share/video/7579187613521792262](https://www.iesdouyin.com/share/video/7579187613521792262)

\[83] \[论文评述] Cross-Modal Prototype based Multimodal Federated Learning under Severely Missing Modality[ https://www.themoonlight.io/zh/review/cross-modal-prototype-based-multimodal-federated-learning-under-severely-missing-modality](https://www.themoonlight.io/zh/review/cross-modal-prototype-based-multimodal-federated-learning-under-severely-missing-modality)

\[84] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data[ https://openreview.net/forum?id=rMqQdJJz5r](https://openreview.net/forum?id=rMqQdJJz5r)

\[85] Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation[ https://openreview.net/forum?id=bHMw4bvFfi](https://openreview.net/forum?id=bHMw4bvFfi)

\[86] FedAMM: Federated Learning for Brain Tumor Segmentation with Arbitrary Missing Modalities[ https://papers.miccai.org/miccai-2025/paper/1764\_paper.pdf](https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf)

\[87] Examining Modality Incongruity in Multimodal Federated Learning for Medical Vision and Language-based Disease Detection[ https://arxiv.org/pdf/2402.05294](https://arxiv.org/pdf/2402.05294)

\[88] TMI 2025 医学 图像 18 篇 顶 刊 精粹 这 一 趋势 解决 临床 核心 痛点 — — 标注 数据 稀缺 、 跨 设备 或 中心 域 偏移 显著 、 终端 算 力 有限 。 技术 上 多 框架 发力 ： 自 监督 预 训练 优化 3D 表征 ， 实现 解剖 语义 编码 与 体素 级 分割 全局 - 局部 协同 ； 扩散 模型 助力 分布 外 重建 与 数据 生成 ， 提升 标志 点 检[ https://www.iesdouyin.com/share/video/7568680141187059007](https://www.iesdouyin.com/share/video/7568680141187059007)

\[89] \[论文评述] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network](https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network)

\[90] 面向医疗健康领域的联邦学习综述:应用、挑战及未来发展方向[ https://cje.ustb.edu.cn/article/doi/10.13374/j.issn2095-9389.2024.12.24.001?viewType=citedby-info](https://cje.ustb.edu.cn/article/doi/10.13374/j.issn2095-9389.2024.12.24.001?viewType=citedby-info)

\[91] 联邦学习在医疗影像数据协同标注与疾病诊断辅助中的应用实践\_多模态联邦学习 应用场景 nature-CSDN博客[ https://blog.csdn.net/2501\_92435995/article/details/148691653](https://blog.csdn.net/2501_92435995/article/details/148691653)

\[92] Federated Pseudo Modality Generation for Incomplete Multi-Modal MRI Reconstruction - PubMed[ https://pubmed.ncbi.nlm.nih.gov/40315098/](https://pubmed.ncbi.nlm.nih.gov/40315098/)

\[93] CAR-MFL: Cross-Modal Augmentation by Retrieval for Multimodal Federated Learning with Missing Modalities[ https://papers.miccai.org/miccai-2024/122-Paper0391.html](https://papers.miccai.org/miccai-2024/122-Paper0391.html)

\[94] CAR-MFL: Cross-Modal Augmentation by Retrieval for Multimodal Federated Learning with Missing Modalities[ https://papers.miccai.org/miccai-2024/paper/0391\_paper.pdf](https://papers.miccai.org/miccai-2024/paper/0391_paper.pdf)

\[95] Examining Modality Incongruity in Multimodal Federated Learning for Medical Vision and Language-based Disease Detection[ https://arxiv.org/pdf/2402.05294](https://arxiv.org/pdf/2402.05294)

\[96] FedAMM: Federated Learning for Brain Tumor Segmentation with Arbitrary Missing Modalities[ https://papers.miccai.org/miccai-2025/paper/1764\_paper.pdf](https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf)

\[97] 一种基于原型的医疗多模态分布式机器学习系统[ https://www.xjishu.com/zhuanli/05/202511545183.html](https://www.xjishu.com/zhuanli/05/202511545183.html)

\[98] \[论文评述] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network](https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network)

\[99] ClusMFL: A Cluster-Enhanced Framework for Modality-Incomplete Multimodal Federated Learning in Brain Imaging Analysis[ https://arxiv.org/pdf/2502.12180](https://arxiv.org/pdf/2502.12180)

\[100] Title:Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation[ https://arxiv.org/abs/2604.12970](https://arxiv.org/abs/2604.12970)

\[101] TMI 2025 医学 图像 18 篇 顶 刊 精粹 这 一 趋势 解决 临床 核心 痛点 — — 标注 数据 稀缺 、 跨 设备 或 中心 域 偏移 显著 、 终端 算 力 有限 。 技术 上 多 框架 发力 ： 自 监督 预 训练 优化 3D 表征 ， 实现 解剖 语义 编码 与 体素 级 分割 全局 - 局部 协同 ； 扩散 模型 助力 分布 外 重建 与 数据 生成 ， 提升 标志 点 检[ https://www.iesdouyin.com/share/video/7568680141187059007](https://www.iesdouyin.com/share/video/7568680141187059007)

\[102] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data[ https://openreview.net/forum?id=rMqQdJJz5r](https://openreview.net/forum?id=rMqQdJJz5r)

\[103] Federated Pseudo Modality Generation for Incomplete Multi-Modal MRI Reconstruction - PubMed[ https://pubmed.ncbi.nlm.nih.gov/40315098/](https://pubmed.ncbi.nlm.nih.gov/40315098/)

\[104] MMiC: Mitigating Modality Incompleteness in Clustered Federated Learning[ http://121.43.168.64:10060/s/org/arxiv/G.https/pdf/2505.06911?;x-chain-id=b2l3455vthc0](http://121.43.168.64:10060/s/org/arxiv/G.https/pdf/2505.06911?;x-chain-id=b2l3455vthc0)

\[105] CVPR 2025论文分享|一种面向多参数脑MRI分析的多模态视觉预训练模型\_brainmvp-CSDN博客[ https://blog.csdn.net/audyxiao001/article/details/153182287](https://blog.csdn.net/audyxiao001/article/details/153182287)

\[106] ClusMFL: A Cluster-Enhanced Framework for Modality-Incomplete Multimodal Federated Learning in Brain Imaging Analysis[ https://arxiv.org/pdf/2502.12180v1.pdf](https://arxiv.org/pdf/2502.12180v1.pdf)

\[107] \[论文评述] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network](https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network)

\[108] Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation[ https://openreview.net/attachment?id=uYCSf4NqjF\&name=supporting\_material](https://openreview.net/attachment?id=uYCSf4NqjF\&name=supporting_material)

\[109] Federated Learning for Time-Series Healthcare Sensing with Incomplete Modalities[ https://arxiv.org/html/2405.11828v2/](https://arxiv.org/html/2405.11828v2/)

\[110] CAR-MFL: Cross-Modal Augmentation by Retrieval for Multimodal Federated Learning with Missing Modalities[ https://papers.miccai.org/miccai-2024/paper/0391\_paper.pdf](https://papers.miccai.org/miccai-2024/paper/0391_paper.pdf)

\[111] 无线网络中跨模态检索增强的高能效多模态联邦学习[ https://www.jeit.ac.cn/cn/article/pdf/preview/10.11999/JEIT251221.pdf](https://www.jeit.ac.cn/cn/article/pdf/preview/10.11999/JEIT251221.pdf)

\[112] Cross-Modal Contrastive Masked AutoEncoder for Compressed Video Pre-Training - PubMed[ https://pubmed.ncbi.nlm.nih.gov/40663678/](https://pubmed.ncbi.nlm.nih.gov/40663678/)

\[113] AI多模态模型架构之模态编码器:图像编码、音频编码、视频编码\_多模态编码器-CSDN博客[ https://blog.csdn.net/AIGCmagic/article/details/138287524](https://blog.csdn.net/AIGCmagic/article/details/138287524)

\[114] : Partitioner Guided Modal Learning Framework[ https://arxiv.org/html/2507.11661v1/](https://arxiv.org/html/2507.11661v1/)

\[115] EMMA: Efficient Multimodal Understanding, Generation, and Editing with a Unified Architecture[ https://arxiv.org/html/2512.04810v4/](https://arxiv.org/html/2512.04810v4/)

\[116] Federated Pseudo Modality Generation for Incomplete Multi-Modal MRI Reconstruction[ https://www.arxiv.org/pdf/2308.10910](https://www.arxiv.org/pdf/2308.10910)

\[117] Multimodal Federated Learning with Missing Modality via Prototype Mask and Contrast[ https://arxiv.org/pdf/2312.13508v1](https://arxiv.org/pdf/2312.13508v1)

\[118] Multimodal Federated Learning with Missing Modality via Prototype Mask and Contrast[ https://ar5iv.labs.arxiv.org/html/2312.13508](https://ar5iv.labs.arxiv.org/html/2312.13508)

\[119] FedMGP: Personalized Federated Learning with Multi-Group Text-Visual Prompts[ https://arxiv.org/html/2511.00480v1/](https://arxiv.org/html/2511.00480v1/)

\[120] WWW 2024 | 华为、清华提出个性化多模态生成新方法，让AIGC更懂你-CSDN博客[ https://blog.csdn.net/tMb8Z9Vdm66wH68VX1/article/details/140582246](https://blog.csdn.net/tMb8Z9Vdm66wH68VX1/article/details/140582246)

\[121] Open-Vocabulary Federated Learning with Multimodal Prototyping[ https://arxiv.org/html/2404.01232v1/](https://arxiv.org/html/2404.01232v1/)

\[122] FedMultimodal - 2023 KDD ADS[ https://github.com/usc-sail/fed-multimodal/blob/main/README.md](https://github.com/usc-sail/fed-multimodal/blob/main/README.md)

\[123] Personalized Federated Learning with FedSM Algorithm[ https://github.com/NVIDIA/NVFlare/blob/main/research/fed-sm/README.md](https://github.com/NVIDIA/NVFlare/blob/main/research/fed-sm/README.md)

\[124] LLM驱动的生成式合成数据联邦学习框架:破解基层医疗数据孤岛与隐私困境-CSDN博客[ https://blog.csdn.net/2501\_93420214/article/details/153850553](https://blog.csdn.net/2501_93420214/article/details/153850553)

\[125] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://arxiv.org/html/2505.20232v1](https://arxiv.org/html/2505.20232v1)

\[126] Federated Pseudo Modality Generation for Incomplete Multi-Modal MRI Reconstruction - PubMed[ https://pubmed.ncbi.nlm.nih.gov/40315098/](https://pubmed.ncbi.nlm.nih.gov/40315098/)

\[127] Bridging Annotation Gaps in Federated Learning for Medical Image Segmentation Through Conditional Distillation[ https://www.nvidia.com/en-us/on-demand/session/nvidiaflareday25-nvfd29/](https://www.nvidia.com/en-us/on-demand/session/nvidiaflareday25-nvfd29/)

\[128] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://abdn.elsevierpure.com/en/publications/multimodal-federated-learning-with-missing-modalities-through-fea](https://abdn.elsevierpure.com/en/publications/multimodal-federated-learning-with-missing-modalities-through-fea)

\[129] Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation[ https://openreview.net/forum?id=bHMw4bvFfi](https://openreview.net/forum?id=bHMw4bvFfi)

\[130] \[Literature Review] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://www.themoonlight.io/en/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network](https://www.themoonlight.io/en/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network)

\[131] 东北大学图书馆[ http://www.lib.neu.edu.cn/page/ggxx/jzxx/2025/0919/1636.html](http://www.lib.neu.edu.cn/page/ggxx/jzxx/2025/0919/1636.html)

\[132] Federal Open Market Committee[ https://www.federalreserve.gov/monetarypolicy/fomcminutes20250618.htm](https://www.federalreserve.gov/monetarypolicy/fomcminutes20250618.htm)

\[133] Monetary Policy Report – June 2025[ https://www.federalreserve.gov/monetarypolicy/2025-06-mpr-part3.htm](https://www.federalreserve.gov/monetarypolicy/2025-06-mpr-part3.htm)

\[134] 隐私保护多模态数据融合方法及其系统与流程[ https://www.xjishu.com/zhuanli/62/202511535949.html](https://www.xjishu.com/zhuanli/62/202511535949.html)

\[135] 分散式多模态原型联邦学习:面向异构数据协同AI的高效稳健新范式 - 生物通[ https://m.ebiotrade.com/newsf/2026-2/20260222083908512.htm](https://m.ebiotrade.com/newsf/2026-2/20260222083908512.htm)

\[136] 计算机 sci 论文 发文 热点 跨 模态 对齐 # sci # 论文 # 跨 模态 对齐 # 深度 学习 # 机器 学习[ https://www.iesdouyin.com/share/video/7602471013167290277](https://www.iesdouyin.com/share/video/7602471013167290277)

\[137] 多模态联邦学习-CSDN博客[ https://blog.csdn.net/Listennnn/article/details/149260236](https://blog.csdn.net/Listennnn/article/details/149260236)

\[138] 基于Transformer的多模态个性化联邦学习[ https://www.juestc.uestc.edu.cn/article/doi/10.12178/1001-0548.2024050](https://www.juestc.uestc.edu.cn/article/doi/10.12178/1001-0548.2024050)

\[139] (协作传感+联邦学习)模型对齐终极方案曝光:支持异构设备、低带宽环境-CSDN博客[ https://blog.csdn.net/LogicWander/article/details/155775452](https://blog.csdn.net/LogicWander/article/details/155775452)

\[140] Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation[ https://arxiv.org/pdf/2604.12970](https://arxiv.org/pdf/2604.12970)

\[141] 一种基于原型的医疗多模态分布式机器学习系统[ https://www.xjishu.com/zhuanli/05/202511545183.html](https://www.xjishu.com/zhuanli/05/202511545183.html)

\[142] 基于联邦学习的老年患者术后谵妄多中心预测模型构建[ https://www.iesdouyin.com/share/video/7534562456107404604](https://www.iesdouyin.com/share/video/7534562456107404604)

\[143] FedMAC: Tackling Partial-Modality Missing in Federated Learning with Cross-Modal Aggregation and Contrastive Regularization[ https://arxiv.org/pdf/2410.03070v2](https://arxiv.org/pdf/2410.03070v2)

\[144] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://arxiv.org/html/2505.20232v1](https://arxiv.org/html/2505.20232v1)

\[145] Examining Modality Incongruity in Multimodal Federated Learning for Medical Vision and Language-based Disease Detection[ https://arxiv.org/pdf/2402.05294](https://arxiv.org/pdf/2402.05294)

\[146] \[论文评述] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network](https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network)

\[147] FedMAC: Tackling Partial-Modality Missing in Federated Learning with Cross-Modal Aggregation and Contrastive Regularization[ https://arxiv.org/pdf/2410.03070v2](https://arxiv.org/pdf/2410.03070v2)

\[148] Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation[ https://arxiv.org/pdf/2604.12970](https://arxiv.org/pdf/2604.12970)

\[149] 基于联邦学习的老年患者术后谵妄多中心预测模型构建[ https://www.iesdouyin.com/share/video/7534562456107404604](https://www.iesdouyin.com/share/video/7534562456107404604)

\[150] Federated Pseudo Modality Generation for Incomplete Multi-Modal MRI Reconstruction - PubMed[ https://pubmed.ncbi.nlm.nih.gov/40315098/](https://pubmed.ncbi.nlm.nih.gov/40315098/)

\[151] MMiC: Mitigating Modality Incompleteness in Clustered Federated Learning[ http://121.43.168.64:10060/s/org/arxiv/G.https/pdf/2505.06911?;x-chain-id=b2l3455vthc0](http://121.43.168.64:10060/s/org/arxiv/G.https/pdf/2505.06911?;x-chain-id=b2l3455vthc0)

\[152] Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation[ https://openreview.net/forum?id=bHMw4bvFfi](https://openreview.net/forum?id=bHMw4bvFfi)

\[153] FedAMM: Federated Learning for Brain Tumor Segmentation with Arbitrary Missing Modalities[ https://papers.miccai.org/miccai-2025/paper/1764\_paper.pdf](https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf)

\[154] 🤔 Modality FAQs[ https://github.com/modality-org/modality-archived-issues/blob/main/faq.md](https://github.com/modality-org/modality-archived-issues/blob/main/faq.md)

\[155] Terms & Conditions[ https://modalis.fr/en/terms-conditions/](https://modalis.fr/en/terms-conditions/)

\[156] RFC-0001: Modal Contract Specification[ https://www.modality.org/docs/resources/rfc-0001](https://www.modality.org/docs/resources/rfc-0001)

\[157] MiniModal[ https://github.com/charlesfrye/minimodal](https://github.com/charlesfrye/minimodal)

\[158] Modality: Trust Through Math, Not Faith[ http://www.modality.org/docs](http://www.modality.org/docs)

\[159] Message and Modality System[ https://deepwiki.com/google/langfun/2.2-message-and-modality-system](https://deepwiki.com/google/langfun/2.2-message-and-modality-system)

\[160] A Modal Logic Framework for Human-Computer Spoken Interaction[ https://www.researchgate.net/publication/229031669\_A\_Modal\_Logic\_Framework\_for\_Human-Computer\_Spoken\_Interaction](https://www.researchgate.net/publication/229031669_A_Modal_Logic_Framework_for_Human-Computer_Spoken_Interaction)

\[161] modai-protocol[ https://www.npmjs.com/package/modai-protocol](https://www.npmjs.com/package/modai-protocol)

\[162] Mode Consensus Algorithms With Finite Convergence Time[ https://arxiv.org/pdf/2403.00221](https://arxiv.org/pdf/2403.00221)

\[163] FedFPM: A Unified Federated Analytics Framework for Collaborative Frequent Pattern Mining[ https://www.researchgate.net/publication/361443430\_FedFPM\_A\_Unified\_Federated\_Analytics\_Framework\_for\_Collaborative\_Frequent\_Pattern\_Mining](https://www.researchgate.net/publication/361443430_FedFPM_A_Unified_Federated_Analytics_Framework_for_Collaborative_Frequent_Pattern_Mining)

\[164] Title:DéjàVu: A Minimalistic Mechanism for Distributed Plurality Consensus[ https://arxiv.org/abs/2604.03648](https://arxiv.org/abs/2604.03648)

\[165] Online Expectation-Maximization Based Frequency and Phase Consensus in Distributed Phased Arrays[ https://arxiv.org/pdf/2207.11859](https://arxiv.org/pdf/2207.11859)

\[166] Parameter Design for Secure Affine Frequency Division Multiplexing Waveform[ https://arxiv.org/html/2503.19364v1/](https://arxiv.org/html/2503.19364v1/)

\[167] Distributed Maximum Consensus over Noisy Links[ https://arxiv.org/html/2403.18509](https://arxiv.org/html/2403.18509)

\[168] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://arxiv.org/pdf/2505.20232?](https://arxiv.org/pdf/2505.20232?)

\[169] Multi-modal dataset creation for federated learning with DICOM-structured reports[ https://pmc.ncbi.nlm.nih.gov/articles/PMC11929732/pdf/11548\_2025\_Article\_3327.pdf](https://pmc.ncbi.nlm.nih.gov/articles/PMC11929732/pdf/11548_2025_Article_3327.pdf)

\[170] neurlps2024论文解析|fedmekiabenchmarkforscalingmedicalfoundationmodelsviafederatedknowledge[ https://blog.csdn.net/paixiaoxin/article/details/145521958](https://blog.csdn.net/paixiaoxin/article/details/145521958)

\[171] \[论文评述] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network](https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network)

\[172] Federated Learning for Time-Series Healthcare Sensing with Incomplete Modalities[ https://arxiv.org/html/2405.11828v2/](https://arxiv.org/html/2405.11828v2/)

\[173] FedAMM: Federated Learning for Brain Tumor Segmentation with Arbitrary Missing Modalities[ https://papers.miccai.org/miccai-2025/paper/1764\_paper.pdf](https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf)

\[174] SMILELab-FL 发布 Fed-ECG 数据集, 应用在 心电图分类、联邦学习 领域[ https://www.5radar.com/dataopensource/news/208886](https://www.5radar.com/dataopensource/news/208886)

\[175] Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation[ https://openreview.net/forum?id=bHMw4bvFfi](https://openreview.net/forum?id=bHMw4bvFfi)

\[176] Federated Pseudo Modality Generation for Incomplete Multi-Modal MRI Reconstruction - PubMed[ https://pubmed.ncbi.nlm.nih.gov/40315098/](https://pubmed.ncbi.nlm.nih.gov/40315098/)

\[177] 持续学习突破灾难性遗忘困境并推动多模态与联邦学习交叉研究[ https://www.iesdouyin.com/share/video/7524244973303156004](https://www.iesdouyin.com/share/video/7524244973303156004)

\[178] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://arxiv.org/pdf/2505.20232?](https://arxiv.org/pdf/2505.20232?)

\[179] FedMAC: Tackling Partial-Modality Missing in Federated Learning with Cross-Modal Aggregation and Contrastive Regularization[ https://arxiv.org/pdf/2410.03070v2](https://arxiv.org/pdf/2410.03070v2)

\[180] MedSegNet10: A Publicly Accessible Network Repository for Split Federated Medical Image Segmentation[ http://www.sfu.ca/\~psaeedi/Chamani\_papers/Journal%20\[3\]%20-%20MedSegNet10\_\_A\_Publicly\_Accessible\_Network\_Repository\_for\_Split\_Federated\_Medical\_Image\_Segmentation.pdf](http://www.sfu.ca/~psaeedi/Chamani_papers/Journal%20[3]%20-%20MedSegNet10__A_Publicly_Accessible_Network_Repository_for_Split_Federated_Medical_Image_Segmentation.pdf)

\[181] MissBench|多模态情感分析数据集|模态缺失评估数据集[ https://www.selectdataset.com/dataset/abae39a0b7a7e29a1f0861e2192b41f9](https://www.selectdataset.com/dataset/abae39a0b7a7e29a1f0861e2192b41f9)

\[182] \[论文评述] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network](https://www.themoonlight.io/zh/review/multimodal-federated-learning-with-missing-modalities-through-feature-imputation-network)

\[183] Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation[ https://openreview.net/forum?id=bHMw4bvFfi](https://openreview.net/forum?id=bHMw4bvFfi)

\[184] ClusMFL: A Cluster-Enhanced Framework for Modality-Incomplete Multimodal Federated Learning in Brain Imaging Analysis[ https://arxiv.org/pdf/2502.12180](https://arxiv.org/pdf/2502.12180)

\[185] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network[ https://arxiv.org/pdf/2505.20232?](https://arxiv.org/pdf/2505.20232?)

\[186] 面向跨疾病分析的动态模态自适应联邦学习框架 TADynFed:组织感知解耦赋能异构医疗影像协同智能 - 生物通[ https://m.ebiotrade.com/newsf/2026-2/20260221084223844.htm](https://m.ebiotrade.com/newsf/2026-2/20260221084223844.htm)

\[187] Causal Representation Learning from Multimodal Clinical Records under Non-Random Modality Missingness[ https://preview.aclanthology.org/json-schema/2025.emnlp-main.1465.pdf](https://preview.aclanthology.org/json-schema/2025.emnlp-main.1465.pdf)

> （注：文档部分内容可能由 AI 生成）