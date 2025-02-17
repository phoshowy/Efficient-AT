# On the Effect of Pruning on Adversarial Robustness （2021-ICCV）
### 攻击方式
- semantic-preserving (δ = 4)
- occlusion (δ = 16)
- FGSM (δ = 8/255)

### 分类器
- VGG16(plain)
- MobileNetV2(lightweight)
- ResNet56 、ResNet50 (residual architectures)

### 剪枝
-  **pruning criterion：** ℓ1-norm、ExpectedABS、HRank、Kl Divergence、PLS
- 修剪filter或layer


### 结论

- 剪枝在不牺牲 **泛化** 的情况下提高了鲁棒性

- 剪枝标准为 **PLS** 时，在多种攻击下平均性能最好
- 剪枝后使用 **Fine-tuning** 调整参数效果最好
- 与现有的防御机制相比，剪枝是鲁棒性和泛化能力的最佳折衷


# Masking Adversarial Damage: Finding Adversarial Saliency for Robust and Sparse Network （2022-CVPR）

### 分类器
- VGG16
- ResNet18
- WRN-28-10
### 数据集
- Cifar10
- SVHN
- Tiny-ImageNet (64*64)
### 攻击方式
- PGD
- FGSM
- CW∞
- AP (Auto-PGD) 
- AA (Auto-Attack)
### 防御方式
- TRADES
- MART
### 剪枝方法
- **MAD(Masking Adversarial Damage)**
- LWM (2019)
- ADMM (2019)
- HYDRA (2020)
### 结论
- 通过利用二阶信息（Hessian），结合掩码优化和分块K-FAC算法，可以精确估计网络参数的对抗显著性
- 初始层在捕获局部特征（如颜色和边缘）方面起着重要作用，**对对抗性损失高度敏感**

# SAMPLES ON THIN ICE: RE-EVALUATING ADVERSARIAL PRUNING OF NEURAL NETWORKS (2023-arxiv)
### 攻击方式
采用更严格的攻击方式（如 AutoAttack 和 RobustBench），避免高估模型性能
AutoAttack framework (plus) 
- untargeted APGD-CE (5restarts)
- untargeted APGD-DLR (5 restarts)
- untargeted FAB(5 restarts)
- Square Attack (5000 queries)
- targeted APGD-DLR (9 target classes)
- targeted FAB (9 target classes)
### 结论
- 关注到剪枝模型与正常模型对样本的分类情况 S<sub>0,0</sub> S<sub>1,0</sub> S<sub>0,1</sub> S<sub>1,1</sub>  
被剪枝模型错误分类的样本（S1,0）和被剪枝模型纠正的样本（S0,1）通常位于决策边界附近（“samples on thin ice”）。
剪枝会导致决策边界发生微小变化，这些变化主要影响靠近边界的样本

# HOLISTIC ADVERSARIALLY ROBUST PRUNING （2023-ICLR）

目前SOTA关注**哪些参数**要修剪，本文认为修剪的量也需要关注，并使用**非均匀修剪**（各层的压缩强度不同）

### HARP
涉及两个参数：**压缩配额$\gamma$**，它是压缩率的一个可学习的表示和用于确定网络连接的**重要性分数S** \
1.**全局压缩控制** 确保了压缩后的参数数量不低于目标压缩率 \
2.开始时主要关注鲁棒性，然后逐渐转向目标压缩率，最后同时处理两个目标($\gamma$逐渐增大后保持不变)

### 性能
- 比ADMM 、 HYDRA、BCS-P好
- 在中等压缩（<90%）,HARP和MAD不相上下，高压缩下HARP更好

# Less is More: Data Pruning for Faster Adversarial Training（2023-AAAI-safaAI）

课程对抗训练（Curriculum Adversarial Training,2018）通过调整PGD步骤从弱攻击强度到强攻击强度来增强DNN\
友好对抗训练（Friendly Adversarial Training,2020）则为对抗示例执行早期停止的PGD
### 数据剪枝
Adv-GLISTER：通过最大化验证数据集上子集的对数似然\
AdvGRAD-MATCH：通过最小化子集与完整数据集之间的梯度差异

- 结合Bullettrain(决策边界)加速训练
- 速度提升但精度下降？！



 
# ADVERSARIAL PRUNING: A SURVEY AND BENCHMARK OF PRUNING METHODS FOR ADVERSARIAL ROBUSTNESS（2024-arxiv）
### 剪枝方法
|unstructured(US)|structured(S)|
|:--:|:--:|
HARP、FlyingBird|TwinRep
TwinRep、FlyingBird|HARP、TwinRep

HARP:HOLISTIC ADVERSARIALLY ROBUST PRUNING （2023-ICLR） \
FlyingBird:Sparsity winning twice: Better robust generalization from more efficient training（2022-ICLR）\
TwinRep：Learning Adversarially Robust Sparse Networks via Weight Reparameterization（2023-AAAI）
#### 安全性曲线？
- 当扰动较小时，HARP通常是最稳健的模型
- FlyingBird通常在8/255附近的一个小窗口内最为稳健
- 随着扰动的增大，HYDRA则是最稳健的模型
### 结论 
- 使用多个攻击方法（如 AutoAttack）来进行鲁棒性评估的重要性，避免单一攻击导致结果不够可靠
- 结构化（S）与非结构化（US）修剪的效果不同，结构化实现对高稀疏性的较低容忍度（因此，这意味着选择较低的sr值范围）。因此，当使用S时，我们用50%、75%和90%的sr修剪每个模型，而当使用US时，我们用90%、95%和99%的sr修剪每个模型。
    -  在高稀疏情况下（90%），非结构化剪枝效果比结构化剪枝好


# A Survey on Deep Neural Network Pruning: Taxonomy, Comparison, Analysis, and Recommendations(2024-TPAMI)
### 结构/非结构
- 只有结构化剪枝才能实现通用的神经网络加速，而不需要特殊的硬件或软件
- 为了提高结构化剪枝的灵活性并在高剪枝率时实现更低的准确度下降，一些工作引入了半结构化剪枝

### 剪枝时机
- 与初始化时修剪的子网络相比，在训练期间或之后通过修剪得到的子网络表现出更高的有效参数计数和更高的表达能力。
- 在 PBT 中，像 SynFlow 这样的数据无关方法能避免对数据的依赖，同时减少训练资源消耗。对于 CNN 而言，SynFlow 可以有效在初始化阶段通过权重梯度流分析找到关键通道和卷积核，从而构建鲁棒的稀疏网络

### 剪枝标准
- 一些后剪枝的目标函数/评分标准
- 基于幅度的修剪导致比幅度不可知方法更快的模型收敛。
- 基于二阶 Taylor 展开的剪枝方法（如 SOSP 和 LLM-Pruner）能精确评估剪枝对损失的影响，适合高精度剪枝
    - 一阶 Taylor 方法适合快速评估，二阶方法则在鲁棒性方面更优
- 结合多种准则（如幅值与敏感性）能提升剪枝的精确性和鲁棒性

### 实验结果
- 非结构化修剪效果好于结构化修剪 **（但是鲁棒性呢？没有管）**
- 结合局部（每层内）和全局（跨层）剪枝，可以更好地平衡稀疏性和鲁棒性
- 如果需要在多个维度上修剪神经网络，可以综合考虑分层修剪（减少模型的深度）、通道修剪（减少模型的宽度）、图像分辨率修剪（降低模型的输入分辨率）或标记修剪（选择性地从文本数据中删除标记）

# Auto-Train-Once: Controller Network Guided Automatic Network Pruning from Scratch （2024-arxiv）
训练中剪枝   结构化剪枝  
ATO 完全自动化地进行深度神经网络的训练和剪枝，不需要额外的后期微调步骤
- 滤波器修剪选择具有较大范数值的重要结构

### ATO

- ATO 引入了 Zero-Invariant Groups (ZIGs) 的概念，将模型的训练参数分为多个不变的组，有效地识别并剪枝对模型没有贡献的部分
    - 更能保留模型的关键特征，同时减少不必要的计算。
- ATO 使用一个控制器网络来动态地引导剪枝过程。
控制器网络动态生成一个二进制掩码（mask），根据掩码决定哪些通道（或层）应该被剪枝。
    - 能够自适应地选择最优的剪枝方案，避免陷入局部最优解。

- 通过结合正则化与剪枝，ATO 能够在减小模型大小的同时，尽量不影响模型性能
- ATO 提供了多种投影算子的选择，例如 半空间投影算子（HSPG） 和 近端梯度投影算子，用于对剪枝后的模型进行参数更新

### 超参数
1. **λ** 控制正则化的强度，影响模型的稀疏性和剪枝效果
    - 适中范围内，过小或过大的 λ 值都会影响剪枝效果和模型性能
    - λ = 10 是实验中一个较好的经验值，可以在保证较高性能的同时，减小计算量

2. **γ**  控制 FLOPs 正则化的强度，影响模型的计算开销
    - 过小的 γ 会导致模型的计算开销无法有效减少，剪枝不明显；而过大的 γ 会使得模型的 FLOPs 剪枝目标难以实现，影响模型的压缩效果
    - γ = 4.0 可以在保证剪枝目标和计算效率的前提下，有效地减少计算开销并保持较好的模型精度
3. **T<sub>w</sub>** 控制器网络热身步数,影响控制器网络的训练时间，确保剪枝过程的精确性
    - Tw = 30% 总训练步数
4. **p** 决定剪枝的强度，影响模型大小和计算量
    - 剪枝比例 p 在 0.35 到 0.45 之间较为理想
5. 投影方法
    - 近端梯度投影 相对来说实现简单，且计算效率较高，适用于大多数剪枝任务 **（默认方法）**
    - HSPG 投影 在一些大比例剪枝的任务中表现更好，但其计算复杂度较高，