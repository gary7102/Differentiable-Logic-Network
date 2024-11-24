---
title: Differential Logic Network

---

# <font color = "#F7A004">Binary Neural Networks (BNN)  </font>
## <font color = green>def.</font>
使用$sign(x)$函數使權重（weights）和激活值（activations）能夠控制在{−1,+1}，使得網路中大部分算術運算可以通過bitwise operation，例如使用 XNOR 和 Popcount 來代替乘法和加法  
**例:** 
$x {\oplus} y = 1$, only if x=y  
$x {\oplus} y = 0$, when x!= y  

  
![image](https://hackmd.io/_uploads/r1ijfiTfJl.png)  
![image](https://hackmd.io/_uploads/BJB-7i6z1e.png)  
$x$是實際floatin point weight或floatin point activation  

## <font color = green>problem</font>
因為$sign(x)$函數為非連續型函數，在x=0時無法微分，x!=0時微分=0，**因此無法微分(nondifferentiable)**，這樣一來就無法使用back propagation或是gradient decent來更新模型參數。

## <font color = green>solution</font>
使用平滑函數$Htanh(x)$取代函數$sign(x)$
![image](https://hackmd.io/_uploads/Bk7l4iaf1e.png)  
這樣當 $x∈[−1,1]$時，微分=1，$x<-1$ or $x>1$時微分=0  
![image](https://hackmd.io/_uploads/SywhHipG1e.png)  

<font size = 4>**優點:**</font>  
計算效率受益於由floatin point 壓縮至binary 數值，且算術運算由大量的乘法加法改完bitwise operation 而大幅提升  

<font size = 4>**缺點:**</font>  
**喪失模型精度**，二值化會大幅降低weights和activation的表達能力，導致模型的精度下降。


# <font color = "#F7A004">XNOR-Net</font>
## <font color = green>def.</font>
其實就是BNN並引入**scaling factor**，目的是為了減少因為bnn帶來的精度損失  
**例:**  
$$XNOR_{Result} = XNOR(Weight,Activation)$$  
$$Binary_{Sum}=Popcount(XNOR_{Result})$$
$$Z=α⋅β⋅Binary_{Sum}$$
也就是比起BNN的結果多乘上 $α⋅β$，分別為$$α= \frac{∥W∥}{n}，β = \frac{∥A∥}{n}$$
$W$及$A$分別為Weight 及Activation Matrix  
  

<font size = 4>**優點:**</font>  
由於引入 Scaling Factor，在計算過程中多乘以$α⋅β$，比起BNN具有更加的模型精度，更接近原始的floating point模型  

<font size = 4>**缺點:**</font>
雖然推論時依舊使用 XNOR 和 Popcount取代乘法及加法，但最終還需要將結果乘以scaling factor，$𝛼$和$𝛽$，使用浮點運算，對極度受限的硬件環境可能仍有影響。

# <font color = "#F7A004">ReactNet </font>
取名自Re-activation，表示重新設計activation function結構來提升二值化神經網路的性能。

論文顯示activation值數值分布差異會對二值化卷積神經網路的性能有顯著影響，為了改善activation值數值分布差異，ReActNet 引入 ReAct-Sign(RSign)函數和 ReAct-PReLU(RPReLU)函數

## <font color = green>RSign vs. Sign</font>

![image](https://hackmd.io/_uploads/Hy9LZaafJx.png)
![image](https://hackmd.io/_uploads/ByvDWaafyx.png)
* $α$ 作為一個可學習的參數，允許網路自適應不同輸入特徵的分佈
* 比固定閾值 0 的 $Sign$ 函數更靈活，能捕捉輸入的細微變化


## <font color = green>RPReLU vs. ReLU</font>
![image](https://hackmd.io/_uploads/HkRjZp6fkg.png)
![image](https://hackmd.io/_uploads/Bk43b66fJx.png)
* $γ$：拐點參數
    - $γ$ 通常設為可學習參數，由網絡在訓練過程中自動調整。
    - 初始值一般設為輸入數據的均值或中位數，方便激活函數快速適應。
* $δ$：平移參數
    - 在正值和負值區域內，$𝛿$ 提供了額外的靈活性，用於補償特徵值的偏移
    - 初始值通常設為0，以避免對初期訓練造成影響
* $β$：負值區域斜率
    - 如果輸入值小於 $𝛾$，$𝛽$決定輸出的平滑程度
    - 初始值通常設為一個小的正數（如 0.25 或0.1），保證在負值區域不完全忽略輸入
    
<font size = 4>**優點:**</font>  

* 適應性更強
    - $γ$ 和 $δ$ 的引入讓激活函數可以根據數據特徵動態調整拐點和輸出範圍 
* 負值區域特徵提取能力提升
    - 傳統的 ReLU 直接將負值輸入截斷為 0，而 RPReLU 通過 $β$ 調整負值區域，保留了更多有用的特徵信息。
* 更平滑的梯度
    - 分段和多參數設計讓 RPReLU 的梯度變化更平滑，有助於訓練穩定性和模型收斂。

## <font color = green>Normal Block 和 Reductino Block</font>

<font size = 4>**normal Block**</font>  
用來提取特徵，透過多層卷積和激活操作逐層提取特徵，構建更深、更精細的數據表示，其中特別設計了RSign 及 RPReLU 來克服Binary Neural Network的特徵表示能力下降的問題。


<font size = 4>**Reduction Block**</font>  
隨著網絡的加深，特徵圖的空間尺寸（寬度和高度）如果保持不變，會導致計算量和存儲需求呈現指數級增長，因此使用Reduction Block 來壓縮特徵，使用stride=2使特徵圖的長和寬都各減一半，  
假設原始特徵圖大小為$H * W$，經過卷積後變成$\frac{H}{2} * \frac{W}{2}$，計算量也減少到原本的$\frac{1}{4}$

![image](https://hackmd.io/_uploads/HJzS_-0zke.png)


## <font color = green>Loss Function</font>

![image](https://hackmd.io/_uploads/r1dDWxCfyg.png)

看不懂，只知道是croos-entropy，

# <font color = "#F7A004">Logic network </font>

LGNs 和 BNN的最大差異：
LGNs 的核心是學習邏輯規則，例如「如果條件 A 和條件 B 成立，則輸出 1」。
它的計算基於邏輯操作，而非數值權重(BNN)。
網絡結構更像是邏輯電路，而非數值計算機(BNN)

## <font color = "green">Poblem</font>
每個neuron輸出如果只是0 or 1的話，那就無法反映模型對於該class的信心程度，
ex:
例如，輸出向量 [貓, 狗, 鳥] = [1,1,0]中「貓」和「狗」都有 1，無法區分優先選擇哪一個類別。 

## <font color = "green">Solution</font>
對每個class使用多個神經元，以增加信心程度
解決方法: 使用**多個神經元**為每個類別生成更多訊息(證據)，透過這些訊息(證據)求和實現更細緻的classfication
ex:
假設對於「貓」類別有 3 個神經元，其輸出分別為[1,0,1]，求和後的總和為2，表示對「貓」的信心程度為 2（越大代表越有信心）  

![image](https://hackmd.io/_uploads/BJhwsg1Xkx.png)
如上圖，panda及Gibbon這兩個class都使用2個neuron，透過bitcount(求和)之後 panda 信心程度為2，因此模型預測輸入圖像屬於Panda


# <font color = "#F7A004">Differential Logic network </font>

## Core Idea
原始的logic network需要在一開始就預先定義每個neuron的logic operation，也就是同一個neuron從頭到尾都是做`&`或是`|`或是`xor`，雖然這樣能對結果能夠著高度的解釋性，但不管是靈活性或是處理複雜情境都表現差勁

因此difflogic出現，也就是將logic gates 先做relaxtion，將離散的logic operation成為可以微分的logic operation，如:
$$A \land B = A*B$$ $$A \lor B = A + B - A * B$$

當logic gates可以微分之後，在訓練階段就可以使用gradient decent學習，學習甚麼?
**學習每個neuron 最適合使用甚麼logic operation**(機率分布)，到了推理階段，便直接選擇最適合的logic operation使用。

<font size = 4>**訓練階段:**</font>
每個神經元對應一個邏輯操作（如 AND、OR、XOR 等），這些邏輯操作並不是一開始就固定的，而是由訓練階段來學習每個神經元的logic gate的機率分布，

<font size = 4>**推理階段:**</font>
在inference階段，模型不再使用概率分佈來表示邏輯操作，而是直接選擇每個神經元概率最高的邏輯操作作為固定的操作，如上panda圖中，選擇72


## <font color = green>Differentiable Logics</font>

<font size = 4>**Step 1**</font>
傳統的logic network 屬於$a ∈ {0, 1}$, we relax all values to probabilistic activations $a ∈ [0, 1]$，也就是從輸入與輸出只能是0 or 1 ，relax to 0~1區間的連續實數，這樣可以用來描述「部分真」或「部分假」，即一個事件發生的機率。

<font size = 4>**Step 2**</font>
將logic gates(and, or, xor)轉換為計算**期望值的機率分布公式**，如:

When $A=0.8$ 且 $B= 0.6$，
公式: $A \land B = A*B$
則，
$$A * B = 0.8*0.6 = 0.48$$

表示A和B同時發生的機率為48%，也就是說，在這個logic network中，事件$𝐴∧𝐵$的發生程度僅為「部分成立」，不是完全的 0 或 1。

<font size = 4>**Activation**</font>

![image](https://hackmd.io/_uploads/rJeaHzkQJg.png)
如上面算到的0.48

## <font color = green>Differentiable Choice of Operator</font>

上面提到的activation雖然可以微分，但在訓練階段卻無法更新，因此使用categorical probability distribution，定義了differentiable logic gate neuron

![image](https://hackmd.io/_uploads/Sk7R0GymJg.png)

$p_i$     為probability of each logic operation(由softmax計算)，總共16種logic operation，
$f_i(a_1, a_2)$ 為輸入$a_1$, $a_2$，第$i$個logic operation 的輸出 (如上0.48)

:::success
訓練的目標是透過back propagation調整每個 logic gate的機率 $p_i$ ，使最適合該神經元的logic gate的機率趨於最大。
但是我們不能直接更新 $p_i$ ，因為 $p_i$ 是softmax計算出來的，需要符合$\sum_{i=0} p_i= 1$，因此back propagation更新的其實是 $w_i$，也就是每個logic gate在該神經元中的「得分」，經過softmax才成為 $p_i$。
論文提到，在訓練最一開始，$w_i$是從normal distribution中隨機選取的，代表每個logic gate的機率分布是均勻的
:::


## <font color = green>Aggregation of Output Neurons</font>

假設網路中output layer有n個神經元，每個神經元在$[0, 1]$區間中，這些輸出值可能需要進一步聚合為 $k$ 個輸出($k$個class，如熊貓圖class = 2)，因此作者希望預測k個更大範圍的值而非僅限於$[0, 1]$，除此之外還需要更細化的輸出(graded output)

所以aggregate the output as:  
![image](https://hackmd.io/_uploads/BJOCoNy7yg.png)

$τ$ is a normalization temperature and $β$ is an optional offset  

每個 $y_i$ 是對一組輸出神經元值的nomalization 和offset後的加總，表示對應類別或輸出目標的信心值

:::success
簡單來說，就是求class 個數 $k$ 的aggregate output，直接表示每個class的信心值
:::

<font size = 4>**Ex:**</font>
* 輸出曾有$n=6$個神經元，分別為$[0.5, 0.8, 0.2, 0.7, 0.3, 0.6]$  
* class 個數 $k = 2$，則需要計算 $\hat y_{1}$ 及 $\hat y_{2}$
* 假設 $τ = 0.5$, $β=0.1$

則:
$$y_{1} = \frac{0.5}{0.5} + \frac{0.8}{0.5} + \frac{0.2}{0.5} + 0.1 = 3.1$$
$$y_{2} = \frac{0.7}{0.5} + \frac{0.3}{0.5} + \frac{0.6}{0.5} + 0.1 = 3.3$$
得最終output layer為:
$$[\hat y_{1}, \hat y_{2}] = [3.1, 3.3]$$

以圖片分類例子而言，神經網路會預測第2個class

## <font color = green>Loss function</font>

<font size = 4>**Softmax Cross-Entropy Loss:**</font>  
$$L = - \sum_it_i * log(q_i)$$

* $t_i$ :圖片分類的正確答案的one-hot 分布，對正確的class $i$，$t_i = 1$，否則為0
* $log(q_i)$ :模型對正確class的預測機率(先通過Softmax)，取 $log$

目的是最小化 Loss，從而提高模型對真實類別的預測機率

<font size = 4>**舉例:**</font>

**Step1: 計算$q_i$，第$i$ class的預測機率**

對aggregate output $y_i$ 使用Softmax，計算每個類別的預測概率：
$$q_i = \frac{e^{\hat y_i}}{\sum_{i}^{k} e^{\hat y_i}}$$


**Step2: 計算 Loss**

將$q_i$帶入Softmax Cross-Entropy Loss，就可以得到loss 了
