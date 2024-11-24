
# <font color = "#F7A004">Logic network </font>
先簡單介紹logic network再介紹diff logic network  

LGNs 和 BNN的最大差異：
LGNs 的核心是學習邏輯規則，例如「如果條件 A 和條件 B 成立，則輸出 1」。
它的計算基於邏輯操作，而非數值權重(BNN)。
網絡結構更像是邏輯電路，而非數值計算機(BNN)

## <font color = "green">Poblem</font>
每個neuron輸出如果只是0 or 1的話，那就無法反映模型對於該class的信心程度，  
ex: 輸出向量 [貓, 狗, 鳥] = [1,1,0]中「貓」和「狗」都有 1，無法區分優先選擇哪一個類別。 

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
公式: $A\land B = A*B$  
則，  

$$A * B = 0.8 * 0.6 = 0.48$$

表示A和B同時發生的機率為48%，也就是說，在這個logic network中，事件 $𝐴∧𝐵$ 的發生程度僅為「部分成立」，不是完全的 0 或 1。

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
