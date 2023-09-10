## 决策树算法（Decision Tree Algorithm）

决策树（decision tree）是一种基本的分类与回归方法，决策树包含三个步骤：决策树生成、特征选择和决策树剪枝。

### 一、信息论

#### 1.1、信息熵的概念

离散型随机变量X的取值为x_1,x_2,x_3,...,x_n，其发生概率分别为p1,p2,p3,...pn，那么信息熵定义为
$$
H(X) = -\sum_{i=1}^np_ilog(p_i)
$$
一般对数的底数为2，也可以换成e，当对数底数为2时，信息熵的单位为**比特**。

#### 1.2、条件熵

假设随机变量（X,Y）具有联合概率分布：
$$
P(X=x_i,Y=y_i) = p_{ij}, \quad i= 1,2,...,n \quad j=1,2,...,m
$$
条件熵H(Y|X)表示在已知随机变量X的条件下随机变量Y的不确定性，(X,Y)同时发生包含的不确定性，减去X单独发生的不确定性，就是在X发生的前提下，Y发生新带来的不确定性。不确定性即为熵。

所以条件熵有如下公式成立：
$$
H(Y|X) = H(X,Y) - H(X)
$$
推导过程如下:
$$
H(X,Y) - H(X) \\
= -\sum_{x,y}^np(x,y)logp(x,y) + \sum_{x}p(x)logp(x) \\
= -\sum_{x,y}^np(x,y)logp(x,y) + \sum_{x}p(x) \sum_{y}p(y)logp(x) \\
= -\sum_{x,y}logp(x,y) + \sum_{x,y}p(x,y)logp(x) \\
= -\sum_{x,y}p(x,y)log\frac{p(x,y)}{p(x)} \\
= -\sum_{x,y}p(x,y)logp(y|x)
$$

#### 1.3、相对熵

在信息理论中，相对熵(KL散度)等价于两个分布的信息熵的差值
$$
KL(P||Q) = -\sum_{x\in X}P(x)log\frac{1}{P(x)} + \sum_{x\in X}P(x)log\frac{1}{Q(x)} = \sum_{x\in X}P(x)log\frac{P(x)}{Q(x)}
$$

#### 1.4、互信息

两个随机变量X和Y的互信息，定义为X、Y的联合分布和独立分布的相对熵，即
$$
I(X,Y) = KL(P(X,Y) || P(X)P(Y))
$$
所以根据KL散度，也就是相对熵的定义，可以推出互信息表达式如下：
$$
I(X,Y) = \sum_{x,y}p(x,y)log[p(x,y)/(p(x)p(y))]
$$
继续推导:
$$
H(Y) - I(X,Y) \\
= -\sum_{y}p(y)logp(y) - \sum_{x,y}p(x,y)log\frac{p(x,y)}{p(x)p(y)} \\
= -\sum_{y}(\sum_{x}p(x,y))logp(y) - \sum_{x,y}p(x,y)log\frac{p(x,y)}{p(x)p(y)}\\
= -\sum_{x,y}p(x,y)logp(y) - \sum_{x,y}p(x,y)log\frac{p(x,y)}{p(x)p(y)} \\
= -\sum_{x,y}p(x,y)log\frac{p(x,y)}{p(x)} \\
= -\sum_{x,y}p(x,y)logp(y|x) \\
= H(Y|X)
$$
因此有:
$$
I(X,Y) = H(Y) - H(Y|X)
$$
结合上述条件熵的表达式，可进一步推出：
$$
H(Y|X) = H(X,Y) - H(X) \\
H(X|Y) = H(X,Y) - H(Y) \quad 1 \\
H(Y|X) = H(Y) - I(X,Y) \\
H(X|Y) = H(X) - I(X,Y)  \quad 2 \\
由1和2可得 \quad I(X,Y) = H(X)+H(Y) - H(X,Y)
$$
同时存在两个不等式
$$
H(X|Y) \leq H(X), H(Y|X) \leq H(Y)
$$
不等式可以理解为**对于一个与X相关的随机变量Y，只要我们得知了一点关于Y的信息，那么X的不确定度就会减小**

![](entropy.png)

### 二、决策树

#### 2.1、案例

为了说明决策树的算法流程，我们给出一个具体的数据案例。

![](tree_data.png)

构造决策树如下：

![](tree_data2.png)

第一层根节点 被分成14份，9是/5否，总体的信息熵为：
$$
H_0 = -[5/14log5/14+9/14log9/14] = 0.9403
$$
第二层 晴：被分为5份，2是/3否，它的信息熵为：
$$
H_1 = -[2/5log2/5+3/5log3/5] = 0.9710 \\
H_2 = -[log1] = 0 \\
H_3 = -[2/5log2/5+3/5log3/5] = 0.9710
$$
假设我们选取天气为分类依据，把它作为根节点，那么第二层的加权信息熵可以定义为：
$$
H^‘ = 5/14*H_1 + 4/14H_2 +5/14H_3 = 0.6936
$$
H^’必须比H0小，随着决策进行，其不确定性程度减小才可以，即决定一定是从一个不确定性到确定性的过程，因此H^’=0.6936 < H0 = 0.9403 符合要求。

**事实上，随着分类的进行，越来越多的信息被知道，那么总体的熵肯定是会下降的。**

同样，对于晴这个节点，它的两个叶子结点的熵都是0，到了叶子结点之后，熵就变为0了，就得到了决策结果。

**因此，决策树采用的是自顶向下的递归方法，其基本思想是以信息熵为度量构造一棵熵值下降最快的树，到叶子节点处的熵值为0，此时每个叶子节点中的实例都属于同一类。**

怎么定义下降最快,下面讲解:

#### 2.2、决策树生成算法

首先我们要选择一个根节点，那么选谁当做根节点呢？比如上面的例子，有天气，湿度以及风级三个属性，所以我们要在三个当中选择一个。三个属性f1,f2,f3，以三个属性分别为根节点可以生成三棵树（从第一层到第二才层），而究竟选择谁来当根节点的准则，有以下三种。

##### 2.2.1、信息增益与ID3

给定一个样本集D，划分前样本集合D的熵是一定的 ，用H0表示； 使用某个特征A划分数据集D，计算划分后的数据子集的熵，用H1表示，则： **信息增益**可以表示为H0−H1，也可以表示为：
$$
g(D,A) = H(D) - H(D|A) = I(D,A)
$$
比如上面实例中我选择天气作为根节点，将根节点一分为三，设f1表示天气，则有:
$$
g(D,A) = H_0 - H^‘ = 0.9403 - 0.6936 = 0.2467
$$
**意思是，没有选择特征f1前，是否去打球的信息熵为0.9403，在我选择了天气这一特征之后，信息熵下降为0.6936，信息熵下降了0.2467，也就是信息增益为0.2467.**

**信息增益的局限：信息增益偏向取值较多的特征**。**原因：当特征的取值较多时，根据此特征划分更容易得到纯度更高的子集，因此划分之后的熵更低，由于划分前的熵是一定的，因此信息增益更大，因此信息增益比较偏向取值较多的特征。**

比如说有一个特征可以把训练集的每一个样本都当成一个分支，也就说有n个样本，该特征就把树分成了n叉树，那么划分后的熵变为0，此时信息增益当然是下降最大的。 也就是说，如果我们在生成决策树的时候以信息增益作为判断准则，那么分类较多的特征会被优先选择。

##### 2.2.2、信息增益率与C4.5

为了解决信息增益的局限，引入了信息增益率的概念。分支过多容易导致过拟合，造成不理想的后果。假设原来的熵为0.9，选择f1特征划分后整体熵变成了0.1，也就是信息增益为0.8，而选择f2划分后，熵变为0.3，也就是信息增益为0.6。我们不想选择f1，因为它让决策树分支太多了，那么就可以定义如下决策指标：
$$
g_r(D,A) = g(D,A) / H(A)
$$
这就是信息增益率，**信息增益率=信息增益/特征本身的熵**。f1划分后分支更多，也就是特征f1本身的熵比f2更大，**大的数除以一个大的数，刚好可以中和一下。**

这个时候我们考虑天气本身的熵，这里算的是天气本身的熵，**而不是样本X,也就是是否外出打球的熵**，这里一定要将二者区分开。天气本身有三种可能，每种概率都已知，则天气的熵为：
$$
H(f_1) = -[5/14log5/14+4/14log4/14+5/14log5/14] = 1.5774
$$
那么选择天气作为分类依据时，信息增益率为：
$$
g_r(D,A) = g(D,A) / H(A) = 0.2467 / 1.5774 = 0.1566
$$
**利用信息增益率作为选择指标来生成决策树的算法称为C4.5算法。**

##### 2.2.3 Gini系数与CART

定义：基尼指数（基尼不纯度）：**表示在样本集合中一个随机选中的样本被分错的概率。**
$$
Gini(p) = \sum_{k=1}^K p_k(1-p_k) = 1-\sum_{k=1}^Kp_k^2 = 1-\sum_{k=1}^K(\frac{|C_k|}{D})^2
$$
一些参数的说明：

1. pk表示选中的样本属于k类别的概率，则这个样本被分错的概率是(1-pk)。
2. 样本集合中有K个类别，一个随机选中的样本可以属于这k个类别中的任意一个。
3. 易知，当样本属于每一个类别的概率都相等即均为1/K时，基尼系数最大，也就是说此时不确定度最大。

关于基尼系数的理解，网上有一种说法比较通俗易懂。现解释如下： 我们知道信息熵的定义式为： 
$$
H(X) = -\sum_{i=1}^n p_ilogp_i
$$
那么基尼系数实际上就是用1−pi来代替了-logpi，画出二者图像：

![](tree_data3.png)

因为概率是属于0到1之间，所以我们只看01区间上的图像：**基尼系数对于信息熵而言，就是在01区间内近似的用切线来代替了对数函数。因此，既然信息熵可以表述不确定度，那么基尼系数自然也可以，只不过存在一些误差。**

CART决策树又称分类回归树，当数据集的因变量为连续性数值时，该树算法就是一个回归树，可以用叶节点观察的均值作为预测值；当数据集的因变量为离散型数值时，该树算法就是一个分类树，可以很好的解决分类问题。 当CART是分类树时，采用GINI值作为结点分裂的依据；当CART是回归树时，采用MSE(均方误差)作为结点分裂的依据。我们这里只讨论分类。

##### 2.3 决策树的评价

若某一叶子结点中包含了所有类别的样本且各类数目相同，则称该结点为**均结点。** 其熵为logK。**那么我们对所有叶子结点的熵求和，该值越小说明越精确**，而又由于各个叶子结点包含的样本数目不同，所以我们采用加权熵和。评价函数定义如下：
$$
C(T) = \sum_{t \in leaf} N_t H(t)
$$
其中N_t表示叶子结点的样本数目，**评价函数越小说明决策树越好。**完美分类结果C(T)等于0。

##### 2.4 决策树的过拟合

当决策树深度过大时，在训练集上表现特别好，往往就会出现过拟合现象，我们需要一些解决办法：

剪枝：由完全树T0开始，剪枝部分结点得到树T1，然后再剪枝部分结点得到树T2，...，直到仅剩树根的Tk。在验证数据集上对这K个树分别评价，选择损失函数最小的树Tα。

**三种决策树的生成算法过程相同，只是对于当前树的评价标准不同。**

### 三、随机森林

随机森林也是为了解决决策树的过拟合问题。

#### 3.1 Bootstrap

假设有一个大小为N的样本，我们希望从中得到m个大小为N的样本用来训练。bootstrap的思想是：首先，在N个样本里随机抽出一个样本x1，然后记下来，放回去，再抽出一个x2，… ，这样重复N次，即可得到N个新样本，**这个新样本里可能有重复的**。重复m次，就得到了m个这样的样本。实际上就是一个有放回的随机抽样问题。每一个样本在每一次抽的时候有同样的概率（1/N）被抽中。

#### 3.2 bagging策略

bagging的名称来源于： Bootstrap Aggregating，意为自助抽样集成。既然出现了Bootstrap那么肯定就会使用到Bootstrap方法，其基本策略是：

1. 利用Bootstrap得到m个样本大小为N的样本集。
2. 在所有属性上，对每一个样本集建立分类器。
3. 将数据放在这m个分类器上，最后根据m个分类器的投票结果，决定数据最终属于哪一类。如果是回归问题，就采用均值。

什么时候用bagging？当模型过于复杂容易产生过拟合时，才使用bagging，决策树就容易产生过拟合。

#### 3.3 out of bag estimate（包外估计）

在使用bootstrap来生成样本集时，由于我们是有放回抽样，那么可能有些样本会被抽到多次，而有的样本一次也抽不到。我们来做个计算：假设有N个样本，每个样本被抽中的概率都是1/N，没被选中的概率就是1-1/N，重复N次都没被选中的概率就是(1−1/N)^N，当N趋于无穷时，这个概率就是1/e，大概为36.8%。也就是说样本足够多的时候，一个样本没被选上的概率有36.8%，那么这些没被选中的数据可以留作**验证集**。每一次利用Bootstrap生成样本集时，其验证集都是不同的。

**以这些没被选中的样本作为验证集的方法称为包外估计。**

#### 3.4 样本随机与特征随机

在我们使用Bootstrap生成m个样本集时，每一个样本集的样本数目不一定要等于原始样本集的样本数目，比如我们可以生成一个含有0.75N个样本的样本集，此处0.75就称为采样率。

同样，我们在利用0.75N个样本生成决策树时，假设我们采用ID3算法，生成结点时以信息增益作为判断依据。我们的具体做法是把每一个特征都拿来试一试，最终信息增益最大的特征就是我们要选的特征。但是，我们在选择特征的过程中，也可以只选择一部分特征，比如20个里面我只选择16个特征。 那可能有的人就要问了，假设你没选的4个特征里面刚好有一个是最好的呢？这种情况是完全可能出现的，但是我们在下一次的分叉过程中，该特征是有可能被重新捡回来的，另外别的决策树当中也可能会出现那些在另一颗决策树中没有用到的特征。

随机森林的定义就出来了，**利用bagging策略生成一群决策树的过程中，如果我们又满足了样本随机和特征随机，那么构建好的这一批决策树，我们就称为随机森林(Random Forest)。**

实际上，我们也可以使用SVM，逻辑回归等作为分类器，这些分类器组成的总分类器，我们习惯上依旧称为**随机森林**。