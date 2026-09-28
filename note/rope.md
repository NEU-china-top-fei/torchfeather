# ROPE
相对位置的好处:我们可以让模型外推,实际上我们也是想要让模型学到相对的位置关系而不是token在文本中的绝对位置   
[detailed node](https://neu-china-top-fei.github.io/2026/01/19/AIbasic/week5/)

## yarn
实际上rope也很难保证模型不会关注到绝对的位置信息
我们的模型总是倾向于学习某种固有的bias
yarn实际上并没有解决这个问题,它用于extend context window

某个位置的嵌入向量,它的不同维度的rotate 幅度也不一样
- 在较低的维度,rotate angle随着position增加是增长的很明显的
- 在较高的维度,增加非常不明显(low frequency=>high wavelength,这代表我们旋转一圈需要经历的position会更多)

引入波长的概念之后,我们可以计算预训练的时候某个dim能旋转多少圈
对low dim:每隔一段时间就会rotate到相同的位置,无法得知绝对位置
对high dim:整个序列都转不完,这样就会让模型推理的时候得知绝对位置


### why not use same for every dimention
theta大的时候,波长小,主要是用来区分邻近的token
theta小的时候,波长大,主要是用来区分间隔较远的token

所以我们不能直接对所有的dim都使用一个theta

### linear interpolation
模型在上述训练配置下,遇到比预训练的窗口更大的上下文的时候会perform bad
直接把theta除对应系数来cheat
比如训练是4096,实际是8192,直接把theta/2

这有些时候效果还行,但是也有问题,因为它把所有theta都放小了,这样高频减小,模型区分邻近token的能力可能减弱

### true yarn

$$
r(d)=\frac{L}{\lambda_D}=\frac{L}{2\pi b^{\frac{2d}{|D|}}}
$$
$$
\gamma(r)=\begin{cases}
0,& r<\alpha\\
1,& r>\beta\\
\frac{r-\alpha}{\beta-\alpha}
\end{cases}
$$

$$
h(\theta_d)=(1-\gamma(r(d)))\frac{\theta_d}{s}+\gamma(r(d))\theta_d
$$