- 区分:嵌入应该是专门的嵌入层查表,所以形状是(vocabsize,dim)
- 最后的output应该是映射回去(dim,vocabsize)

开始的嵌入和最后的权重共享:可以做也可以不做


register_buffer:注册跟随模型但是不参与训练的参数


RMSnorm:
$$
\text{output}=\gamma \times \frac{x}{std}
$$