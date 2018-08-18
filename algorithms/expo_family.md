## 指数分布族

如果某概率分布满足以下形式，就属于指数分布族。
$$
p(y;\eta )=b(y)exp(\eta ^{T}T(y)-a(\eta ))
$$

其中

$$
\eta     为参数向量
$$

$$
T(y)为充分统计量
$$

$$
exp^{-a(\eta )}起到归一化的作用
$$



## 伯努利分布：

$$
p(y;\phi )=\phi ^{y}(1-\phi )^{1-y}\\=exp[y\log \phi + (1-y)\log (1-\phi )]\\=exp[y\log \frac{\phi }{1-\phi } + \log (1-\phi )]
$$

把伯努利分布可以写成指数分布族的形式，且
$$
T(y)=y
\\\eta =\log \frac{\phi  }{1-\phi }
\\a(\eta )=-\log (1-\phi )=\log (1+e^{^{\eta }})
\\b(y)=1
$$
可以看到
$$
\phi =\frac{1}{1+e^{-\eta }}
$$
就是Logistic sigmoid的形式。



## 高斯分布：

$$
p(y;u)=\frac{1}{2\pi^{\frac{1}{2}}}*\exp (-\frac{1}{2}(y-u)^{2})
\\=\frac{1}{2\pi^{\frac{1}{2}}}\exp (-\frac{1}{2}y^{2})\exp (uy-\frac{1}{2}u^{2})
$$

对应的
$$
\eta =u
\\T(y)=y
\\a(\eta )=\frac{u^{2}}{2}
\\b(y)=\frac{1}{2\pi^{\frac{1}{2}}}\exp (-\frac{1}{2}y^{2})
$$



## 逻辑斯提回归：

LR是二类分类问题，可以选择伯努利分布：
$$
p(y/x;\theta )--Bernoulli(\theta )
$$
那么
$$
h_{\theta }(x)=E[y/x;\theta ]
\\=\phi 
\\=\frac{1}{1+e^{-\eta }}
\\=\frac{1}{1+e^{-\theta ^{T}x}}
$$
