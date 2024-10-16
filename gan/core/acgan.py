# -*- coding: utf-8 -*-
# @Time    : 2024/10/16 18:13
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

"""
辅助分类生成对抗网络
(Auxiliary Classifier GAN, ACGAN)

关键优化：
1.条件生成器与判别器：生成器在噪声向量的基础上加上了嵌入标签，判别器不仅要判断图片真伪，
  还要预测生成图片所属的类别。
2.嵌入层优化：通过嵌入层（nn.Embedding）对类别标签进行嵌入，使得标签与噪声向量可以自然地结合。
3.辅助分类损失：在判别器中加入辅助分类器(aux_layer),通过交叉熵损失来增强生成图像的类别一致性
"""