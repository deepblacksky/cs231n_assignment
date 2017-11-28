# Assignment 1
Details about this assignment can be found on [Assignment 1](https://cs231n.github.io/assignments2017/assignment1/).

We need to complete 4 kinds of **Image Classifier** on the [CIFAR 10](https://www.cs.toronto.edu/~kriz/cifar.html)
## kNN Image Classifier
Implement and apply a k-Nearest Neighbor (kNN) classifier

kNN(k-NearestNeighbor) is not very complicated algorithm. We just compute the L2 distance between the test image and train images. <br>
Complete [knn.ipynb](https://github.com/deepblacksky/cs231n_assignment/blob/master/assignment1/knn.ipynb)
There are three function to complete L2 distance.
`compute_distances_two_loops`,`compute_distances_one_loops`, `compute_distances_no_loops`.

In the `compute_distances_no_loops`, we don't use any "for_loop". And we should learn about matrix operation of `numpy`,
such as "broadcast sum".

## SVM
Implement and apply a Multiclass Support Vector Machine (SVM) classifier

SVM is a linear classifier. It is based on scoring results to classify.
Input sample ![](http://bit.ly/2ncgEkG),
the score for j-th class of ![](http://bit.ly/2ncgEkG)
is
![](http://www.sciweavers.org/tex2img.php?eq=s_j%20%3D%20f%28x_i%3B%20W%2C%20b%29_j%20%3D%20%28Wx_i%20%2B%20b%29_j&bc=White&fc=Black&im=jpg&fs=12&ff=mathptmx&edit=0)

About multiclass SVM, The i-th sample loss is:

![](http://www.sciweavers.org/tex2img.php?eq=L_i%3D%5Csum_%7Bj%20%5Cneq%20y_i%7Dmax%280%2Cs_j-s_%7By_i%7D%2B%5CDelta%29&bc=White&fc=Black&im=jpg&fs=12&ff=mathptmx&edit=0)

where ![]() is the margin(=1). The total loss:

![](http://www.sciweavers.org/tex2img.php?eq=L%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%7DL_i%20%2B%20%5Clambda%20%5Csum_%7Bk%7D%5Csum_%7Bl%7DW_%7Bk%2Cl%7D%5E%7B2%7D&bc=White&fc=Black&im=jpg&fs=12&ff=mathptmx&edit=0)

Next, we need to compute Gradient to optimize the parameter.
The Gradient formula is 

![](http://www.sciweavers.org/tex2img.php?eq=%5Cleft%5C%7B%5Cbegin%7Baligned%7D%0A%5Cnabla_%7Bw_%7By_i%7D%7D%20L_i%20%3D%20%26%20-%5Cleft%28%5Csum_%7Bj%20%5Cne%20y_i%7D1%28w_j%5ETx_i%20-%20w_%7By_i%7D%5ETx_i%20%2B%20%5CDelta%20%3E%200%29%5Cright%29x_i%20%26%20j%20%3D%20y_i%20%5C%5C%0A%5Cnabla_%7Bw_j%7D%20L_i%20%3D%20%26%201%28w_j%5ETx_i%20-%20w_%7By_i%7D%5ETx_i%20%2B%20%5CDelta%20%3E%200%29%20x_i%20%26%20j%20%5Cne%20y_i%0A%5Cend%7Baligned%7D%5Cright.&bc=White&fc=Black&im=jpg&fs=12&ff=mathptmx&edit=0)

complete [svm.ipynb](https://github.com/deepblacksky/cs231n_assignment/blob/master/assignment1/svm.ipynb)

## softmax
Implement and apply a Softmax classifier

It is samilier with SVM, the only one difference is loss function. Detials can read [linear classification notes](https://cs231n.github.io/linear-classify/). The is softmax loss (cross-entropy Loss) function:

![](http://www.sciweavers.org/tex2img.php?eq=L_i%20%3D%20-%20%5Clog%20%5Cleft%28%20%5Cfrac%7Be%5E%7Bf_%7By_i%7D%7D%7D%7B%5Csum_j%20e%5E%7Bf_j%7D%7D%20%20%5Cright%29%5Cquad%20%5Ctext%7Bor%20equivalently%7D%5Cquad%20L_i%20%3D%20-f_%7By_i%7D%20%2B%20%5Clog%5Csum_j%20e%5E%7Bf_j%7D&bc=White&fc=Black&im=jpg&fs=12&ff=mathptmx&edit=0)

where ![](http://www.sciweavers.org/tex2img.php?eq=f&bc=White&fc=Black&im=jpg&fs=12&ff=mathptmx&edit=0) is the score like SVM.
And Gradient is:

![](http://www.sciweavers.org/tex2img.php?eq=%5Cleft%5C%7B%5Cbegin%7Baligned%7D%0A%5Cnabla_%7Bw_%7By_i%7D%7D%20L_i%20%3D%20%26%20%28-1%20%2B%20%5Cfrac%7Be%5E%7Bf_%7By_i%7D%7D%7D%7B%5Csum_j%20e%5E%7Bf_j%7D%7D%20%29x_i%20%26%20j%20%3D%20y_i%20%5C%5C%0A%5Cnabla_%7Bw_j%7D%20L_i%20%3D%20%26%20%5Cfrac%7Be%5E%7Bf_j%7D%7D%7B%5Csum_j%20e%5E%7Bf_j%7D%7D%20x_i%20%26%20j%20%5Cne%20y_i%0A%5Cend%7Baligned%7D%5Cright.&bc=White&fc=Black&im=jpg&fs=12&ff=mathptmx&edit=0)

And when you complete [softmax.ipynb](https://github.com/deepblacksky/cs231n_assignment/blob/master/assignment1/softmax.ipynb) you can reference this note [neural-networks-case-study](https://cs231n.github.io/neural-networks-case-study/)

## Two Layer Neural Network
Implement and apply a Two layer neural network classifier

The key of this part is back propagation. The forward pass is samilier with Softmax. But, we should notice the activation function is ReLU.

About BP algorithm:
1. Use dimension analysis, do not derivation directly
2. Use the chain rules, do not step in place

You can refer to these two articles.

[](http://cs231n.stanford.edu/handouts/derivatives.pdf)

[](https://zhuanlan.zhihu.com/p/25202034)

complete [two_layer_net.ipynb](https://github.com/deepblacksky/cs231n_assignment/blob/master/assignment1/two_layer_net.ipynb)

## Image Feature
Get a basic understanding of performance improvements from using higher-level representations than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)

complete [features.ipynb](https://github.com/deepblacksky/cs231n_assignment/blob/master/assignment1/features.ipynb)

