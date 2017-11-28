# Assignment 1
Details about this assignment can be found on [Assignment 1](https://cs231n.github.io/assignments2017/assignment1/).

We need to complete 4 kinds of **Image Classifier** on the [CIFAR 10](https://www.cs.toronto.edu/~kriz/cifar.html)
## kNN Image Classifier
kNN(k-NearestNeighbor) is not very complicated algorithm. We just compute the L2 distance between the test image and train images. <br>
Complete [knn.ipynb](https://github.com/deepblacksky/cs231n_assignment/blob/master/assignment1/knn.ipynb)
There are three function to complete L2 distance.
`compute_distances_two_loops`,`compute_distances_one_loops`, `compute_distances_no_loops`.

In the `compute_distances_no_loops`, we don't use any "for_loop". And we should learn about matrix operation of `numpy`,
such as "broadcast sum".

## SVM
SVM is a linear classifier. It is based on scoring results to classify.
Input sample ![](http://chart.googleapis.com/chart?cht=tx&chl=x_i),
the score for ![](http://chart.googleapis.com/chart?cht=tx&chl=j) class of ![](http://chart.googleapis.com/chart?cht=tx&chl=x_i)
is

The Loos Function is:

![](http://www.sciweavers.org/tex2img.php?eq=L_i%3D%5Csum_%7Bj%20%5Cneq%20y_i%7Dmax%280%2Cs_j-s_%7By_i%7D%2B%5CDelta%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

ff
