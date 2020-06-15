# Palmprint_Recognition
This project is mainly to complete the palmprint feature extraction and classification tasks. The data set contains 99 people's palm print pictures, in which 3 palm print pictures of each person are distributed in the training set, and the other 3 palm print pictures are distributed in the test set. In this project, I tried the traditional method use SIFT to extract features and KNN for classification which get accuracy of 97.31%, and also tried the convolutional neural network method such as ResNet which get accuracy of 83.16%. In addition, I also tried to use the Gaussian filter, Gabor filter,etc. to process the palmprint image and extract the texture from the palmprint image, but these methods have not improved the accuracy of palmprint recognition.

## 参考博客：
[【Pytorch】使用ResNet-50迁移学习进行图像分类训练](https://blog.csdn.net/heiheiya/article/details/103028543)

[【pytorch】数据增强](https://wizardforcel.gitbooks.io/learn-dl-with-pytorch-liaoxingyu/4.7.1.html)

[opencv python SIFT（尺度不变特征变换）](https://segmentfault.com/a/1190000015709719)

[OpenCV-Python教程:41.特征匹配](https://www.jianshu.com/p/ed57ee1056ab)

[opencv python 特征匹配](https://segmentfault.com/a/1190000015735549)

[opencv中 cv2.KeyPoint和cv2.DMatch的理解](https://blog.csdn.net/qq_29023939/article/details/81130987)

[K近邻算法](https://www.cnblogs.com/ybjourney/p/4702562.html)
