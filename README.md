# MNIST-Data-Classification-Problem
Using different classification algorithms, Logistic regression, Neural Network, SVM, Random Forest) on MNIST data for predicting digits.

Problem Statement-:
The problem is about recognizing the 28 X 28 handwritten digit images of gray scale, known as the MNIST data set, making it a 10 -class classification problem (digit can be from 0 to 9). This problem is a supervised learning problem and I have used 4 different classifiers on the same data set for training the model and predicting the outputs. 
 
About Dataset-: 
We have an MNIST data set, with the size of each image as 28 * 28, which has got 784 features for each sample. Training has to be done using 50000 records for MNIST data set, followed by cross validation and testing on the trained model with 10000 records each. With the above data set we have been given the USPS dataset, which has to be used for testing on the MNIST model . 
 
Classifiers -: 

1) Logistic Regression-: 
It is a classification algorithm used for predicting the output, that belongs to discreet set of classes. In our case we have 10 classes, as the predicted digit can be anything from 0-9. The loss function being used for logistic is the cross entropy loss function and function being used for mapping the prediction to the probability is softmax. The formula for softmax is given below-: 
 

2) Support Vector Machine-: 
It is actually a binary classification algorithm, but using the technique of one vs all, can be used for multiclass classification problem. Unlike 
logistic regression algorithm finds the largest separating margin. The loss function used for SVM is Hinge loss function. 
 
For tuning I have used SVM API from the sklearn library with gamma as  value 1, which highlights the spread of the rbf kernel. More is the value less is the spread and vice versa. I have used SVM with 2 different kernels ie rbf(radial basis function) and linear kernel. The best accuracy was achieved upon the rbf kernel with default configurations  
 
3)  Neural networks-:
It is a famous and most accurate classification algorithm. I have used  ANN (Artifical Neural Network) as the 4 layered network, having the softmax activation at the output layer(for multiclass problem) and using relu activation function on the hidden layers. The output from the neural network would be a hot encoded vector. 
I have used keras library for implementing the neural network classifier, which is backed by the tensorflow library. The loss function used for ANN is the categorical_entropy loss function and optimizer used is Adam. 4) Random Forest Algorithm-: Random forest is a supervised learning classification algorithm which builds multiple decision trees  and merges them together to form a more accurate and stable prediction. In this algorithm decision trees are made for different subsets of the data, and the leaf nodes of the trees give the decisions. 
I have used RandomForestClassifier from sklearn library to implement the above algorithm. The value of n_estimators=1000, highlights the number of decision trees to be used in forest

4) Random Forest Algorithm-: 
Random forest is a supervised learning classification algorithm which builds multiple decision trees  and merges them together to form a more accurate and stable prediction. In this algorithm decision trees are made for different subsets of the data, and the leaf nodes of the trees give the decisions. 
I have used RandomForestClassifier from sklearn library to implement the above algorithm. The value of n_estimators=1000, highlights the number of decision trees to be used in forest. 
