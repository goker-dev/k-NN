#k-NN

###k-nearest neighbors algorithm

This is a Python k-NN sample. I coded this for my  second home work of Machine Learning.
It can recognize digits between 0-9 in [MNIST](http://yann.lecun.com/exdb/mnist/) (`digits.dat`) data set.
The data set has 64 features and a digit info on column 65.

And, I validated the results by k-fold cross validation.
I compared results for 1-NN, 3-NN and 5-NN with 10-fold.

Also, the sample has two more tiny data set (`simple.dat`, `test.dat`) for testing.

###Test data set

    1.1,1.1,1.1,A
    2.1,2.1,2.1,B
    2.5,2.5,2.5,C
    4.1,4.1,4.1,D
    5.1,5.1,5.1,E
    1.2,1.2,1.2,A
    2.2,2.2,2.2,B
    3.5,3.5,3.5,C
    4.2,4.2,4.2,D
    5.2,5.2,5.2,E
    1.3,1.3,1.3,A
    2.3,2.3,2.3,B
    3.3,3.3,3.3,C
    3.6,3.6,3.6,D
    5.3,5.3,5.3,E
    1.4,1.4,1.4,A
    2.4,2.4,2.4,B
    3.4,3.4,3.4,C
    4.4,4.4,4.4,D
    5.4,5.4,5.4,E

###Inputs

    k-NN :: k-nearest neighbors algorithm
    Data set file (default digits.dat):test.dat
    k value for k-fold cross validation (default 10):2
    k value for k-NN (default 1):1,3,5

###Outputs

    k-NN ==============================
     data set file is 'test.dat'
     2-fold data set and each part has  ['10', '10']  samples
     data set has 5 classes and in order ['A', 'B', 'C', 'D', 'E']
     ----------------------------------
     | Calculating 100 neighbors ...
     | Calculated in 0.0005 seconds
     ----------------------------------
     1-NN-----------------------------
     | Train set has 10 items
     | Test set [0] has 10 items
     ----------------------------------
        A  B  C  D  E
     A [2, 0, 0, 0, 0]
     B [0, 2, 1, 0, 0]
     C [0, 0, 0, 0, 0]
     D [0, 0, 1, 2, 0]
     E [0, 0, 0, 0, 2]
     Accuracy: 8 / 10 = 0.8

     Classified in 0.0003 seconds
     ----------------------------------


     3-NN-----------------------------
     | Train set has 10 items
     | Test set [0] has 10 items
     ----------------------------------
        A  B  C  D  E
     A [2, 0, 0, 0, 0]
     B [0, 2, 1, 0, 0]
     C [0, 0, 1, 0, 0]
     D [0, 0, 0, 2, 0]
     E [0, 0, 0, 0, 2]
     Accuracy: 9 / 10 = 0.9

     Classified in 0.0004 seconds
     ----------------------------------


     5-NN-----------------------------
     | Train set has 10 items
     | Test set [0] has 10 items
     ----------------------------------
        A  B  C  D  E
     A [2, 2, 0, 0, 0]
     B [0, 0, 1, 0, 0]
     C [0, 0, 1, 2, 0]
     D [0, 0, 0, 0, 2]
     E [0, 0, 0, 0, 0]
     Accuracy: 3 / 10 = 0.3

     Classified in 0.0005 seconds
     ----------------------------------


     ----------------------------------
     | Calculating 100 neighbors ...
     | Calculated in 0.0009 seconds
     ----------------------------------
     1-NN-----------------------------
     | Train set has 10 items
     | Test set [1] has 10 items
     ----------------------------------
        A  B  C  D  E
     A [2, 0, 0, 0, 0]
     B [0, 1, 0, 0, 0]
     C [0, 1, 2, 1, 0]
     D [0, 0, 0, 1, 0]
     E [0, 0, 0, 0, 2]
     Accuracy: 8 / 10 = 0.8

     Classified in 0.0002 seconds
     ----------------------------------


     3-NN-----------------------------
     | Train set has 10 items
     | Test set [1] has 10 items
     ----------------------------------
        A  B  C  D  E
     A [2, 0, 0, 0, 0]
     B [0, 2, 0, 0, 0]
     C [0, 0, 1, 0, 0]
     D [0, 0, 1, 2, 0]
     E [0, 0, 0, 0, 2]
     Accuracy: 9 / 10 = 0.9

     Classified in 0.0002 seconds
     ----------------------------------


     5-NN-----------------------------
     | Train set has 10 items
     | Test set [1] has 10 items
     ----------------------------------
        A  B  C  D  E
     A [2, 1, 0, 0, 0]
     B [0, 1, 0, 0, 0]
     C [0, 0, 2, 1, 0]
     D [0, 0, 0, 1, 2]
     E [0, 0, 0, 0, 0]
     Accuracy: 6 / 10 = 0.6

     Classified in 0.0002 seconds
     ----------------------------------


    Result ============================
     1-NN
        A  B  C  D  E
     A [4, 0, 0, 0, 0]
     B [0, 3, 1, 0, 0]
     C [0, 1, 2, 1, 0]
     D [0, 0, 1, 3, 0]
     E [0, 0, 0, 0, 4]
     Accuracy: 16 / 20 = 0.8

     3-NN
        A  B  C  D  E
     A [4, 0, 0, 0, 0]
     B [0, 4, 1, 0, 0]
     C [0, 0, 2, 0, 0]
     D [0, 0, 1, 4, 0]
     E [0, 0, 0, 0, 4]
     Accuracy: 18 / 20 = 0.9

     5-NN
        A  B  C  D  E
     A [4, 3, 0, 0, 0]
     B [0, 1, 1, 0, 0]
     C [0, 0, 3, 3, 0]
     D [0, 0, 0, 1, 4]
     E [0, 0, 0, 0, 0]
     Accuracy: 9 / 20 = 0.45

    Completed in 0.0042 seconds


##License
Absolutely, it is totally open source and under [MIT License](https://github.com/gokercebeci/ann/blob/master/LICENSE.md "MIT License")

##Bibliography
  *  [k-nearest neighbors algorithm, Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
  *  Jason Brownlee, September 12, 2014, [Tutorial To Implement k-Nearest Neighbors in Python From Scratch](http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)
