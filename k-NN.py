import csv
import math
import operator
import time


def loadDataset(filename, k):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataSet = list(lines)
        columnCount = len(dataSet[0])
        lineCount = len(dataSet)
        classes = set()
        # set features as a float number except class
        for sampleIndex in range(lineCount):
            for featureIndex in range(columnCount - 1):
                dataSet[sampleIndex][featureIndex] = float(dataSet[sampleIndex][featureIndex])
            # collect  classes from data set
            classes.add(dataSet[sampleIndex][-1])
        # split data set as k parts array for k value of k-fold
        partSize = lineCount / k
        partionedData = []
        for x in range(k):
            start = x * partSize
            end = lineCount if x == k - 1 else (start + partSize)
            partionedData.append(dataSet[start:end])
        return sorted(classes, key=str), partionedData


def setDatasets(dataSets, testSetIndex):
    trainingSet = []
    testSet = []
    # set test set and merge training set
    for partIndex in range(len(dataSets)):
        if partIndex == testSetIndex:
            testSet = dataSets[partIndex]
        else:
            trainingSet += dataSets[partIndex]
    return trainingSet, testSet


def euclideanDistance(sample1, sample2):
    distance = 0
    # except the last column which is the class information (-1)
    for feature in range(len(sample1) - 1):
        distance += pow((sample1[feature] - sample2[feature]), 2)
    return math.sqrt(distance)


def getNeighbors(testSet, trainingSet):
    print ' ----------------------------------'
    print ' | Calculating %d neighbors ...' % (len(testSet) * len(trainingSet))
    start = time.time()
    neighbors = [[]] * len(testSet)
    for testSample in range(len(testSet)):
        distances = []
        for sample in range(len(trainingSet)):
            distances.append([trainingSet[sample], euclideanDistance(testSet[testSample], trainingSet[sample])])
        neighbors[testSample] = sorted(distances, key=operator.itemgetter(1))
    print " | Calculated in %.4f seconds" % (time.time() - start)
    print ' ----------------------------------'
    return neighbors


def getClass(neighbors, classes):
    classVotes = {}
    for x in range(len(neighbors)):
        # get class info from last column
        result = classes.index(neighbors[x][0][-1])
        # vote classes
        if result in classVotes:
            classVotes[result] += 1
        else:
            classVotes[result] = 1
    # sort classes descending order and the biggest is the result
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(matrix):
    t = 0
    f = 0
    for column in range(len(matrix)):
        for row in range(len(matrix[column])):
            if column == row:
                t += matrix[column][row]
            else:
                f += matrix[column][row]
    accuracy = (float(t) / float(t + f)) if t > 0 else 0
    print ' Accuracy: ' + repr(t) + ' / ' + repr(t + f) + ' = ' + repr(accuracy)


def getConfusionMatrix(testSet, predictions, classes):
    matrix = [[0] * len(classes) for x in range(len(classes))]
    for x in range(len(testSet)):
        matrix[int(predictions[x])][classes.index(testSet[x][-1])] += 1
    return matrix


def matrixMerge(matrixes):
    matrix = [[0] * len(matrixes[0][0]) for x in range(len(matrixes[0]))]
    for x in range(len(matrixes)):
        for column in range(len(matrixes[x])):
            for row in range(len(matrixes[x][column])):
                matrix[column][row] += int(matrixes[x][column][row])
    return matrix


def printMatrix(matrix, classes):
    print '  ',
    for x in range(len(matrix[1])):
        print ' ' + str(classes[x]),
    print
    for x, element in enumerate(matrix):
        print ' ' + str(classes[x]), "".join(str(element))
    getAccuracy(matrix)
    print


def kNN(i, k, trainingSet, testSet, neighbors, classes):
    start = time.time()
    predictions = []
    print ' %d-NN-----------------------------' % k
    print ' | Train set has %d items' % len(trainingSet)
    print ' | Test set [%d] has %d items' % (i, len(testSet))
    print ' ----------------------------------'
    for sampleIndex in range(len(testSet)):
        # cut neighbors on k th value [:k]
        result = getClass(neighbors[sampleIndex][:k], classes)
        predictions.append(result)
    matrix = getConfusionMatrix(testSet, predictions, classes)
    printMatrix(matrix, classes)
    print " Classified in %.4f seconds" % (time.time() - start)
    print ' ----------------------------------'
    print
    print
    return matrix


def main():
    print 'k-NN :: k-nearest neighbors algorithm'
    input1 = raw_input('Data set file (default digits.dat):').lower()
    input1 = input1.replace(" ", "") if input1 else "digits.dat"
    input2 = raw_input('k value for k-fold cross validation (default 10):')
    input2 = int(input2) if input2 else 10
    input3 = raw_input('k value for k-NN (default 1):')
    input3 = input3.replace(" ", "").split(',') if input3 else [1]
    start = time.time()
    classes, dataSets = loadDataset(input1, input2)
    print 'k-NN =============================='
    print ' data set file is ' + repr(input1)
    print ' %d-fold data set and each part has ' % input2,
    print ['%d' % len(e) for e in dataSets],
    print ' samples'
    print ' data set has ' + repr(len(classes)) + ' classes and in order ' + repr(classes)
    matrixFold = [[[] for x in range(len(dataSets))] for x in range(len(input3))]
    for dataSetIndex in range(len(dataSets)):
        trainingSet, testSet = setDatasets(dataSets, dataSetIndex)
        neighbors = getNeighbors(testSet, trainingSet)
        for x in range(len(input3)):
            matrixFold[x][dataSetIndex] = kNN(dataSetIndex, int(input3[x]), trainingSet, testSet, neighbors, classes)
    print 'Result ============================'
    for x in range(len(input3)):
        matrix = matrixMerge(matrixFold[x])
        print " %d-NN" % int(input3[x])
        printMatrix(matrix, classes)
    print "Completed in %.4f seconds" % (time.time() - start)


main()