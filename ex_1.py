import sys
import numpy
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

sample, centroids = sys.argv[1], sys.argv[2]
fs, y = scipy.io.wavfile.read(sample)
pointsList = np.array(y.copy())

NumOfIterations = 30
centroids = np.loadtxt(centroids)

if centroids.shape == (2,):
    centroids = centroids.reshape((1,2))
kMeansArr = [len(centroids)]


def main():
    for i in kMeansArr:
        kMeansAlg(i)


def kMeansAlg(k):
    clusters = {}
    lossArray = []
    text_file = open("output.txt", "w")
    for i in range(NumOfIterations):
        loss = 0
        clusters = {new_list: [] for new_list in range(k)}
        for point in pointsList:
            # find nearest centroid.
            result, loss = findNearestCentroid(point, centroids, loss)
            # assign the point x to cluster j.
            clusters[result].append(point)
        if len(pointsList != 0):
            loss = loss / len(pointsList)
            lossArray.append(loss)
        for l in range(k):
            firstSum = 0
            secondSum = 0
            for num in clusters[l]:
                firstSum += num[0]
                secondSum += num[1]
            if len(clusters[l]) != 0:
                firstSum /= len(clusters[l])
                firstSum = round(firstSum)
            if len(clusters[l]) != 0:
                secondSum /= len(clusters[l])
                secondSum = round(secondSum)
            if firstSum == 0 and secondSum == 0:
                continue
            centroids[l] = [firstSum, secondSum]
        if i != 0:
            if lossArray[i - 1] == loss:
                break
        printIteration(i, centroids,text_file)
    text_file.close()


def printIteration(i, centroidim,text_file):
    text_file.write(f"[iter {i}]:{','.join([str(i) for i in centroidim])}\n")


def findNearestCentroid(numOfPoint, centroidsList, loss):
    minDistance = sys.float_info.max
    minIndex = 0
    for index in range(len(centroidsList)):
        closetDistance = numpy.linalg.norm(numOfPoint - centroidsList[index])
        if closetDistance < minDistance:
            minIndex = index
            minDistance = closetDistance
    loss += minDistance
    return minIndex, loss


if __name__ == '__main__':
    main()