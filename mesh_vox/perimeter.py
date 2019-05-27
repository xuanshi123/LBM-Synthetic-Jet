import math
from collections import defaultdict
import numpy as np
from numba import jit


def linesToVoxels(lineList, pixels):
    for x in range(len(pixels)):
        isBlack = False
        lines = list(findRelevantLines(lineList, x))
        targetYs = list(map(lambda line:int(generateY(line, x)), lines))
        for y in range(len(pixels[x])):
            if isBlack:
                pixels[x][y] = -1
            if y in targetYs:
                for line in lines:
                    if onLine(line, x, y):
                        isBlack = not isBlack
                        pixels[x][y] = -1
                        break

        if isBlack:
            print("an error has occured at x%sz%s"%(x,lineList[0][0][2]))


def findRelevantLines(lineList, x, ind=0):
    for line in lineList:
        same = False
        above = False
        below = False
        for pt in line:
            if pt[ind] > x:
                above = True
            elif pt[ind] == x:
                same = True
            elif pt[ind] < x:
                below = True

            if below and above:
                yield line
                break
            elif below and same:
                yield line
                break
            elif same and above:
                yield line
                break


# def linesToVoxels(lineList, pixels):
#
#     for x in range(len(pixels)):
#         isBlack = False
#
#         tf = np.zeros((len(lineList)), dtype=np.bool)
#
#         findRelevantLines(lineList, x, tf)
#
#         targetYs = list(map(lambda line:int(generateY(line,x)),lines))
#         for y in range(len(pixels[x])):
#             if isBlack:
#                 pixels[x][y] = -1
#             if y in targetYs:
#                 for line in lines:
#                     if onLine(line, x, y):
#                         isBlack = not isBlack
#                         pixels[x][y] = -1
#
#         if isBlack:
#             print("an error has occured at x%sz%s"%(x,lineList[0][0][2]))
#
#
# TPB = 16
#
# @cuda.jit
# def findRelevantLines(lineList, xi, tf):
#
#     x, y = cuda.grid(2)
#
#     tx = cuda.threadIdx.x
#     ty = cuda.threadIdx.y
#     bpg = cuda.gridDim.x
#
#     if x >= len(lineList):
#         # Quit if (x, y) is outside of valid C boundary
#         return
#
#     if y >= len(lineList[x]):
#         return
#
#     if lineList[x][y+1][0] > xi and lineList[x][y][0] <= xi:
#         tf[x] = True

def generateY(line, x):
    if line[1][0] == line[0][0]:
        return -1
    ratio = (x - line[0][0]) / (line[1][0] - line[0][0])
    ydist = line[1][1] - line[0][1]
    newy = line[0][1] + ratio * ydist
    return newy



def onLine(line, x, y):
    newy = generateY(line, x)
    if int(newy) != y:
        return False
    if int(line[0][0]) != x and int(line[1][0]) != x and (max(line[0][0], line[1][0]) < x or min(line[0][0], line[1][0]) > x):
        return False
    if int(line[0][1]) != y and int(line[1][1]) != y and (max(line[0][1], line[1][1]) < y or min(line[0][1], line[1][1]) > y):
        return False
    return True


