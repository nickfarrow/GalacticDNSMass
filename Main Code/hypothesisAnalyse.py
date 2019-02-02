import sys
import os
import matplotlib
import datetime
matplotlib.use('Agg')     


import numpy as np
from numpy import exp, log
import matplotlib.pyplot as plt
import sys, os
import json
import pymultinest




modelSetB = [['singleGaussian', 'singleGaussian', 2, 2, ['mu', 'sigma', 'mu', 'sigma']],
            ['singleGaussian', 'twoGaussian', 2, 5, ['mu', 'sigma', 'mu1', 'mu2', 'sigma1', 'sigma2', 'alpha']],
            ['singleGaussian', 'uniform', 2, 2, ['mu', 'sigma', 'mMin', 'mMax']],
            ['twoGaussian', 'singleGaussian', 5, 2, ['mu1', 'mu2', 'sigma1', 'sigma2', 'alpha', 'mu', 'sigma']],
            ['twoGaussian', 'twoGaussian', 5, 5, ['mu1', 'mu2', 'sigma1', 'sigma2', 'alpha', 'mu1', 'mu2', 'sigma1', 'sigma2', 'alpha']],
            ['twoGaussian', 'uniform', 5, 2, ['mu1', 'mu2', 'sigma1', 'sigma2', 'alpha', 'mMin', 'mMax']],
            ['uniform', 'singleGaussian', 2, 2, ['mMin', 'mMax', 'mu', 'sigma']],
            ['uniform', 'twoGaussian', 2, 5, ['mMin', 'mMax', 'mu1', 'mu2', 'sigma1', 'sigma2', 'alpha']],
            ['uniform', 'uniform', 2, 2, ['mMin', 'mMax', 'mMin', 'mMax']]]


modelSetA = [['singleGaussian', 2, ['mu', 'sigma']],
            ['twoGaussian', 5, ['mu1', 'mu2', 'sigma1', 'sigma2', 'alpha']],
            ['uniform', 2, ['mMin', 'mMax']]]

def roundNumber(number):
    return round(number, 4)


def cleanRound(number, dec=3):
    newNum = str(round(number, 3))
    print(newNum)
    missing = dec - len(newNum.split('.')[1])
    return newNum + '0' * missing



def readStats(statsFile):
    globalEvidenceLine = statsFile.readline()
    nestedGlobalEvidenceLine = statsFile.readline()
    
    globalEvidenceLine
    
    print(globalEvidenceLine.split())
    globalEvidence = float(globalEvidenceLine.split()[-3])
    print(globalEvidenceLine.split()[-1])
    evidenceStd = float(globalEvidenceLine.split()[-1])
    print(globalEvidence, evidenceStd)
    
    return globalEvidence, evidenceStd


def fractionalEvidence(resultList, resultDirectory, saveName="fracResults.txt"):
    with open(resultDirectory + saveName,"w+") as z:
        
        totalEv = 0
        for result in resultList:
            totalEv += result[-2]
        
        
        ### Hypo B
        if len(resultList[0]) == 4:
            for result1 in resultList:
                tb = ' & '
                tableString = '    ' + result1[0][:4] + '-' + result1[1][:4] + tb + str(result1[2]/totalEv) + "\\" * 2
                z.write(tableString + '\n')
                
        else:
            for result1 in resultList:
                tb = ' & '
                tableString = '    ' + result1[0][:4] + tb + str(result1[1]/totalEv) + "\\" * 2
                z.write(tableString + '\n')
    return



def makeTableB(resultList, resultDirectory, saveName="tableResults.txt"):
    with open(resultDirectory + saveName,"w+") as z:
        
        reverseNameTitle = '    ' * 4 + ' & '.join([result[0][:4] + '-' + result[1][:4] for result in resultList[::-1]]) + "\\" * 2
        z.write(reverseNameTitle + '\n')
        
        for index, result1 in enumerate(resultList):
            print(index, result1)
            evidence1 = result1[2]

            lineBFs = []
            #for result2 in resultList[:-index - 1]:
            for result2 in resultList[:index:-1]:
                evidence2 = result2[2]
                bayesFactor = evidence1/evidence2
                lineBFs.append(cleanRound(bayesFactor))
            
            tb = ' & '
            tableString = '    ' + result1[0][:4] + '-' + result1[1][:4] + tb + tb.join(lineBFs) + "\\" * 2
            z.write(tableString + '\n')
    return



def makeTableA(resultList, resultDirectory, saveName="tableResults.txt"):
    with open(resultDirectory + saveName,"w+") as z:
        
        reverseNameTitle = '    ' * 3 + ' & '.join([result[0][:4] for result in resultList[::-1]]) + "\\" * 2
        z.write(reverseNameTitle + '\n')
        
        for index, result1 in enumerate(resultList):
            print(index, result1)
            evidence1 = result1[1]

            lineBFs = []
            #for result2 in resultList[:-index - 1]:
            for result2 in resultList[:index:-1]:
                evidence2 = result2[1]
                bayesFactor = evidence1/evidence2
                lineBFs.append(cleanRound(bayesFactor))
            
            tb = ' & '
            tableString = '    ' + result1[0][:4] + tb + tb.join(lineBFs) + "\\" * 2
            z.write(tableString + '\n')
    return


    
    

def analyseMainB(resultDirectory):
    with open(resultDirectory + "collectedResults.txt","w+") as g:
        g.write("Model1 ------ Model2 ------ Evidence ------ LogEvidence std\n")
        resultsList = []
        for modelName1, modelName2, ndim1, ndim2, paramNames in modelSetB:
            ndim = ndim1 + ndim2
            print(modelName1, modelName2)
            
            prefix = resultDirectory + modelName1[:4] + "/" + modelName2[:4] + "/"
            
            with open(prefix + "stats.dat","r") as statsFile:
                logevidence, evidenceStd = readStats(statsFile)
                evidence = np.exp(logevidence)
                print("{}, {}: {} +- {}\n".format(modelName1, modelName2, evidence, evidenceStd))
                
                
                g.write("{}, {}: {} +- {}\n".format(modelName1, modelName2, evidence, evidenceStd))
                
                resultsList.append([modelName1, modelName2, evidence, evidenceStd])
            
            
        evidenceSum = 0
        for i in range(len(resultsList)):
            evidenceSum += float(resultsList[i][2])
        g.write("Total evidence: {}".format(evidenceSum))
        
        #makeTableB(resultsList, resultDirectory)
        
        
        sortedResults = sorted(resultsList, key = lambda x: x[2], reverse=True)
        makeTableB(sortedResults, resultDirectory, "sortedResultsB.txt")
        
        fractionalEvidence(sortedResults, resultDirectory, "fracResultsB.txt")
        
        
    return

def analyseMainA(resultDirectory):
    with open(resultDirectory + "collectedResults.txt","w+") as g:
        g.write("Model------ Evidence ------ LogEvidence std\n")
        
        resultsList = []
        for modelName, ndim, paramNames in modelSetA:
            print(modelName)
            
            prefix = resultDirectory + modelName[:4] + "/"
            
            with open(prefix + "stats.dat","r") as statsFile:
                logevidence, evidenceStd = readStats(statsFile)
                evidence = np.exp(logevidence)
                print("{}: {} +- {}\n".format(modelName, evidence, evidenceStd))
                g.write("{}: {} +- {}\n".format(modelName, evidence, evidenceStd))
                resultsList.append([modelName, evidence, evidenceStd])
            
        
        evidenceSum = 0
        for i in range(len(resultsList)):
            evidenceSum += float(resultsList[i][1])
        g.write("Total evidence: {}".format(evidenceSum))
        
        sortedResults = sorted(resultsList, key = lambda x: x[1], reverse=True)
        makeTableA(sortedResults, resultDirectory, "sortedResultsA.txt")
        
        fractionalEvidence(sortedResults, resultDirectory, "fracResultsA.txt")
    return


### Get results directory

if len(sys.argv) != 3:
    print("ERROR")
    print("Usage: hypothesisAnalyse.py [-A, -B] /out_directory/")
else:
    outputfileName = sys.argv[2]
    if sys.argv[1] == 'A':
        analyseMainA(outputfileName)
    elif sys.argv[1] == 'B':
        analyseMainB(outputfileName) 
    else:
        print("Argument Error")

 
