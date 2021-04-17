import pandas as pd
from collections import Counter
import sys
import glob, os
import matplotlib.pyplot as plt
from sys import maxsize as MAX_INT
from cycler import cycler
from matplotlib import rcParams

rcParams.update({'figure.autolayout':True}) # to avoid having the xlabel cut

def readResFile(fileName: str):
    colnames = ['MaxTile','Score']
    res = pd.read_csv(fileName + '.txt',names = colnames, sep=';')
    print('-'*100)
    print('Results algorithm :', fileName, '| Number of games : ', res.shape[0])
    print('-'*100)
    maxTiles = res['MaxTile'].tolist()
    scores = res['Score'].tolist()
    print('Max Tile reached :', res['MaxTile'].max())
    print('Maximum Tile Mean :', round(res['MaxTile'].mean(),2))
    print('-'*100)
    print('Score Max reached :', res['Score'].max())
    print('Score Mean :', round(res['Score'].mean(),2))
    print('-'*100)
    print('Probability to reach the following values')
    counter = Counter(maxTiles)
    probs = []
    labels = []
    for key in counter:
        probability = counter[key] / len(maxTiles) * 100
        print(key, '|', str(round(probability,2)) + '%')
        probs.append(round(probability,2))
        labels.append(key)
    print(labels)
    plt.title(fileName)
    plt.pie(probs, startangle=90, autopct='%.2f%%')
    plt.legend(labels, loc='upper right', bbox_to_anchor = (1.3, 0.9), fontsize=15)
    plt.show()


def printGraph():
    os.chdir("./")
    plt.xlabel("Games")
    plt.ylabel("Max Tile")
    colnames = ['MaxTile','Score']
    for file in glob.glob("*.txt"):
        if file != "supervisedData.txt":
            res = pd.read_csv(file, names=colnames, sep=';')
            maxTiles = res['MaxTile'].tolist()
            plt.plot(maxTiles, label=file)
    plt.legend()
    plt.show()


def printHisto():
    
    os.chdir("./")
    colnames = ['MaxTile','Score']
    algoNames = []

    listProbs = []
    listLabels = []

    for file in glob.glob("*.txt"):
        if file != "supervisedData.txt":
            algoNames.append(file)
            res = pd.read_csv(file, names=colnames, sep=';')
            maxTiles = res['MaxTile'].tolist()
            counter = Counter(maxTiles)
            probs = []
            labels = []
            for key in counter:
                probability = counter[key] / len(maxTiles) * 100
                probs.append(round(probability,2))
                labels.append(key)
            listProbs.append(probs)
            listLabels.append(labels)

    print("Algos", algoNames)
    print("Labels", listLabels)
    print("Probs", listProbs)

    setLabels = sum(listLabels,[])
    allLabels = sorted((list(set(setLabels))))
    print(allLabels)

    dfdata = pd.DataFrame({},
        index = algoNames, columns = allLabels
    )
    for algo in range(len(algoNames)):
        listtoappend = []
        for i in range(len(allLabels)):
            index = listLabels[algo].index(allLabels[i]) if allLabels[i] in listLabels[algo] else -1
            listtoappend.append(listProbs[algo][index]) if index != -1 else listtoappend.append(0)
            print(listtoappend)
        dfdata.loc[algoNames[algo]] = listtoappend
    algoNamesNoTxt = [name.replace('.txt','') for name in algoNames]
    dfdata.index = algoNamesNoTxt
    print(dfdata)
    dfdata.plot(kind="bar",stacked=True, rot=45, include_bool = 1,title='Percentage of maximum value for all algorithms')
    
    plt.legend(loc='upper right')
    plt.show()


def main(argv):
    if argv == "graph":
        printGraph()
    elif argv == "histo":
        printHisto()
    else:
        readResFile(argv)

if __name__ == "__main__":
    main(sys.argv[1])