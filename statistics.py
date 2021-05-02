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
    plt.title("Max Tile evolution for each algorithm")
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

    setLabels = sum(listLabels,[])
    allLabels = sorted((list(set(setLabels))))

    dfdata = pd.DataFrame({},
        index = algoNames, columns = allLabels
    )
    for algo in range(len(algoNames)):
        listtoappend = []
        for i in range(len(allLabels)):
            index = listLabels[algo].index(allLabels[i]) if allLabels[i] in listLabels[algo] else -1
            listtoappend.append(listProbs[algo][index]) if index != -1 else listtoappend.append(0)
        dfdata.loc[algoNames[algo]] = listtoappend
    algoNamesNoTxt = [name.replace('.txt','') for name in algoNames]
    dfdata.index = algoNamesNoTxt
    print(dfdata)
    
    plt.style.use('ggplot')
    ax = dfdata.plot(stacked=True, kind='bar', figsize=(12, 8), rot='horizontal',title='Percentage maximum value for all algorithms')

    # .patches is everything inside of the chart
    for rect in ax.patches:
        # Find where everything is located
        height = round(rect.get_height(),2)
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()
        
        # The height of the bar is the data value and can be used as the label
        label_text = f'{height}'  # f'{height:.2f}' to format decimal values
        
        # ax.text(x, y, text)
        label_x = x + width / 2
        label_y = y + height / 2

        # plot only when height is greater than specified value
        if height > 0:
            ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8)
        
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)    
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