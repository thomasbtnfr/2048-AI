import pandas as pd
from collections import Counter
import sys

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
    for key in counter:
        probability = counter[key] / len(maxTiles) * 100
        print(key, '|', str(round(probability,2)) + '%')

def main(argv):
    readResFile(argv)

if __name__ == "__main__":
    main(sys.argv[1])
