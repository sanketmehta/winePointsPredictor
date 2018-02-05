from flask import Flask, render_template, request
import numpy as np 
import pandas as pd 
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle
from time import gmtime, strftime


app = Flask(__name__)
app.debug = True

decision_tree_pkl_filename = 'wineRatingClassifier_1.pkl'
wineData_pkl_filename = 'wineData_1.pkl'
columnList_pkl_filename = 'columnList_1.pkl'

# Loading the objects with pickle
decision_tree_model_pkl = open(decision_tree_pkl_filename, 'rb')
decision_tree_model = pickle.load(decision_tree_model_pkl)

wineData_pkl = open(wineData_pkl_filename, 'rb')
wineData = pickle.load(wineData_pkl)

columnList_pkl = open(columnList_pkl_filename, 'rb')
columnList = pickle.load(columnList_pkl)

clf = decision_tree_model

variety_category = sorted(wineData['variety'].unique().tolist())
winery_category = sorted(wineData['winery'].unique().tolist())
region_1_category = sorted(wineData['region_1'].unique().tolist())



def add_missing_dummy_columns( d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0
        
def predict_points(varietyVal, priceVal, wineryVal, region1Val):
    df1 = pd.DataFrame({'variety': varietyVal, 'price': priceVal, 'winery': wineryVal, 'region_1': region1Val}, index=[0])
    new_sample = pd.get_dummies(df1)
    add_missing_dummy_columns(new_sample, columnList)
    new_sample1_prd = clf.predict(new_sample)
    return new_sample1_prd[0]

wine150k_pkl_filename = 'wine150k_1.pkl'

wine150k_pkl = open(wine150k_pkl_filename, 'rb')
wine150k = pickle.load(wine150k_pkl)

wine150k = wine150k.drop('Unnamed: 0',  axis=1)

wine130k_pkl_filename = 'wine130k_1.pkl'

wine130k_pkl = open(wine130k_pkl_filename, 'rb')
wine130k = pickle.load(wine130k_pkl)

wine130k = wine130k.drop('Unnamed: 0',  axis=1)

df = pd.merge(wine130k, wine150k, how='outer', indicator=True)
onlyIn130kList_allCol = df[df['_merge']=='left_only'][wine130k.columns]
onlyIn130kList = onlyIn130kList_allCol[['variety', 'province', 'price', 'winery', 'region_1', 'points']]
uniqueIn130kList = onlyIn130kList[onlyIn130kList.duplicated(keep=False)]
uniqueIn130kList = uniqueIn130kList.drop_duplicates(keep='first')
uniqueIn130kList = uniqueIn130kList.dropna(subset=['province','variety','price','points','region_1'], how='any')
lastIndex = len(uniqueIn130kList)


def getDataBetweenRecrds(startIndex, endIndex):
    resRecords = uniqueIn130kList[startIndex:endIndex]
    return resRecords

def selectRandomWine():
    randomWineIndex = randint(1, lastIndex)
    selectedRecord = getDataBetweenRecrds(randomWineIndex-1,randomWineIndex)
    return selectedRecord

def isValPresent(valToCheck, lstToCheck):
    if valToCheck in lstToCheck:
        return True
    else:
        return False

def canWineBePredicted(variety,winery,region):
    varietyPresent = isValPresent(variety, variety_category)
    wineryPresent = isValPresent(winery, winery_category)
    region_1Present = isValPresent(region, region_1_category)
    if (varietyPresent and wineryPresent and region_1Present):
        return True
    else:
        return False

def selectAllDetails(selectedWine):
    lst1 = []
    onlyIn130kList_allCol.reset_index()
    for x in range(0,len(onlyIn130kList_allCol)):
        if(selectedWine['variety'].values[0] == onlyIn130kList_allCol[x:x+1]['variety'].values[0] and
           selectedWine['winery'].values[0] == onlyIn130kList_allCol[x:x+1]['winery'].values[0] and
           selectedWine['price'].values[0] == onlyIn130kList_allCol[x:x+1]['price'].values[0] and
           selectedWine['points'].values[0] == onlyIn130kList_allCol[x:x+1]['points'].values[0] and
           selectedWine['region_1'].values[0] == onlyIn130kList_allCol[x:x+1]['region_1'].values[0]):
            lst1.append(onlyIn130kList_allCol[x:x+1])
    return lst1[0]
    
def selectWineForPrediction(variety_category,winery_category,region_1_category):
    lstVal, retVal = [], []
    selectedWine = selectRandomWine()
    predictWine = canWineBePredicted(selectedWine['variety'].values[0],selectedWine['winery'].values[0],selectedWine['region_1'].values[0])
    if predictWine:
        selWine = selectAllDetails(selectedWine)
        lstVal.append(selWine)
        predictedVal = predict_points(selectedWine['variety'].values[0],selectedWine['price'].values[0],selectedWine['winery'].values[0],selectedWine['region_1'].values[0])
        lstVal.append(predictedVal)
        retVal = lstVal
    else:
        retVal = selectWineForPrediction(variety_category,winery_category,region_1_category)
    return retVal

def selectWine():
    selWine = selectWineForPrediction(variety_category,winery_category,region_1_category)
    return selWine

@app.route('/', methods=['GET'])
def dropdown():
    a = predictPoints()
    return a

@app.route("/predictPoints" , methods=['GET', 'POST'])
def predictPoints():
    selWine = selectWine()
    selectedWine = selWine[0]
    predictedRating = selWine[1]
    if int(selectedWine['points'].values[0]) >= int(predictedRating):
        if (int(selectedWine['points'].values[0]) - int(predictedRating)) == 0:
            predctdRange = str((predictedRating - 2)) + ' - ' + str((predictedRating + 2))
        elif (int(selectedWine['points'].values[0]) - int(predictedRating)) == 1:
            predctdRange = str((predictedRating - 1)) + ' - ' + str((predictedRating + 3))
        else:
            predctdRange = str((predictedRating)) + ' - ' + str((predictedRating + 4))
    else:
        if (int(predictedRating) - int(selectedWine['points'].values[0])) == 1:
            predctdRange = str((int(selectedWine['points'].values[0]) - 1)) + ' - ' + str((int(selectedWine['points'].values[0]) + 3))
        else:
            predctdRange = str((int(selectedWine['points'].values[0]))) + ' - ' + str((int(selectedWine['points'].values[0]) + 4))

    if(request.form.get('varieties') is None and request.form.get('wineries') is None and request.form.get('regions') is None and request.form.get('price') is None):
        variety = variety_category[0]
        winery = winery_category[0]
        region = region_1_category[0]
        price = float(10)
    else:
        variety = request.form.get('varieties')
        winery = request.form.get('wineries')
        region = request.form.get('regions')
        price = float(request.form.get('price'))
    df1 = pd.DataFrame({
    'variety': variety,
    'winery': winery,
    'region_1': region,
    'price': price
    }, index=[0])

    predctdVal = predict_points(variety, price, winery, region)
    predctdRange2 = str((predctdVal - 2)) + ' - ' + str((predctdVal + 2))
    return render_template('index.html', varieties=variety_category, wineries=winery_category, regions=region_1_category, results2=selectedWine.to_html(index=False),results3=predictedRating, result=df1.to_html(index=False,col_space=10), result2=predctdVal, result3=predctdRange, result4=predctdRange2)


if __name__ == "__main__":
    app.run()