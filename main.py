import re
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, classification_report


trainData = pd.read_csv("train.csv")
testData = pd.read_csv("test.csv")
sampleData = pd.read_csv("sample (1).csv")

print(trainData.head())
print(trainData.shape)
print(testData.shape)
print(trainData.info())
print(trainData.describe().transpose())
print(trainData.isnull().sum())
print(testData.isnull().sum())


imputer = SimpleImputer(strategy = 'most_frequent')
imputedData1 = imputer.fit_transform(trainData)
trainData = pd.DataFrame(imputedData1,columns = trainData.columns)
print(trainData.isnull().sum())

imputedData2 = imputer.fit_transform(testData)
testData = pd.DataFrame(imputedData2, columns=testData.columns)
print(testData.isnull().sum())
print(testData.shape)

yTrain = trainData["sentiment"]
xTrain = trainData.drop(['sentiment'], axis = 1)
print(yTrain.value_counts())

trainData.drop(columns=['movieid','reviewerName'],inplace=True)
testData.drop(columns=['movieid','reviewerName'],inplace=True)

def text_clean1(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text =re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*', '', text)
    return text
cleaned1 = lambda x : text_clean1(x)

trainData['cleaned_reviewText']=pd.DataFrame(trainData.reviewText.apply(cleaned1))
testData['cleaned_reviewText']=pd.DataFrame(testData.reviewText.apply(cleaned1))
print(trainData.head(10))

print(testData.head())
def text_clean_2(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\[I\], "', '', text)
    return text
cleaned2 = lambda x: text_clean_2(x)

trainData['cleaned_reviewText_new']=pd.DataFrame(trainData.cleaned_reviewText.apply(cleaned2))
testData['cleaned_reviewText_new']=pd.DataFrame(testData.cleaned_reviewText.apply(cleaned2))
print(trainData.head(10))

print(testData.head(10))

xTrain = trainData.cleaned_reviewText_new
yTrain = trainData.sentiment

xTest = testData.cleaned_reviewText_new

vectorizer = CountVectorizer()
xTrainVectorized = vectorizer.fit_transform(xTrain)
xTestVectorized = vectorizer.transform(xTest)


# Create a Perceptron classifier
model1 = Perceptron(random_state=64)

# Train the Perceptron on your training data
model1.fit(xTrainVectorized, yTrain)
yPred = model1.predict(xTrainVectorized)

print(f1_score(yTrain,yPred,pos_label="POSITIVE"))

print(yPred.shape)
yTestPred = model1.predict(xTestVectorized)

yTestPred.shape

model2 = AdaBoostClassifier(random_state = 1729,n_estimators=70)
model2.fit(xTrainVectorized, yTrain)
yPred = model2.predict(xTrainVectorized)

print(f1_score(yTrain,yPred,pos_label="POSITIVE"))

print(yPred.shape)

yTestPred = model2.predict(xTestVectorized)

print(yTestPred.shape)


model3 = MultinomialNB()
model3.fit(xTrainVectorized, yTrain)
yPred = model3.predict(xTrainVectorized)

print(f1_score (yTrain,yPred,pos_label="POSITIVE"))

yTestPred = model3.predict(xTestVectorized)

print(yTestPred.shape)

print(f1_score (yTrain,yPred,pos_label="POSITIVE"))

sub = pd.DataFrame(yTestPred, columns=['sentiment'])
sub.index.name = 'id'
sub.to_csv("submission.csv", encoding='utf-8')

output = pd.read_csv("submission.csv")