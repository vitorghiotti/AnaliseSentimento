import re
import string
import csv
import nltk
import sqlite3
import unicodedata

from sqlite3 import Error
from nltk.classify import PositiveNaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.corpus import stopwords
from nltk.sentiment.util import *
from nltk.metrics import ConfusionMatrix
from sklearn.model_selection import train_test_split
#*******************************************************************************
def features(sentence):
    words = sentence.lower().split()
    d={}
    for w in words:
        if len(w)>1:
            d['contains(%s)' % w]=True
        
    return d
#***********************************************************************************
def processSentence(sentence):
    # process the tweets

    #Convert to lower case
    sentence = sentence.lower()
    #Convert www.* or https?://* to URL
    sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',sentence)
    #Convert @username to AT_USER
    sentence = re.sub('@[^\s]+','AT_USER',sentence)
    #Remove additional white spaces
    sentence = re.sub('[\s]+', ' ', sentence)
    #Replace #word with word
    sentence = re.sub(r'#([^\s]+)', r'\1', sentence)
    #trimd
    sentence = sentence.strip('\'"')
    #retira açentos e caracteres especiais
    sentence = str(unicodedata.normalize('NFKD', sentence).encode('ascii','ignore'),'utf8')
    #*************************   
    sentence = sentence.replace("tmb", "tambem")
    sentence = sentence.replace("tbm", "tambem")
    sentence = sentence.replace("vc", "voce")
    sentence = sentence.replace("sdd", "saudade")
    sentence = sentence.replace("sdds", "saudade")
    sentence = sentence.replace("dnv", "novamente")
    sentence = sentence.replace("pfvr", "por favor")
    sentence = sentence.replace("/", " ")
    sentence = sentence.replace("numca", "nunca")
    return sentence
#***********************************************************************************
#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    #return pattern.sub(r"\1\1", s)
    return pattern.sub(r"\1", s)
#***********************************************************************************
#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    global nltkStopWords
    stopWords = []
    
    if nltkStopWords==True:
        stopWords = nltk.corpus.stopwords.words('portuguese')
    else:
        fp = open(stopWordListFileName, 'r')
        line = fp.readline()
        while line:
            word = line.strip()
            stopWords.append(word)
            line = fp.readline()
        fp.close()
        
    #****************************************
    stopWords.append('AT_USER')
    stopWords.append('URL')
    stopWords.append('url')
    stopWords.append('pra')
    stopWords.append('q')
    stopWords.append('at_user')
    stopWords.append('poder')
    stopWords.append('profunda')
    stopWords.append('voce')
    stopWords.append('to')
    #*****************************************
    stopWords.remove('nem')
    #*****************************************
    return stopWords
#***********************************************************************************
#start getfeatureVector
def getFeatureVector(sentence,stopWords):
    featureVector = []
    #split tweet into words
    words = sentence.split()
    for w in words:        
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        if len(w)>0:
            if 1!=1:
                #print(len(w))
                stemmer = nltk.stem.SnowballStemmer('portuguese')
                w = stemmer.stem(w)
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
        
    return featureVector
#***********************************************************************************
def extract_features(sentence):
    sentence_words = set(sentence)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in sentence_words)
    return features

#***********************************************************************************
def classifica(frase):
    global classifier
    global stopWords
    dic = {}
    
    processedTestTweet = processSentence(frase)
    aux = " ".join((getFeatureVector(processedTestTweet,stopWords)))
    dic = features(aux)
    if len(dic)>1:
        return classifier.classify(dic)
    else:
        return None
#***********************************************************************************
def Prob_false(frase,label=False):
    global classifier
    global stopWords
    dic = {}
    
    processedTestTweet = processSentence(frase)
    aux = " ".join((getFeatureVector(processedTestTweet,stopWords)))
    dic = features(aux)
    if len(dic)>1:
        return classifier.prob_classify(dic).prob(label)
    else:
        return 0
#***********************************************************************************
def treinaModelo(use_nltkStopWords=True):
    #Read the sentence one by one and process it
    global stopWords
    global featureVectorPositive
    global featureVectorNonPositive
    global classifier
    global nltkStopWords
    
    stopWords = []
    featureList = []   
    featureVectorPositive=[]
    featureVectorNonPositive=[]
    nltkStopWords=use_nltkStopWords
    
    #inpTweets = csv.reader(open('BaseTreinamento.csv', 'r'), delimiter=',')
    conn = sqlite3.connect('tweets.sqlite')
    cur = conn.cursor()
    cur.execute("SELECT * FROM tweet where tweet_id == '0'")
    inpSentence = cur.fetchall()

    stopWords = getStopWordList('portuguese')

    # Get tweet words
    cont_neg=0
    cont_pos=0
    cont_neu=0
    cont_max=3000
    item_vec_max = 1

    for row in inpSentence:
        if len(row)==4:
            if row[3] == 1 and cont_pos <= cont_max:
                cont_pos +=1
                sentiment = row[3]
                sentence = row[2]
                processedSentence = processSentence(sentence)
                featureVector = getFeatureVector(processedSentence, stopWords)
                if len(featureVector)>item_vec_max:
                    featureVectorPositive.append(" ".join(featureVector))
                    1==1
                else:
                    #print(featureVector, sentiment)
                    1==1

            elif row[3] == -1 and cont_neg <= cont_max:
                cont_neg +=1
                sentiment = row[3]
                sentence = row[2]
                processedSentence = processSentence(sentence)
                featureVector = getFeatureVector(processedSentence, stopWords)
                if len(featureVector)>item_vec_max:
                    featureVectorNonPositive.append(" ".join(featureVector))
                    1==1
                else:
                    #print(featureVector, sentiment)
                    1==1

            elif row[3] == 0 and cont_neu <= cont_max:
                cont_neu +=1
                sentiment = row[3]
                sentence = row[2]
                processedSentence = processSentence(sentence)
                featureVector = getFeatureVector(processedSentence, stopWords)
                if len(featureVector)>item_vec_max:
                    featureVectorPositive.append(" ".join(featureVector))
                    1==1
                else:
                    #print(featureVector, sentiment)
                    1==1

                
    featureVectorNonPositive = list(map(features, featureVectorNonPositive))
    featureVectorPositive = list(map(features, featureVectorPositive))
    classifier = PositiveNaiveBayesClassifier.train(featureVectorPositive,featureVectorNonPositive)
#***********************************************************************************
#initialize stopWords
classifier=nltk.classify.PositiveNaiveBayesClassifier
featureVectorPositive=[]
featureVectorNonPositive=[]
nltkStopWords = True
stopWords = []
treinaModelo()
