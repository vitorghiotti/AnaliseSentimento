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
    excessoes = ['nao','tambem']
    if len(words)>1:
        for w in words:
            if len(w)>1:
                d['contains(%s)' % w]=True
    elif len(words)==1:
        if (words[0] in excessoes)==False:
            d['contains(%s)' % words[0]]=True
                
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
    sentence = sentence.replace(" ta ", " esta ")
    sentence = sentence.replace("vc", "voce")
    sentence = sentence.replace("sdd", "saudade")
    sentence = sentence.replace("sdds", "saudade")
    sentence = sentence.replace("dnv", "novamente")
    sentence = sentence.replace("pfvr", "por favor")
    sentence = sentence.replace("/", " ")
    sentence = sentence.replace("numca", "nunca")
    sentence = sentence.replace("magico", "magica")
    sentence = sentence.replace("lindo", "linda")
    sentence = sentence.replace("amore", "amor")
    sentence = sentence.replace(" n ", " nao ")
    sentence = sentence.replace("ambas", "ambos")
    sentence = sentence.replace("tjm", "estamos juntos")
    return sentence
#***********************************************************************************
def processa_plural(word):
    dict_plural ={'rs':'','atras':'','aspas':'','apos':'','antes':'','alas':'','ademais':'','quantas':'','vamos':'','eses':'','esses':'','vezes':'','damos':'','queremos':'','apenas':'','duas':'','dois':'','atraves':'','caos':'','demais':'','ambos':'','detras':'','marques':''}
    if dict_plural.get(word)==None:
        if re.search(r's$', word):

            if re.search(r'as$', word):
                word = re.sub(r'irmas$', 'irma', word)
                word = re.sub(r'ais$', 'al', word)
                word = re.sub(r'aos$', 'ao', word)
                word = re.sub(r'as$', 'a', word)
                
            elif re.search(r'es$', word):
                if re.search(r'maes$', word):
                    word = 'maes'
                elif re.search(r'bres$', word):
                    word = re.sub(r'bres$', 'bre', word)
                elif re.search(r'zes$', word):
                    word = re.sub(r'zes$', 'z', word)
                elif re.search(r'tres$', word):
                    word = re.sub(r'tres$', 'tre', word)
                elif re.search(r'ores$', word):
                    word = re.sub(r'tres$', 'tre', word)
                else:
                    word = re.sub(r'oes$', 'ao', word)
                    word = re.sub(r'aes$', 'ao', word)
                    word = re.sub(r'gues$','gues', word)
                    word = re.sub(r'res$', 'r', word)
                    word = re.sub(r'es$', 'e', word)

            elif re.search(r'is$', word):
                if re.search(r'veis$', word):
                    word = re.sub(r'veis$', 'vel', word)
                else:
                    word = re.sub(r'ais$', 'al', word)
                    word = re.sub(r'zis$', 'zil', word)
                    word = re.sub(r'eis$', 'il', word)

            elif re.search(r'os$', word):
                if re.search(r'emos$', word):
                    word = re.sub(r'emos$', 'emos', word)
                elif re.search(r'amos$', word):
                    word = re.sub(r'amos$', 'amos', word)
                elif re.search(r'armos$', word):
                    word = re.sub(r'armos$', 'armos', word)
                else:
                    word = re.sub(r'ois$', 'oi', word)
                    word = re.sub(r'os$', 'o', word)

            elif re.search(r'us$', word):
                word = re.sub(r'eus$', 'eu', word)
                word = re.sub(r'us$', 'u', word)

            else:
                word = re.sub(r'ns$', 'm', word)

    return word
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
    stopWords.append('tao')
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
        if len(w)>1:
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
                #processa plural
                w = processa_plural(w)
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
    global minDicLen
    dic = {}
    
    processedTestSentence = processSentence(frase)
    aux = " ".join((getFeatureVector(processedTestSentence,stopWords)))
    dic = features(aux)
    if len(dic)>minDicLen:
        return classifier.classify(dic)
    else:
        return None
#***********************************************************************************
def Prob_Label(frase,label=False):
    global classifier
    global stopWords
    global minDicLen
    dic = {}
    
    processedTestTweet = processSentence(frase)
    aux = " ".join((getFeatureVector(processedTestTweet,stopWords)))
    dic = features(aux)
    if len(dic)>minDicLen:
        return classifier.prob_classify(dic).prob(label)
    else:
        return 0
#***********************************************************************************
def get_features(frase):
    global classifier
    global stopWords
    global minDicLen
    dic = {}
    
    processedTestTweet = processSentence(frase)
    aux = " ".join((getFeatureVector(processedTestTweet,stopWords)))
    dic = features(aux)
    if len(dic)>minDicLen:
        return dic
    else:
        return None
#***********************************************************************************
def carrega_base_treino():
    conn = sqlite3.connect('tweets.sqlite')
    cur = conn.cursor()
    #cur.execute("SELECT * FROM tweet where tweet_id = '0' and text like '%não%inútil%'")
    cur.execute("SELECT reacao,sentimento FROM reacao where mark_for_train=1")
    base_treino = cur.fetchall()

    for row in base_treino:
        #print(row)
        cur.execute("insert into tweet (tweet_id, text, sentiment) values ('0', '" + row[0] + "', '" + row[1] + "')").fetchone()
        conn.commit()
    
    conn.close()
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
    conn.close()
    
    featureVectorNonPositive = list(map(features, featureVectorNonPositive))
    featureVectorPositive = list(map(features, featureVectorPositive))
    classifier = PositiveNaiveBayesClassifier.train(featureVectorPositive,featureVectorNonPositive)
#***********************************************************************************
#initialize stopWords
classifier=nltk.classify.PositiveNaiveBayesClassifier
minDicLen = 0
featureVectorPositive=[]
featureVectorNonPositive=[]
nltkStopWords = True
stopWords = []
carrega_base_treino()
#treinaModelo(True)
