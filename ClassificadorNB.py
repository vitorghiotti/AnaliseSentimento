import re
import string
import csv
import nltk
import sqlite3
import unicodedata

from sqlite3 import Error
from nltk.util import ngrams 
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.corpus import stopwords
from nltk.sentiment.util import *
from nltk.metrics import ConfusionMatrix
from sklearn.model_selection import train_test_split
#*******************************************************************************
lst_excessoes = None
lst_plural = None
lst_replace = None
lst_stopw = None
lst_base_treino = None
lst_add_base_treino = None
featureVectorBase = []
featureList=[]

conn = None
#*******************************************************************************
def word_grams(words,nro): 
    s = [] 
    maximo = nro + 1
    
    for n in range(1, maximo): 
        for ngram in ngrams(words, n): 
            s.append('_'.join(str(i) for i in ngram)) 
    
    return s 
#*******************************************************************************
def features(sentence):
    global lst_excessoes
    
    words = sentence.lower().split()
    d={}

    if len(words)>1:
        for w in words:
            if len(w)>1:
                d['contains(%s)' % w]=True
    elif len(words)==1:
        excessao = words in [list(row) for row in lst_excessoes]
        if excessao==False:
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
    for rpl in lst_replace:
        sentence = sentence.replace(rpl[0], rpl[1])
     
    return sentence
#***********************************************************************************
def processa_plural(word):
    global lst_plural
    
    excessao_plural = [word] in [list(row) for row in lst_plural]
    
    if excessao_plural==False:
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
    global lst_stopw
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
    for sw in lst_stopw:
        if sw[0]==None:
            stopWords.remove(sw[1])
            #print('remove: ' + str(sw[1]))
        else:
            stopWords.append(sw[0])
            #print('append: ' + str(sw[0]))
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
        #w = replaceTwoOrMore(w)
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
                #replace two or more with two occurrences
                w = replaceTwoOrMore(w)
                #processa plural
                w = processa_plural(w)
                featureVector.append(w.lower())
        
    return featureVector
#***********************************************************************************
def extract_features(sentence):
    global featureList
    
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
    global ngram
    
    dic = {}
    
    #print(frase)
    frase = processSentence(frase)
    #print(frase)
    frase = getFeatureVector(frase,stopWords)
    #print(frase)
    frase = word_grams(frase,ngram)
    #print(frase)
    dic = extract_features(frase)
    #print(dic)
    if len(frase)>minDicLen:
        return classifier.classify(dic)
    else:
        return -999
#***********************************************************************************
def Prob_Label(frase,label):
    global classifier
    global stopWords
    global minDicLen
    global ngram
    
    dic = {}
    
    frase = processSentence(frase)
    frase = getFeatureVector(frase,stopWords)
    frase = word_grams(frase,ngram)
    dic = extract_features(frase)
    if len(frase)>minDicLen:
        return classifier.prob_classify(dic).prob(label)
    else:
        return 0
#***********************************************************************************
def get_features(frase):
    global classifier
    global stopWords
    global minDicLen
    global ngram
    
    dic = {}
   
    frase = processSentence(frase)
    frase = getFeatureVector(frase,stopWords)
    frase = word_grams(frase,ngram)
    dic = extract_features(frase)
    if len(frase)>minDicLen:
        return str(frase)
    else:
        return 0
#***********************************************************************************
def carrega_listas_from_db():
    global lst_excessoes
    global lst_plural
    global lst_replace
    global lst_stopw
    global lst_base_treino
    global conn
    
    conn = sqlite3.connect('tweets.sqlite')
    cur = conn.cursor()
    #************************************************************************************
    cur.execute("SELECT de FROM excessoes where tipo = 'dic'")
    lst_excessoes = cur.fetchall()
    #************************************************************************************
    cur.execute("SELECT de FROM excessoes where tipo = 'pl'")
    lst_plural = cur.fetchall()
    #************************************************************************************
    cur.execute("SELECT de,para FROM excessoes where tipo = 'rpl'")
    lst_replace = cur.fetchall()
    #************************************************************************************
    cur.execute("SELECT de,para FROM excessoes where tipo = 'sw'")
    lst_stopw = cur.fetchall()
    #************************************************************************************
    cur.execute("SELECT * FROM tweet where tweet_id in (0,1)")
    lst_base_treino = cur.fetchall()
    
    conn.close()
 #***************************************************************************************
def add_base_treino():
    global conn
    global lst_add_base_treino

    conn = sqlite3.connect('tweets.sqlite')
    cur = conn.cursor()
    cur.execute("SELECT reacao,sentimento FROM reacao where mark_for_train=1")
    lst_add_base_treino = cur.fetchall()
    
    if lst_add_base_treino!=None:
        conn = sqlite3.connect('tweets.sqlite')
        cur = conn.cursor()
        for row in lst_add_base_treino:
            cur.execute("insert into tweet (tweet_id, text, sentiment) values ('1', '" + str(row[0]) + "', '" + str(row[1]) + "')").fetchone()
            cur.execute("update reacao set mark_for_train=2 where mark_for_train=1")
            conn.commit()
    
    conn.close()
#***********************************************************************************    
def treinaModelo(use_nltkStopWords=True,ngrms=2,prob=0.975):
    #Read the sentence one by one and process it
    global stopWords
    global featureVectorBase
    global featureList
    global classifier
    global nltkStopWords
    global lst_base_treino
    global ngram
    
    ngram = ngrms
    stopWords = []
    featureList = []   
    featureVectorBase=[]
    nltkStopWords=use_nltkStopWords

    add_base_treino()
    carrega_listas_from_db()
    stopWords = getStopWordList('portuguese')

    # Get tweet words
    cont_neg=0
    cont_pos=0
    cont_neu=0
    cont_max=5000
    item_vec_max = 1

    for row in lst_base_treino:
        if len(row)==4:
            if row[3] == 1 and cont_pos <= cont_max:
                cont_pos +=1
                sentiment = row[3]
                sentence = row[2]
                processedSentence = processSentence(sentence)
                featureVector = getFeatureVector(processedSentence, stopWords)
                if len(featureVector)>item_vec_max:
                    featureList.extend(featureVector)
                    featureVector = word_grams(featureVector,ngram)
                    featureVectorBase.append((featureVector, sentiment))
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
                    featureList.extend(featureVector)
                    featureVector = word_grams(featureVector,ngram)
                    featureVectorBase.append((featureVector, sentiment))
                    1==1
                else:
                    #print(featureVector, sentiment)
                    1==1

            elif row[3] == 0 and cont_neu <= cont_max:
                cont_neu +=1
                sentiment = 1
                sentence = row[2]
                processedSentence = processSentence(sentence)
                featureVector = getFeatureVector(processedSentence, stopWords)
                if len(featureVector)>item_vec_max:
                    featureList.extend(featureVector)
                    featureVector = word_grams(featureVector,ngram)
                    #featureVectorBase.append((featureVector, sentiment))
                    1==1
                else:
                    #print(featureVector, sentiment)
                    1==1
    
    featureList = list(set(featureList))
    training_set = nltk.classify.util.apply_features(extract_features, featureVectorBase)
    classifier = nltk.NaiveBayesClassifier.train(training_set)
#***********************************************************************************
classifier=nltk.NaiveBayesClassifier

carrega_listas_from_db()
add_base_treino()
ngram = 2
minDicLen = 0
featureVectorBase=[]
nltkStopWords = True
stopWords = []
#carrega_base_treino()
#treinaModelo(True)