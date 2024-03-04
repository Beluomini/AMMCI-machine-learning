import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # for regex
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Carregando o dataset para treinamento
data = pd.read_csv('IMDB-Dataset.csv')


# substitui todos positivo/negativo por 0/1
data.sentiment.replace('positive',1,inplace=True)
data.sentiment.replace('negative',0,inplace=True)


# usa regex para limpar o texto
# tira tags html
def clean(text):
    cleaned = re.compile(r'<.*?>')
    # substitui o pattern por '' na string text e retorna a string substituida
    return re.sub(cleaned,'',text)
# remove caracteres especiais
def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem
# coloca o texto todo em minusculo
def to_lower(text):
    return text.lower()
# remove as stopwords, palavras que não tem significado
def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]
# ???????????????
def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

data.review = data.review.apply(clean)
data.review = data.review.apply(to_lower)
data.review = data.review.apply(rem_stopwords)
data.review = data.review.apply(stem_txt)


X = np.array(data.iloc[:,0].values)
y = np.array(data.sentiment.values)

cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(data.review).toarray()


train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
print("Train shapes : X = {}, y = {}".format(train_x.shape,train_y.shape))
print("Test shapes : X = {}, y = {}".format(test_x.shape,test_y.shape))

logistic = linear_model.LogisticRegression(solver="newton-cg", tol=1)
rbm = BernoulliRBM(random_state=0, verbose=True)

rbm_features_classifier = Pipeline(steps=[("rbm", rbm), ("logistic", logistic)])

# Training RBM-Logistic Pipeline
rbm_features_classifier.fit(train_x, train_y)

seila = rbm_features_classifier.predict(test_x)

print("Sei la: ", accuracy_score(test_y, seila))
