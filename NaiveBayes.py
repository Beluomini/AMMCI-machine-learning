import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # for regex
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # for regex
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



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


# y é um array com as sentimentos
y = np.array(data.sentiment.values)
# converte o texto em uma matriz de contagem de tokens (??)
cv = CountVectorizer(max_features = 1000)

# treino seguido de uma transformacao
X = cv.fit_transform(data.review).toarray()


trainx,testx,trainy,testy = train_test_split(X,y,test_size=0.2,random_state=9)

gnb,mnb,bnb = GaussianNB(),MultinomialNB(alpha=1.0,fit_prior=True),BernoulliNB(alpha=1.0,fit_prior=True)
gnb.fit(trainx,trainy)
mnb.fit(trainx,trainy)
bnb.fit(trainx,trainy)

ypg = gnb.predict(testx)
ypm = mnb.predict(testx)
ypb = bnb.predict(testx)

print("Gaussian score = ",accuracy_score(testy,ypg))
print("Multinomial score = ",accuracy_score(testy,ypm))
print("Bernoulli score = ",accuracy_score(testy,ypb))