import pandas as pd
import nltk
import re
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

import pickle


def wrangling():

    df=pd.read_csv('Spam/spam.csv',
                   encoding='latin-1'
                   )
    df = df.rename(columns={'v1': 'labels', 'v2': 'message'})
    print(df.columns)



    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],inplace=True,axis=1)

    print(df.head())


    stemmer=PorterStemmer()

    lemmentizer=WordNetLemmatizer()
    corpus=[]
    for i in range(len(df)):
        msg=re.sub('[^a-zA-Z]',' ',df['message'][i])
        msg=msg.lower()
        msg=msg.split()
        msg=[lemmentizer.lemmatize(word) for word in msg if not word  in set(stopwords.words('english'))]
        msg=' '.join(msg)
        corpus.append(msg)

    print(corpus[:8])

    bow=CountVectorizer(max_features=5000)
    X=bow.fit_transform(corpus).toarray()

    pickle.dump(bow,open('transformer.pkl','wb'))

    le=LabelEncoder()
    Y=le.fit_transform(df['labels']) # 0->ham 1->spam
    print(Y[:8])



    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=101)


    print(x_train[0].shape)

    spam_classifier=LogisticRegression()

    spam_classifier.fit(x_train,y_train)

    predictions=spam_classifier.predict(x_test)


    print(confusion_matrix(y_test,predictions))

    print(classification_report(y_test,predictions))


    print(accuracy_score(y_test,predictions))


    '''filename = 'Spam-Classifier.pkl'
    joblib.dump(spam_classifier,filename)'''






    pass



if __name__ == '__main__':
    wrangling()





