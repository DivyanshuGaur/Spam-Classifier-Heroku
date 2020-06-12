

from sklearn.externals import joblib
import pickle



msg='A [redacted] loan for £950 is approved for you if you receive this SMS. 1 min verification & cash in 1 hr at www.[redacted].co.uk to opt out reply stop'
msg1=[msg]

cv=pickle.load(open('transformer.pkl','rb'))
clsfr=joblib.load('Spam-Classifier.pkl')


vect=cv.transform(msg1).toarray()

print(clsfr.predict(vect)[0])
