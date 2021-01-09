from flask import Flask,render_template,url_for,request, jsonify
import pandas as pd 
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
r=pd.read_csv('r.csv')
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
clf0 = pickle.load(open('predict0.pkl', 'rb'))
clf1 = pickle.load(open('predict1.pkl', 'rb'))
clf2 = pickle.load(open('predict2.pkl', 'rb'))
clf3 = pickle.load(open('predict3.pkl', 'rb'))
clf4 = pickle.load(open('predict4.pkl', 'rb'))
clf5 = pickle.load(open('predict5.pkl', 'rb'))
cv=pickle.load(open('tokenizor.pkl','rb'))
api = Flask(__name__)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
@api.route('/')
def hello_world():
    # print('welcome')
    return 'This is my first API call!'
@api.route('/pridict', methods=["POST"])
def predict():
    # print('request sent')
    input_json = request.get_json(force=True) 
    text=input_json['text']
    text=[text]
    text=pd.Series(text)
    vect = cv.transform(text)
    dictToReturn={}
    preds = np.zeros((1, len(label_cols)))
    for i, j in enumerate(label_cols):
        if i==0:
            preds[:,i] = clf0.predict_proba(vect.multiply(r.iloc[i,:]))[:,1]
        if i==1:
            preds[:,i] = clf1.predict_proba(vect.multiply(r.iloc[i,:]))[:,1]
        if i==2:
            preds[:,i] = clf2.predict_proba(vect.multiply(r.iloc[i,:]))[:,1]
        if i==3:
            preds[:,i] = clf3.predict_proba(vect.multiply(r.iloc[i,:]))[:,1]
        if i==4:
            preds[:,i] = clf4.predict_proba(vect.multiply(r.iloc[i,:]))[:,1]
        if i==5:
            preds[:,i] = clf5.predict_proba(vect.multiply(r.iloc[i,:]))[:,1]
        
        dictToReturn[j]=preds[0,i]
    return jsonify(dictToReturn)

if __name__ == '__main__':
	api.run(debug=True)