import numpy as np
import flask
from flask import Flask, request, render_template
from flask_restful import Resource, Api
from flask_cors import CORS
import pickle

app = flask.Flask(__name__)
api = Api(app)
CORS(app)

###--- FLASK APP MULAI DARI SINI ---###
@app.route('/')
def index():
    return render_template('/index.html')

@app.route('/product', methods=['GET', 'POST'])
def prediction():
    
    if request.method == 'GET':
        return render_template('/product.html')
    elif request.method == 'POST':
        
        with open('model.pkl', 'rb') as r:
            pred = pickle.load(r)
        
        tipe = int(request.form['tipe'])
        orang = int(request.form['orang'])
        ac = int(request.form['ac'])
        km = int(request.form['km'])
        wifi = int(request.form['wifi'])
        listrik = int(request.form['listrik'])
        
        features = np.array((orang, km, wifi, listrik, ac, tipe))
        datas = np.reshape(features, (1,-1))
        
        hargakos = int(pred.predict(datas))
        final = "Rp" + str(hargakos)
        
        return render_template('/product.html', tipe=tipe, orang=orang, ac=ac, km=km, wifi=wifi, listrik=listrik, result=final)
    
@app.route('/about')
def about():
    return render_template('/about.html')

@app.route('/review')
def review():
    return render_template('/review.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0")