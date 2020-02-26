from flask import Flask, g, request, render_template,Markup
from flask import make_response
from flask import Response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import *

SQL_ID = "ai"
SQL_PASS = "ai"

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://' + SQL_ID + ':' + SQL_PASS + '@localhost:3306/capstone'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.debug = True

db = SQLAlchemy(app)
engine = create_engine('mysql+pymysql://' + SQL_ID + ':' + SQL_PASS + '@localhost:3306/capstone')
session = db.session


#첫 home 화면
@app.route('/')
def main():
    return render_template('base.html')

#기업정보-기업소개
@app.route('/intro')
def intro():
    return render_template('introduce.html')

#제품구매-드론구매
@app.route('/dron-order')
def dronorder():
    return render_template('dron-order.html')

#화면기본틀 html
@app.route('/text')
def text():
    return render_template('text.html')


# 임시 진입점
if __name__ == "__main__":
    app.run(host='127.0.0.1')  # 127.0.0.1 ==localhost



