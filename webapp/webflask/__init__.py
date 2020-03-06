from flask import Flask, g, request, render_template,Markup
from flask import make_response
from flask import Response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import *
from signup import *
from login import *

SQL_ID = "integer"
SQL_PASS = "dmswjd331"

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
    return render_template('base.html', name="Sign In")

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

#내역조회
@app.route('/history')
def history():
    return render_template('view-history.html')

# sign up 버튼 클릭시 db로 데이터 insert
@app.route('/signup', methods = ['POST'])
def signupButton():
    sign_up(session)

    return render_template('base.html', name="")

# 로그인 클릭시 화면
@app.route('/signin', methods = ['POST'])
def loginButton():
    name = log_in(session)

    return render_template('base.html', name=name)

# 임시 진입점
if __name__ == "__main__":
    app.run(host='127.0.0.1')  # 127.0.0.1 ==localhost



