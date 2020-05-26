from flask import Flask, g, request, render_template,Markup
from flask import make_response
from flask import Response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import *
from signup import *
from login import *
from viewhistory import *

SQL_ID = "integer"
SQL_PASS = "dmswjd331"

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://' + SQL_ID + ':' + SQL_PASS + '@localhost:3306/capstone'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.debug = True

db = SQLAlchemy(app)
engine = create_engine('mysql+pymysql://' + SQL_ID + ':' + SQL_PASS + '@localhost:3306/capstone')
session = db.session

name = "Sign in"
code = ""

#첫 home 화면
@app.route('/')
def main():
    return render_template('base.html', name=name)

#기업정보-기업소개
@app.route('/intro')
def intro():
    return render_template('introduce.html', name=name)

#기업정보-기업소개
@app.route('/vision')
def vision():
    return render_template('vision.html', name=name)

#제품구매-드론구매
@app.route('/dron-order')
def dronorder():
    return render_template('dron-order.html', name=name)

#드론관리-드론정보
@app.route('/dron-info')
def droninfo():
    return render_template('dron-info.html', name=name)

#화면기본틀 html
@app.route('/text')
def text():
    return render_template('text.html', name=name)



#실시간 탐지 html
@app.route('/real-time')
def time():
    return render_template('real-time.html', name=name, image_file="img/testimg.jpg")

#내역조회 html
@app.route('/history')
def history():
    result_data = view_history(session, code)

    return render_template('view-history.html', name=name, result_data=result_data)

# sign up 버튼 클릭시 db로 데이터 insert
@app.route('/signup', methods = ['POST'])
def signupButton():
    sign_up(session)

    return render_template('base.html',name=name)

# 로그인 클릭시 화면
@app.route('/signin', methods = ['POST'])
def loginButton():
    global name
    global code
    name, code = log_in(session)

    return render_template('base.html', name=name)

# 기록 눌렀을 시 화면
@app.route('/save', methods = ['POST'])
def saving():
    save(session, code)
    return render_template('/real-time.html')

# 임시 진입점
#if __name__ == "__main__":
app.run(host='127.0.0.1')  # 127.0.0.1 ==localhost