from flask import Flask, g, request, render_template,Markup
from flask import make_response
from flask import Response
app = Flask(__name__)
app.debug = True

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




