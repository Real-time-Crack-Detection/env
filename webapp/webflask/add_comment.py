from flask import request
from models import *

def addComment(session):

    comment1 = str(request.form['addcommenttext'])
    title1 = str(request.form['hiddenvalue'])

    print("comment : ", comment1)
    print("title1 : ", title1)
    try:
        a = session.query(Image)
        b = a.filter(Image.title1 == title1)

        c = b.update({Image.comment:comment1})

        session.commit()
        print("record update finish")
    except:
        print("record update error")
