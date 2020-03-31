from __init__ import request
from sqlalchemy import text

# 쿼리해야할 정보 : title, save_date, comment
def view_history(session, code):
    code = '1'
    title = session.query('title').from_statement(text("select * from image where code = " + code)).all()
    save_date = session.query('save_date').from_statement(text("select * from image user where code = " + code)).all()
    comment = session.query('comment').from_statement(text("select * from image where code = " + code)).all()

    for i in title:
        print(i)
