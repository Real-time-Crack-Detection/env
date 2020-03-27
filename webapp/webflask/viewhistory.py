from __init__ import request
from sqlalchemy import text

# 쿼리해야할 정보 : title, save_date, comment
def view_history(session, code):
    code= "1"  # 임시 코드
    title = session.query('title').from_statement(text("select * from image where code = " + code)).all()
    save_date = session.query('save_date').from_statement(text("select * from image user where code = " + code)).all()
    path = session.query('path').from_statement(text("select * from image where code = " + code)).all()
    comment = session.query('comment').from_statement(text("select * from image where code = " + code)).all()

    result_data = []
    for i in range(len(title)):
        result_data.append([])
        result_data[i].append(title[i])
        result_data[i].append(save_date[i])
        result_data[i].append(path[i])
        result_data[i].append(comment[i])
    #print(result_data)

    return result_data