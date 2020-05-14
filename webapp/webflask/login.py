from flask import request
from sqlalchemy import text

def log_in(session):

    id = "'" + str(request.form['defaultRegisterLoginID']) + "'"
    password = request.form['defaultRegisterLoginPassword']

    try:
        query = session.query('password').from_statement(text("select * from user where id = "+ id)).all()
        name = str(session.query('name').from_statement(text("select * from user where id = "+ id)).all()[0])[2:-3]
        code = session.query('code').from_statement(text("select * from user where id = "+ id)).all()
        code = "'" + str(code[0])[2:-3] + "'"
        # id, password 일치시 이름을 리턴, 불일치시 False 리턴
        if str(query[0])[2:-3] == password:
            print("login")
        else:  # id는 맞지만 password가 틀렸을때
            print("login fail")
            name = "Sign In"
    except:  # id가 틀렸을 때
        name = "Sign In"

    finally:
        return name, code

#def log_in_fail():