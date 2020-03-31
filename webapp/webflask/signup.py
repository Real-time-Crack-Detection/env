from __init__ import request
from models import *

def sign_up(session):
    id = request.form['defaultRegisterFormID']
    password = request.form['defaultRegisterFormPassword']
    name = str(request.form['defaultRegisterFormFirstName']) + " " + str(request.form['defaultRegisterFormLastName'])
    email = request.form['defaultRegisterFormEmail']
    phonenumber = request.form['defaultRegisterPhonenumber']
    code = request.form['defaultRegisterFormCode']

    try:
        new_user = User(id=id, password=password, name=name, email=email, phonenumber=phonenumber, code=code)
        session.add(new_user)
        session.commit()
        print("record insert finish")
    except:
        print("record insert error")
