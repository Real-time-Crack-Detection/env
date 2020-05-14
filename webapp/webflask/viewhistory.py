from sqlalchemy import text

# 쿼리해야할 정보 : title, save_date, comment
def view_history(session, code):
    if code == "":  # 로그인 없이 창을 켰을때
        code = '-1'

    title1 = session.query('title1').from_statement(text("select * from image where code = " + code)).all()
    save_date = session.query('save_date').from_statement(text("select * from image user where code = " + code)).all()
    path = session.query('path').from_statement(text("select * from image where code = " + code)).all()
    comment = session.query('comment').from_statement(text("select * from image where code = " + code)).all()
    title2 = session.query('title2').from_statement(text("select * from image where code = " + code)).all()

    result_data = []

    for i in range(len(title1)):
        result_data.append([])
        result_data[i].append(title1[i])
        result_data[i].append(save_date[i])
        result_data[i].append(path[i])
        result_data[i].append(comment[i])
        result_data[i].append(title2[i])
    return result_data