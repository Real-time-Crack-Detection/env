from bs4 import BeautifulSoup
import cv2
import os
import time
from models import *

def save(session, code):
    code = code[1:-1]
    html = open("Templates/real-time.html",encoding="UTF8").read()
    soup = BeautifulSoup(html,"html.parser")

    img1 = soup.find(id="crack1")
    img2 = soup.find(id="crack2")

    try:
        cap = cv2.VideoCapture(img1["src"])
        a, im1 = cap.read()
        im2 = cv2.imread(img2["src"])

        count = os.listdir("./static/img/images")

        cv2.imwrite("static/img/images/title"+str(int(len(count)/2+1))+".jpg", im1)
        cv2.imwrite("static/img/images/title"+str(int(len(count)/2+1))+"-1.jpg", im2)


        path = "static/img/images"
        date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        title1 = "title"+str(int(len(count)/2+1))+".jpg"
        title2 = "title"+str(int(len(count)/2+1))+"-1.jpg"
        new_image = Image(path=path, save_date=date, code=code, comment="", title1=title1, title2=title2)
        session.add(new_image)
        session.commit()
        print("record insert finish")
    except:
        print("record insert error")
