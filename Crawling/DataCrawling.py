from selenium import webdriver
import time
import os
from selenium.webdriver.common.keys import Keys
from urllib.request import urlopen

# It requires some modules.
# pip install selenuim
# pip install urllib
# pip install bs4

DEFAULT_PATH = './cracks'

browser = webdriver.Chrome('chromedriver.exe')
# Need to Chrome driver - https://chromedriver.chromium.org/downloads
count = 0

keyword = 'cracks' # search keyword
page_down_count = 120

photo_list = []
result = []
url = "https://www.google.co.in/search?q=" + keyword + "&source=lnms&tbm=isch"


if not os.path.exists(DEFAULT_PATH):
    os.mkdir(DEFAULT_PATH)
if not os.path.exists(DEFAULT_PATH + '/' + keyword):
    os.mkdir(DEFAULT_PATH + '/' + keyword)

# url open
browser.get(url)
time.sleep(1)


body = browser.find_element_by_tag_name('body')

for i in range(0, page_down_count):
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.1)

    # if browser.find_element_by_class_name("mye4qd") is not None:
    #     browser.find_element_by_class_name("mye4qd").click()
    #     time.sleep(2)



photo_list = browser.find_elements_by_tag_name("img.rg_i")

print('Find object : ', len(photo_list))
index = 0
for i in range(0, len(photo_list)):
    try:
        photo_list[i].click()
        print(photo_list[i])

        time.sleep(2)


        html_objects = browser.find_element_by_css_selector('#Sva75c > div > div > div.pxAole > div.tvh9oe.BIB1wf > div > div.OUZ5W > div.zjoqD > div > div.v4dQwb > a > img')
        current_src = html_objects.get_attribute('src')
        print("src :" + current_src)

    except Exception as e:
        print(e)

    try:
        t = urlopen(current_src).read()
        print(t)
        if t is None :
            continue
        
        filename ="{}/{}/{}{}{}".format(DEFAULT_PATH, keyword,keyword, str(index), ".jpg")
        with open(filename, "wb") as f:
            f.write(t)
            index += 1
            current_src = ""
            print("Img Save Success")
        index += 1
    except Exception as e:
        print(e)


    exit_button = browser.find_element_by_css_selector('#Sva75c > div > div > div:nth-child(2) > a > div > svg > polygon')
    exit_button.click()

    time.sleep(1)

print('Dataset Crawling complete..')
browser.close()
