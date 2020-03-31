import cv2 as cv
import os

SAVE_CANDIDATES_PATH = './Candidates'
if not os.path.exists(SAVE_CANDIDATES_PATH):
    os.mkdir(SAVE_CANDIDATES_PATH)

class Candidate:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def findCandidates_canny(imgPath):
    candidates = []
    image = cv.imread(imgPath, cv.IMREAD_COLOR)
    copy = image.copy()
    image2 = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)
    # Image read and convert Gray Level.

    blur = cv.GaussianBlur(image2, (3, 3), 0)
    # It is need to run canny edge function
    canny = cv.Canny(blur, 125, 255)
    # canny edge detection

    contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # contours are possible which crack candidates
    # one of contours consist of points(x, y) and rect size(w, h)

    print('Find contours : ', len(contours))
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv.contourArea(cnt)
        x, y, w, h = cv.boundingRect(cnt)
        rect_area = w * h
        float_w = float(w)
        aspect_ratio = float_w / h

        if (aspect_ratio >= 0.6) and (rect_area >= 200):
            # This if can effectively remove unnecessary contours
            c = Candidate(x, y, w, h)
            candidates.append(c)
            # candidates store
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # drawing the rectangle in the candidate areas in the image

    # 밑의 코드는 효과적으로 작동하지 않음.
    # duplited_candidates = []
    # # 겹치는 후보들 제거하는 로직
    # for c in candidates:
    #     x_area = c.x + c.w
    #     y_area = c.y + c.h
    #     for i in range(0, len(candidates)):
    #         cx_area = candidates[i].x + candidates[i].w
    #         cy_area = candidates[i].y + candidates[i].h
    #
    #         if (candidates[i].x <= c.x) and (abs(candidates[i].x - c.x) <= 5) and \
    #                 (candidates[i].y >= c.y) and (abs(candidates[i].y - c.y) <= 5) and \
    #                 (x_area <= cx_area) and (y_area <= cy_area):
    #             cv.rectangle(image, (c.x, c.y), (c.x + c.w, c.y + c.h), (0, 255, 0), 1)
    #             duplited_candidates.append(c)
    #             break
    # for i in duplited_candidates:
    #     candidates.remove(i)

    print('Result candidates : ', len(candidates))
    cv.imshow('Canny image', canny)
    cv.imshow('Original image', image)
    cv.waitKey()
    return candidates

def save_candidates(imgPath, method_name, candidates):
    '''
    Candidates save to image
    :param imgPath: Object image path
    :param method_name: Types of method ex. canny, suft and so on.
    :param candidates: extracted candidates list
    :return: void
    '''

    TEMP = imgPath.split('/')[-1]
    FINAL_SAVE_PATH = './Candidates/' + TEMP[:-4]
    print(FINAL_SAVE_PATH)
    if not os.path.exists(FINAL_SAVE_PATH):
        os.mkdir(FINAL_SAVE_PATH)

    image = cv.imread(imgPath, cv.IMREAD_COLOR)
    index = 0
    for c in candidates:
        SAVE_NAME = FINAL_SAVE_PATH + '/' + method_name + str(index) + '.jpg'
        print('Save name : ', SAVE_NAME)

        start_y = c.y
        end_y = c.y + c.h
        start_x = c.x
        end_x = c.x + c.w

        roi = image[start_y:end_y, start_x:end_x]

        cv.imwrite(SAVE_NAME, roi)
        index += 1

# CRACK_PATH = './cracks/concrete-crack-texture'
# # cracks = os.listdir(CRACK_PATH)
#
# canny_candidates = findCandidates_canny('Brick2.jpg')
# save_candidates(CRACK_PATH + '/Brick2.jpg' )
#
# # for c in cracks:
# #     obj = CRACK_PATH + '/' + c
# #     print(obj)
# #     canny_candidates = findCandidates_canny(obj)
# #     save_candidates(obj, 'canny', canny_candidates)

