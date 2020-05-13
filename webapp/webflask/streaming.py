import cv2
import time
import datetime
import numpy
import os


def streaming():
    cap = cv2.VideoCapture("testvideo.webm")

    cap.set(3, 500)
    cap.set(4, 500)
    prevTime = 0
    record = False

    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
    length = int(cv2.VideoCapture.get(cap, property_id))

    if cap.isOpened() == False:
        print("can't open the Cam")
        exit()

    while (True):

        success, frame = cap.read()
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime

        cv2.imwrite("static\img\\testimg.jpg", frame)
#        print(os.path)
        #    now = datetime.datetime.now().strftime("%d-%H-%M-%s")
        try:
            fps = 1 / (sec)
        except:
            pass

#        print("Time={0}".format(sec))
#        print("Frame rate(fps) ={0}".format(fps))

        str = "FPS : %0.1f" % fps

        cv2.putText(frame, str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        if success == True:

            cv2.imshow('Camera Window', frame)
            key = cv2.waitKey(1) & 0xFF

            if (key == 27):
                break

            elif (key == 26):
                print("capture")
                cv2.imwrite("home/mjpg/" + str(now) + ".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100])

            elif (key == 24):
                print("video recording start")
                video = cv2.VideoWriter(str(now) + ".avi", fourcc, 20.0, (frame.shape[1], frmae.shape[0]))
            elif (key == 3):
                print("video recording stop")
                record = False
                video.release()

            if record == True:
                print("video recording.....")
                video.write(frame)

    print(length)

    cap.release()
    cv2.destroyAllWindows()
