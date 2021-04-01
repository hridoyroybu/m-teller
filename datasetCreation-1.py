import cv2
import os
import imutils
import time
import csv

alg = "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(alg)

Name = str(input("Enter your Name: "))
Roll_Number = int(input("Enter your Roll Number: "))

dataset = 'dataset'
sub_data = Name
path = os.path.join(dataset,sub_data)

if not os.path.isdir(path):
    os.mkdir(path)
    print(sub_data)

info = [str(Name) , str(Roll_Number)]
with open ('student.csv','a') as csvFile:
    write = csv.writer(csvFile)
    write.writerow(info)
csvFile.close()

print("Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)
total_pic_taken = 0

while (total_pic_taken < 50):
    frame = cam.read()[1]
    img = imutils.resize(frame , width = 400)
    grayImg = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(grayImg ,scaleFactor= 1.1, minNeighbors=5,minSize=(30,30))

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        p = os.path.sep.join([path,"{}.png".format(str(total_pic_taken).zfill(5))])
        cv2.imwrite(p,img)
        #onlyFace = grayImg[y:y+h,x:x+w]
        #resizeImg = cv2.resize(onlyFace,(width,height))
        #cv2.imwrite("%s/%s.jpg"%(path,total_pic_taken),resizeImg)
        total_pic_taken+=1
        #print(total_pic_taken)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
