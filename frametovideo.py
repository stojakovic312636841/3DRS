# encoding: UTF-8
import glob as gb
import cv2
import os
import numpy as np


img_path = gb.glob("D:\\testPic\\MVBronze\\*.bmp") 

videoWriter = cv2.VideoWriter('D:\\testPic\\MVBronze\\testtt.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 30, (1440,480))  #cv2.VideoWriter_fourcc(*'MJPG')

reader = cv2.VideoCapture("D:\\testPic\\MVBronze\\Oigin_SDBronze.mp4")
framenum = 2
ret,img2 = reader.read()    #read first
ret,img2 = reader.read()    #read second   
for path in img_path:
    img  = cv2.imread(path) 
    print path
    #img = cv2.resize(img,(720,480))
    
    if framenum%2==1:
    	ret,img2 = reader.read()
    img3=np.concatenate((img,img2),axis=1)
	#cv2.imwrite("D://testPic//MVBronze//img3//"+str(framenum)+".bmp",img3)
    framenum = framenum + 1

    #cv2.imshow('im3',img3)
    videoWriter.write(img3)
    




