from PIL import Image
import random
import os

#Constants
CONST_MAX_SAD = 16384
CONST_S0_SAD_PENALTY = 44
CONST_S0_DETAIL_PENALTY = 4
CONST_S1_SAD_PENALTY = 44
CONST_S1_DETAIL_PENALTY = 4
CONST_ZMV_SAD_PENALTY = 48
CONST_ZMV_DETAIL_PENALTY = 0
CONST_T0_SAD_PENALTY = 56
CONST_T0_DETAIL_PENALTY = 5
CONST_T1_SAD_PENALTY = 56
CONST_T1_PENALTY_PENALTY = 5
CONST_RAND0_PENALTY = 60
CONST_RAND1_PENALTY = 114

CONST_SPATIAL_CAND0 = (-1,-1)
CONST_SPATIAL_CAND1 = (1,-1)
CONST_TEMPORAL_CAND0 = (-2,2)
CONST_TEMPORAL_CAND1 = (2,2)


class MotionMap():
    def __init__(self,width,height):
        self.width = width
        self.height = height
        self.mvBuffer = [[(0,0) for y in range(height)] for x in range(width)]
    
    def getMotion(self,x,y):
        if x in range(self.width) and y in range(self.height):
            return self.mvBuffer[x][y]
        else:
            return None

    def writeMV(self,x,y,final_mv):
        self.mvBuffer[x][y] = final_mv


class MyImage():
    def __init__(self,img,size):
        self.img = img
        self.pix = self.img.load()
        self.prev_pix = None

        self.width, self.height = img.size
        self.blockSize = size


    def writeMotion2Buffer(self,mvBuffer):
        for x in range(int(self.width/4)):
            for y in range(int(self.height/4)):
                #print(x,y)
                self.calMotionVector(x,y,mvBuffer)


    def calMotionVector(self,x,y,mvBuffer):
        sad = CONST_MAX_SAD

        mvFromS0 = mvBuffer.getMotion(x+CONST_SPATIAL_CAND0[0],y+CONST_SPATIAL_CAND0[1]),CONST_S0_SAD_PENALTY,CONST_S0_DETAIL_PENALTY
        mvFromS1 = mvBuffer.getMotion(x+CONST_SPATIAL_CAND1[0],y+CONST_SPATIAL_CAND1[1]),CONST_S1_SAD_PENALTY,CONST_S1_DETAIL_PENALTY
        mvFromT0 = mvBuffer.getMotion(x+CONST_TEMPORAL_CAND0[0],y+CONST_TEMPORAL_CAND0[1]),CONST_T0_SAD_PENALTY,CONST_T0_DETAIL_PENALTY
        mvFromT1 = mvBuffer.getMotion(x+CONST_TEMPORAL_CAND1[0],y+CONST_TEMPORAL_CAND1[1]),CONST_T1_SAD_PENALTY,CONST_T1_PENALTY_PENALTY
        #print(mvFromS0)  #((0, 0), 44, 4)

		#From S0 to get S0+U0
        if mvBuffer.getMotion(x+CONST_SPATIAL_CAND0[0],y+CONST_SPATIAL_CAND0[1])!= None:
            mvFromU0 = (mvBuffer.getMotion(x+CONST_SPATIAL_CAND0[0],y+CONST_SPATIAL_CAND0[1])[0]+random.randint(-8,8),
                        mvBuffer.getMotion(x+CONST_SPATIAL_CAND0[0],y+CONST_SPATIAL_CAND0[1])[1]+random.randint(-8,8)),CONST_RAND0_PENALTY,0
            #print(mvFromU0)	#((7, 5), 60, 0)
        else:
            mvFromU0 = None

		#From S1 to get S1+U1
        if mvBuffer.getMotion(x+CONST_SPATIAL_CAND1[0],y+CONST_SPATIAL_CAND1[1])!= None:
            mvFromU1 = (mvBuffer.getMotion(x+CONST_SPATIAL_CAND1[0],y+CONST_SPATIAL_CAND1[1])[0]+random.randint(-8,8),
                        mvBuffer.getMotion(x+CONST_SPATIAL_CAND1[0],y+CONST_SPATIAL_CAND1[1])[1]+random.randint(-8,8)),CONST_RAND1_PENALTY,0
            #print(mvFromU1)	#((7, 5), 60, 0)
        else:
            mvFromU1 = None

        mv_cand = [mvFromS0,mvFromS1,mvFromT0,mvFromT1,mvFromU0,mvFromU1,((0,0),CONST_ZMV_SAD_PENALTY,0)]	#(0,0) --> 0 vector
        #print(x,y,mv_cand)	#(12, 40, [((6, 0), 44, 4), ((0, 0), 44, 4), ((0, 0), 56, 5), ((0, 0), 56, 5), ((-2, -6), 60, 0), ((0, 4), 114, 0), ((0, 0), 48, 0)])
        final_mv = mvBuffer.getMotion(x,y)
        #print(x,y,final_mv)	#(111, 99, (0, 0))

        for cand in mv_cand:
            if cand!= None and cand[0] != None:	#cand[0] --> (6,0)
                tmp  = self.getSAD(x*4,y*4,cand[0][0],cand[0][1],cand[1],cand[2])
                #print(x*4,y*4,tmp)
                if tmp < sad:
                    sad = tmp
                    #find the min SAD as the final motion vector
                    final_mv = cand[0]
        #print(x,y,final_mv)	#(161, 29, (0, 0))
        mvBuffer.writeMV(x,y,final_mv)

        for a in range(4):
            for b in range(4):
                #print(x,x+a,y,y+b,self.pix[x*4+a,y*4+b])
                cb = 128+final_mv[0]*16
                if cb >= 255:
                    cb = 255
                if cb <0:
                    cb = 0;
                cr = 128+final_mv[1]*16
                if cr >= 255:
                    cr = 255
                if cr <0:
                    cr = 0;
                #Mark the change
                #self.pix[x*4+a,y*4+b] = (self.pix[x*4+a,y*4+b][0],cb,cr)



    def getSAD(self,x,y,motion_x,motion_y,penalty,detail_penalty):
        
        predict_x = x +motion_x
        predict_y = y +motion_y

        if motion_x == 0 and motion_y == 0:
            penalty = CONST_ZMV_SAD_PENALTY	#48
            detail_penalty = CONST_ZMV_DETAIL_PENALTY #0

        if (x <0 or x > self.width-8 or predict_x <0 or predict_x > self.width-8):
            return CONST_MAX_SAD
        if (y <0 or y > self.height-8 or predict_y <0 or predict_y > self.height-8):
            return CONST_MAX_SAD

        sad = 0
        detail = 0
        prev = 0

		#8*8 block
        for a in range(8):
            for b in range(8):
				#only lum layer  --> pix[x+a,y+b] is current block ; prev_pix[predict_x+a,predict_y+b] is t-1 pre-block
                sad = sad + abs(self.pix[x+a,y+b][0]-self.prev_pix[predict_x+a,predict_y+b][0])
                if (a !=0):
                    #only lum layer  -->  pix Smoothness
                    detail = detail+abs(self.pix[x+a,y+b][0]-prev)
                    #print(a,b,prev,abs(self.pix[x+a,y+b][0]-prev))
                prev = self.pix[x+a,y+b][0]


        if detail_penalty !=0:
            detail = detail/detail_penalty
            #print(penalty,detail,sad,penalty-detail,sad+penalty-detail)
        else:
            detail = 0

        return sad+penalty-detail
    

    def setPrev(self,prev):
        self.prev_pix = prev.load()


    def convert(self,opt):
        return self.img.convert(opt)


    def load(self):
        return self.img.load()


    def getImg(self):
        return self.img 

#image_name = "SDHQV/SDHQV"
image_name = "SDBronze/SDBronze"
img_cur = None
width,height = 720,480
#mvBuffer.size = 180,120
mvBuffer = MotionMap(int(width/4),int(height/4))

for i in range(2,100):
    #break;
    cur_name = image_name+ str(i).zfill(4)+".bmp"
    print(cur_name)

    img_prev = img_cur
    img_cur = MyImage(Image.open(cur_name).convert('YCbCr'),8)

    if img_prev != None:
        img_cur.setPrev(img_prev)
        width, height = img_cur.getImg().size
		#key step
        img_cur.writeMotion2Buffer(mvBuffer)

        img_mv = img_cur.img.convert('RGB')
        img_mv.save("MVBronze/MV_SDBronze"+str(i).zfill(4)+".bmp")
        #print(mvBuffer.mvBuffer)
        #os._exit(0)

