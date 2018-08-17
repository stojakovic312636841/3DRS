from PIL import Image
import random
import os,time

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

BLOCK_SIZE = 4


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
		self.if_img = img
		self.pix = self.img.load()
		self.prev_pix = None
		self.if_pix = self.if_img.load()

		self.width, self.height = img.size
		self.blockSize = size


	def writeMotion2Buffer(self,mvBuffer):
		for x in range(int(self.width/BLOCK_SIZE)):
			for y in range(int(self.height/BLOCK_SIZE)):
				#print(x,y)
				self.calMotionVector(x,y,mvBuffer)


	def calMotionVector(self,x,y,mvBuffer):
		min_sad = CONST_MAX_SAD
		max_sad = 0

		mvFromS0 = mvBuffer.getMotion(x+CONST_SPATIAL_CAND0[0],y+CONST_SPATIAL_CAND0[1]),CONST_S0_SAD_PENALTY,CONST_S0_DETAIL_PENALTY
		mvFromS1 = mvBuffer.getMotion(x+CONST_SPATIAL_CAND1[0],y+CONST_SPATIAL_CAND1[1]),CONST_S1_SAD_PENALTY,CONST_S1_DETAIL_PENALTY
		mvFromT0 = mvBuffer.getMotion(x+CONST_TEMPORAL_CAND0[0],y+CONST_TEMPORAL_CAND0[1]),CONST_T0_SAD_PENALTY,CONST_T0_DETAIL_PENALTY
		mvFromT1 = mvBuffer.getMotion(x+CONST_TEMPORAL_CAND1[0],y+CONST_TEMPORAL_CAND1[1]),CONST_T1_SAD_PENALTY,CONST_T1_PENALTY_PENALTY
		#print(mvFromS0)  #((0, 0), 44, 4)

		#From S0 to get S0+U0
		if mvBuffer.getMotion(x+CONST_SPATIAL_CAND0[0],y+CONST_SPATIAL_CAND0[1])!= None:
			mvFromU0 = (mvBuffer.getMotion(x+CONST_SPATIAL_CAND0[0],y+CONST_SPATIAL_CAND0[1])[0]+random.randint(-2,2)*BLOCK_SIZE,
						mvBuffer.getMotion(x+CONST_SPATIAL_CAND0[0],y+CONST_SPATIAL_CAND0[1])[1]+random.randint(-2,2))*BLOCK_SIZE,CONST_RAND0_PENALTY,0
			#print(mvFromU0)	#((7, 5), 60, 0)
		else:
			mvFromU0 = None

		#From S1 to get S1+U1
		if mvBuffer.getMotion(x+CONST_SPATIAL_CAND1[0],y+CONST_SPATIAL_CAND1[1])!= None:
			mvFromU1 = (mvBuffer.getMotion(x+CONST_SPATIAL_CAND1[0],y+CONST_SPATIAL_CAND1[1])[0]+random.randint(-2,2)*BLOCK_SIZE,
						mvBuffer.getMotion(x+CONST_SPATIAL_CAND1[0],y+CONST_SPATIAL_CAND1[1])[1]+random.randint(-2,2)*BLOCK_SIZE),CONST_RAND1_PENALTY,0
			#print(mvFromU1)	#((7, 5), 60, 0)
		else:
			mvFromU1 = None

		mv_cand = [mvFromS0,mvFromS1,mvFromT0,mvFromT1,mvFromU0,mvFromU1,((0,0),CONST_ZMV_SAD_PENALTY,0)]	#(0,0) --> 0 vector
		#print(x,y,mv_cand) #(12, 40, [((6, 0), 44, 4), ((0, 0), 44, 4), ((0, 0), 56, 5), ((0, 0), 56, 5), ((-2, -6), 60, 0), ((0, 4), 114, 0), ((0, 0), 48, 0)])
		final_mv = mvBuffer.getMotion(x,y)
		#print(x,y,final_mv)	#(111, 99, (0, 0))

		sad_list = [None] * 7
		for index , cand in enumerate(mv_cand):
			#print(index ,cand )
			if cand!= None and cand[0] != None: #cand[0] --> (6,0)
				#tmp	 = self.getSAD(x*BLOCK_SIZE,y*BLOCK_SIZE,cand[0][0],cand[0][1],cand[1],cand[2])
				tmp	 = self.getOnlySAD(x*BLOCK_SIZE,y*BLOCK_SIZE,cand[0][0],cand[0][1],cand[1],cand[2])
				sad_list[index] = tmp
				#print(x*BLOCK_SIZE,y*BLOCK_SIZE,tmp)
				if tmp >= max_sad:
					max_sad = tmp
				#if tmp < min_sad:
					#min_sad = tmp
					#find the min SAD as the final motion vector
					#final_mv = cand[0]
		#print(x,y,sad_list)	#(161, 29, (0, 0))
		
		for index , sad in enumerate(sad_list):
			if sad != None:	
				if index ==2 or index == 3:
					sad += max_sad*0.008
				if index ==4 or index == 5:
					sad += max_sad*0.004
				if index ==6:
					sad += max_sad*0.016
				if sad < min_sad:
					min_sad = sad
					final_mv = mv_cand[index][0]
		#print(max_sad,min_sad,mv_cand[index][0])					

		mvBuffer.writeMV(x,y,final_mv)

		for a in range(BLOCK_SIZE):
			for b in range(BLOCK_SIZE):
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
				#self.pix[x*BLOCK_SIZE+a,y*BLOCK_SIZE+b] = (self.pix[x*BLOCK_SIZE+a,y*BLOCK_SIZE+b][0],cb,cr)
				#MC
				#Direct algorithm
				#self.pix[x*BLOCK_SIZE+a,y*BLOCK_SIZE+b] = self.prev_pix[x*BLOCK_SIZE+a+final_mv[0],y*BLOCK_SIZE+b+final_mv[1]]
				#Map algorithm
				tmpx = int(0.5*final_mv[0])
				tmpy = int(0.5*final_mv[1])
				index_x = x*BLOCK_SIZE+a+tmpx
				index_y = y*BLOCK_SIZE+b+tmpy
				index_x_ = x*BLOCK_SIZE+a-tmpx
				index_y_ = y*BLOCK_SIZE+b-tmpy
				if index_x < 0 or index_x >self.width-1 or index_y < 0 or index_y >self.height-1 or index_x_ < 0 or index_x_ >self.width-1 or index_y_ < 0 or index_y_ >self.height-1:
				   pass 
				else:				   
				   tmp0 = self.prev_pix[x*BLOCK_SIZE+a+tmpx,y*BLOCK_SIZE+b+tmpy]
				   tmp1 = self.pix[x*BLOCK_SIZE+a-tmpx,y*BLOCK_SIZE+b-tmpy]
				   sum = (int(0.5*(tmp0[0]+tmp1[0])),int(0.5*(tmp0[1]+tmp1[1])),int(0.5*(tmp0[2]+tmp1[2])))
				   self.if_pix[x*BLOCK_SIZE+a,y*BLOCK_SIZE+b] = sum
				#print(self.pix[x*BLOCK_SIZE+a,y*BLOCK_SIZE+b])
				#os._exit(0)

	def getOnlySAD(self,x,y,motion_x,motion_y,penalty,detail_penalty):
	
		predict_x = x +motion_x
		predict_y = y +motion_y

		if (x <0 or x > self.width-BLOCK_SIZE*2 or predict_x <0 or predict_x > self.width-BLOCK_SIZE*2):
			return CONST_MAX_SAD
		if (y <0 or y > self.height-BLOCK_SIZE*2 or predict_y <0 or predict_y > self.height-BLOCK_SIZE*2):
			return CONST_MAX_SAD

		sad = 0

		#4*4 block
		for a in range(BLOCK_SIZE*2):
			for b in range(BLOCK_SIZE*2):
				#only lum layer	 --> pix[x+a,y+b] is current block ; prev_pix[predict_x+a,predict_y+b] is t-1 pre-block
				sad = sad + abs(self.pix[x+a,y+b][0]-self.prev_pix[predict_x+a,predict_y+b][0])


		return sad
		
	
	def getSAD(self,x,y,motion_x,motion_y,penalty,detail_penalty):
		
		predict_x = x +motion_x
		predict_y = y +motion_y

		if motion_x == 0 and motion_y == 0:
			penalty = CONST_ZMV_SAD_PENALTY #48
			detail_penalty = CONST_ZMV_DETAIL_PENALTY #0

		if (x <0 or x > self.width-BLOCK_SIZE*2 or predict_x <0 or predict_x > self.width-BLOCK_SIZE*2):
			return CONST_MAX_SAD
		if (y <0 or y > self.height-BLOCK_SIZE*2 or predict_y <0 or predict_y > self.height-BLOCK_SIZE*2):
			return CONST_MAX_SAD

		sad = 0
		detail = 0
		prev = 0

		#4*4 block
		for a in range(BLOCK_SIZE*2):
			for b in range(BLOCK_SIZE*2):
				#only lum layer	 --> pix[x+a,y+b] is current block ; prev_pix[predict_x+a,predict_y+b] is t-1 pre-block
				sad = sad + abs(self.pix[x+a,y+b][0]-self.prev_pix[predict_x+a,predict_y+b][0])
				if (a !=0):
					#only lum layer	 -->  pix Smoothness
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
image_name = "./SDBronze/SDBronze"
img_cur = None
width,height = 720,480
#mvBuffer.size = 180,120
mvBuffer = MotionMap(int(width/BLOCK_SIZE),int(height/BLOCK_SIZE))

for i in range(2,100):
	#break;
	cur_name = image_name+ str(i).zfill(4)+".bmp"
	start_time = time.time()
	print(cur_name)

	img_prev = img_cur
	img_cur = MyImage(Image.open(cur_name).convert('YCbCr'),8)

	if img_prev != None:
		#img_cur.setPrev(img_prev)
		img_cur.setPrev(MyImage(Image.open(image_name+ str(i-1).zfill(4)+".bmp").convert('YCbCr'),8))
		width, height = img_cur.getImg().size
		#key step
		img_cur.writeMotion2Buffer(mvBuffer)

		#img_mv = img_cur.img.convert('RGB')
		img_if = img_cur.if_img.convert('RGB')
		img_if.save("./MVBronze/MV_SDBronze"+str(i).zfill(4)+"_if.bmp")
		#img_if.save("./test/MV_SDBronze"+str(i-1).zfill(4)+"_test.bmp")		
		#img_mv.save("./MVBronze/MV_SDBronze"+str(i).zfill(4)+".bmp")
		#print(mvBuffer.mvBuffer)
	print('the cost time is %f'%(time.time() - start_time))
