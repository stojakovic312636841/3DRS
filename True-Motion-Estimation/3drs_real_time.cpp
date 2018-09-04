#include <opencv2/opencv.hpp>

#include <math.h> 
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <boost/format.hpp>
#include <ratio>

#include <string.h>
#include <thread>

#include <smmintrin.h>


using namespace std;
using namespace cv;

int BLOCKSIZE = 4;
int CPU_THREAD = 1;
int DISTANCE_THEROSHOLD = 80;
int MV_THRESHOLD = 80;
int ERROR_VALUE = 65535;

float get_max(float *input, int size)
{
    float max_sad = -1.;
    for (int i = 0; i < size; i++)
    {
        if ( *(input+i) > max_sad )
        {
            max_sad = *(input + i);
        }
    }
    return max_sad;
}

//motion from src to dist
//point(x,y) means (col, row)
int get_sad(uchar *src, uchar *dist, int dx, int dy, int sx, int sy, int col)
{
    int block_size(BLOCKSIZE);
    sx -= block_size/2; sy -= block_size/2;
    dx -= block_size/2; dy -= block_size/2;
/*
	//left top point of the block
    uchar *pCur, *pRef, *tmpCur, *tmpRef;
    pCur = (dist + dx*col + dy);
    pRef = (src + sx*col + sy);

	int sum_sad(0);
	for (int a = 0; a < block_size; a++)
	{
		for (int b = 0; b< block_size; b++)
		{
			tmpCur = pCur + a*col+b;
			tmpRef = pRef + a*col+b;
			sum_sad += fabs(*tmpCur - *tmpRef);
		}
	}
	//cout << ":" << sum_sad << endl;
	return sum_sad;
*/


    __m128i s0, s1, s2;
    s2 = _mm_setzero_si128();

    uchar *pCur, *pRef;
    pCur = (dist + dy*col + dx);
    pRef = (src + sy*col + sx);

    int sad_test(0);
    for (int row = 0; row < block_size; row++)
    {
        s0 = _mm_loadu_si128((__m128i*) pRef);
        s1 = _mm_loadu_si128((__m128i*) pCur);
        s2 = _mm_sad_epu8(s0, s1);
        sad_test += int(*((uint16_t*)&s2));

        pCur += col;
        pRef += col;
    }
    return sad_test;

}

inline float norm(int *point)
{
    return sqrt( pow(float(*(point) ), 2) + pow(float( *(point + 1) ),2) );
}


//point(x,y) means (col, row)
void oaat(uchar *src, uchar *dist, int *output, int *center, int *block_center, int searching_area, int *updates, int src_rows, int src_cols)
{
    int block_size(BLOCKSIZE), radius(block_size/2);
    output[0] = *(block_center);
    output[1] = *(block_center+1);

    while (true)
    {
        int cs[6] = { *(block_center) + *(updates)*block_size,      *(block_center + 1) + *(updates + 1)*block_size,
                      *(block_center) + *(updates + 2)*block_size,  *(block_center + 1) + *(updates + 3)*block_size,
                      *(block_center) + *(updates + 4)*block_size,  *(block_center + 1) + *(updates + 5)*block_size,
        };

		//printf("block.center = %d,%d\n",*(block_center),*(block_center+1));
        int min_SAD = 100000;
        int update[2] = {0,0};
        int decision(-1);
        for (int i = 0; i < 3; i++) //3 candidate
        {
            //cs[i*2] -> x, cs[i*2+1] -> y
            if (cs[i*2] < radius or cs[i*2+1] < radius or cs[i*2] > src_cols - radius or cs[i*2+1] > src_rows - radius)
                continue;

            int sad = get_sad(src, dist, *center, *(center+1), cs[i*2], cs[i*2+1], src_cols);
            if (sad < min_SAD)
            {
                min_SAD = sad;
                update[0] = cs[i*2];
                update[1] = cs[i*2 + 1];
                decision = i;
            }

            //if choose center of 3 point, then it should be considered as converged	-->	Point(0,0)
            if (sad < min_SAD + 40. and i == 1)
            {
                output[0] = cs[i*2];
                output[1] = cs[i*2 + 1];
                return;
            }
        }

        //Point offset(update - center);
        int offset[2] = {update[0] - *center, update[1] - *(center+1)};
		//printf("offset = %d,%d\n",offset[0],offset[1]);

        if (abs(offset[0]) > searching_area or abs(offset[1]) > searching_area)
        {
            output[0] = update[0];
            output[1] = update[1];
            break;
        }

        if (update[0] == *block_center and update[1] == *(block_center+1))
        {
            output[0] = update[0];
            output[1] = update[1];
            break;
        }

        *block_center = update[0];
        *(block_center + 1) = update[1];
    }
}


//cal the round block score (8 block)
void cal_block_score(int ** array_map, int candidate)
{
	//sort --> x
	for (int i =0; i < candidate-1; i++)
	{
		if(array_map[i][1] > array_map[i+1][1])
		{
			for (int j = 0; j<4; j++)
			{
				int temp = array_map[i][j];
				array_map[i][j] = array_map[i+1][j];
				array_map[i+1][j] = temp;
			}

		}
	}
	
	//add socre
	for (int i =0; i < candidate; i++)
	{
		array_map[i][3] = fabs(i-(candidate)/2);
	}

	//sort --> y
	for (int i =0; i < candidate-1; i++)
	{
		if(array_map[i][2] > array_map[i+1][2])
		{
			for (int j = 0; j<4; j++)
			{
				int temp = array_map[i][j];
				array_map[i][j] = array_map[i+1][j];
				array_map[i+1][j] = temp;
			}

		}
	}

	//add socre
	for (int i =0; i < candidate; i++)
	{
		array_map[i][3] += fabs(i-(candidate)/2);
	}

	//get the vector of the max socre
	for (int i =0; i < candidate-1; i++)
	{
		if(array_map[i][3] > array_map[i+1][3])
		{
			for (int j = 0; j<4; j++)
			{
				int temp = array_map[i][j];
				array_map[i][j] = array_map[i+1][j];
				array_map[i+1][j] = temp;
			}

		}
	}

}


//pmm[y,x]
void MV_without_abnormal(int *pmm, int src_rows, int src_cols)
{
    int block_size(BLOCKSIZE);
    int searching_area(block_size*block_size), radius(block_size/2);
    int block_row(src_rows/block_size), block_col(src_cols/block_size);

	for (int r = 0; r < block_row; r++)
	{
		for (int c = 0; c < block_col; c++)
		{
			//frame boundary
			if (r == 0 or c == 0 or c == block_col-1 or r == block_row-1)
			{

				//round 8 block --> 
				int cs[16] = {-1,-1,-1,0,-1,1,0,-1,0,1,1,-1,1,0,1,1};
				int candidate = 0;

				//init 								
				int **array_map = new int*[8]; //[i][x][y][socre]
				for (int i=0;i<8;i++)
				{
					array_map[i] = new int[4];
				}
		
				//3*3 filter and assign value	
				for (int i=0; i<8;i++)
				{
					int block_r = r+cs[i*2];
					int block_c = c+cs[i*2+1];

					if (block_r < 0 or block_r >= block_row or block_c < 0 or block_c >= block_col)
					{
						//cout << i << endl;
						continue;
					}	
					else
					{
						//assign value	
						array_map[candidate][0] = candidate;					
						array_map[candidate][1] = *(pmm + (block_r*block_col*2 + block_c*2 + 1));
						array_map[candidate][2] = *(pmm + (block_r*block_col*2 + block_c*2 + 0));
						array_map[candidate][3] = 0;					
						
						candidate++;
					}

				}

				//cal the round block score (8 block)
				cal_block_score(array_map, candidate);	

				//get motion vector of the center block 
				int c_x = *(pmm + (r*block_col*2 + c*2 + 1));
				int c_y = *(pmm + (r*block_col*2 + c*2 + 0));

				//cal the distance of ed
				int distance = fabs(c_x-array_map[0][1])+fabs(c_y-array_map[0][2]);

				//judgment
				if(distance > 10)
				{
					*(pmm + (r*block_col*2 + c*2 + 1)) = array_map[0][1];
					*(pmm + (r*block_col*2 + c*2 + 0)) = array_map[0][2];
				}
				//cout << distance <<endl;

				delete [] array_map;

			}
			else
			{
				//round 8 block --> 
				int cs[16] = {-1,-1,-1,0,-1,1,0,-1,0,1,1,-1,1,0,1,1};
				int candidate = 0;

				//init 								
				int **array_map = new int*[8]; //[i][x][y][socre]
				for (int i=0;i<8;i++)
				{
					array_map[i] = new int[4];
				}
		
				//3*3 filter and assign value	
				for (int i=0; i<8;i++)
				{
					int block_r = r+cs[i*2];
					int block_c = c+cs[i*2+1];
					
					//assign value	
					array_map[candidate][0] = candidate;					
					array_map[candidate][1] = *(pmm + (block_r*block_col*2 + block_c*2 + 1));
					array_map[candidate][2] = *(pmm + (block_r*block_col*2 + block_c*2 + 0));
					array_map[candidate][3] = 0;					
					
					candidate++;
				}

				//cal the round block score (8 block)
				cal_block_score(array_map, candidate);	

				//get motion vector of the center block 
				int c_x = *(pmm + (r*block_col*2 + c*2 + 1));
				int c_y = *(pmm + (r*block_col*2 + c*2 + 0));

				//cal the distance of ed
				int distance = fabs(c_x-array_map[0][1])+fabs(c_y-array_map[0][2]);

				//judgment
				if(distance > DISTANCE_THEROSHOLD)
				{
					*(pmm + (r*block_col*2 + c*2 + 1)) = array_map[0][1];
					*(pmm + (r*block_col*2 + c*2 + 0)) = array_map[0][2];
				}
				//cout << distance <<endl;

				delete [] array_map;

			}
		}
	}
}






void tdrs_thread(uchar *dsrc, uchar *ddist, int *plmm, int *pmm, int src_rows, int src_cols, int cur_thread, int num_thread)
{
    int block_size(BLOCKSIZE);
    int searching_area(block_size*block_size), radius(block_size/2);
    int block_row(src_rows/block_size), block_col(src_cols/block_size);

 	//cout << "thread num is " << cur_thread << " working" << endl;

    for (int r = cur_thread; r < block_row; r += num_thread)
    {
        for (int c = 0; c < block_col; c++)
        {
            int center[2] = {c*block_size + radius,  r*block_size + radius};

            //boundary, use one-at-a-time
            if (r == 0 or c == 0 or c == block_col - 1)
            {
				//printf("cur_t = %d, r=%d\n",cur_thread,r);
                //horizontal search
                int updates_h[6] = {0,-1,0,0,0,1};      //vector<Point> updates_h {Point(-1,0), Point(0,0), Point(1,0)};
                int block_center[2] = {0};
                int init_block_center[2] = {center[0], center[1]};
                oaat(dsrc, ddist, block_center, center, init_block_center, searching_area, updates_h, src_rows, src_cols);
                //vertical search
                int updates_v[6] = {-1,0,0,0,1,0};     //vector<Point> updates_v {Point(0,-1), Point(0,0), Point(0,1)};
                int from_point[2] = {0};
                oaat(dsrc, ddist, from_point, center, block_center, searching_area, updates_v, src_rows, src_cols);

                *(pmm + (r*block_col*2 + c*2 + 0)) = (center[0] - from_point[0]);  //what I trying to do : motion_map[r][c] = (center - from_point);
                *(pmm + (r*block_col*2 + c*2 + 1)) = (center[1] - from_point[1]);
            }
            // 3d recursive searching
            else
            {
                //magic number from paper
                int p(8);
                int updates[16] = {0,4, 0,-4, 2,0, -2,0, -1,1 ,1,-1, 2,5, -2,-5};

                //initial estimation is early cauculated value
				//fix big bug!!!
                int Da_current[2] = { *(pmm+((r-1)*block_col*2 + (c-1)*2 + 0)), *(pmm + ( (r-1)*block_col*2 + (c-1)*2 + 1))}; //motion_map[r-1][c-1]	//Sa
                int Db_current[2] = { *(pmm+((r-1)*block_col*2 + (c+1)*2 + 0)), *(pmm + ( (r-1)*block_col*2 + (c+1)*2 + 1))}; //motion_map[r-1][c+1]	//Sb

                //inital CAs
                int Da_previous[2] = {0,0};
                int Db_previous[2] = {0,0};

                //if there is CAs
                if (c > 1 and r < block_row -1 and c < block_row -1)
                {
                    Da_previous[0] =  *(plmm + ((r+1)*block_col*2 + (c+1)*2 + 0)); //last_motion[r+2][c+2]	//Ta
                    Da_previous[1] =  *(plmm + ((r+1)*block_col*2 + (c+1)*2 + 1));
                    Db_previous[0] =  *(plmm + ((r+1)*block_col*2 + (c-1)*2 + 0)); //last_motion[r+2][c-2]	//Tb
                    Db_previous[1] =  *(plmm + ((r+1)*block_col*2 + (c-1)*2 + 1));
                }

                int block_cnt_a(r*c), block_cnt_b(r*c+2);
				//printf("block_cnt = %d,%d\n",block_cnt_a,block_cnt_b);
                float SAD_a(100000.), SAD_b(100000.);
                int not_update_a(0), not_update_b(0);

/*
				int candidate[26] = {0};
				float candidate_sad[13] = {0};
				bool candidate_index[13] = {0};

				//inital 13 candidate set
                int cs[26] = {Da_current[0], Da_current[1],
								Db_current[0], Db_current[1],
                                Da_current[0] + updates[0],  Da_current[1] + updates[1],
                                Da_current[0] + updates[2],  Da_current[1] + updates[3],
                                Da_current[0] + updates[4],  Da_current[1] + updates[5],
                                Da_current[0] + updates[6],  Da_current[1] + updates[7],
                                Da_current[0] + updates[8],  Da_current[1] + updates[9],
                                Da_current[0] + updates[10], Da_current[1] + updates[11],
                                Da_current[0] + updates[12], Da_current[1] + updates[13],
                                Da_current[0] + updates[14], Da_current[1] + updates[15],
                                Da_previous[0], Da_previous[1],
								Db_previous[0], Db_previous[1],
                                0,0};

				//get SAD from each candidate, there are 13 candidate each time
                for (int index = 0; index < 13; index++)
                {
                    bool out_of_boundary(false);
                    int eval_center[2] = {center[0] - cs[index*2],  center[1] - cs[index*2 + 1]}; //???? - or +
                    if (eval_center[0] < radius or eval_center[1] < radius or eval_center[0] > src_cols - radius or eval_center[1] > src_rows - radius)
                        out_of_boundary = true;
                    if (out_of_boundary)
                        continue;

                    if (not out_of_boundary)
                    {
                        candidate[index*2] = cs[index*2];
                        candidate[index*2 + 1] = cs[index*2 + 1];
                        candidate_sad[index] = get_sad(dsrc, ddist, center[0], center[1], eval_center[0], eval_center[1] , src_cols);
                        candidate_index[index] = 1;
                    }

                }

				//motion vector(0,0) been penalty
				if(candidate_sad[12] <= 45)
				{
					candidate_sad[12] = candidate_sad[12] * 0.9;
				}else
				{
					candidate_sad[12] = candidate_sad[12] * 1.1;
				}

				//get the min SAD
				float min_sad(100000.);
				int tmp_update[2] = {0};
				for (int i = 0; i < 13; i++)
				{
		            if (!candidate_index[i])
		                continue;

                    if (candidate_sad[i] <= min_sad )
                    {
                        min_sad = candidate_sad[i];
                        tmp_update[0] = candidate[i*2];
                        tmp_update[1] = candidate[i*2+1];
                    }	
					
				}
				//cout << tmp_update[0] << " " << tmp_update[1]<< endl;
				//cout << endl;

				//assert value
		        *(pmm + (r*block_col*2 + c*2)) = tmp_update[0];
		        *(pmm + (r*block_col*2 + c*2 + 1)) = tmp_update[1]; 
*/				

                while (true)
                {
                    int candidate_a[10] = {0};        
                    float candidate_sad_a[5] = {0}; 
                    bool candidate_index_a[5] = {0};

                    int update_a[2] = {updates[(block_cnt_a %p)*2], updates[(block_cnt_a %p)*2 + 1]};
                    block_cnt_a++;

                    //inital candidate set, 1st : ACS, 2nd : CAs, 3th : 0
                    int cs_a[10] = {Da_current[0], Da_current[1],
								   Db_current[0], Db_current[1],
                                   Da_current[0] + update_a[0], Da_current[1] + update_a[1],
                                   Da_previous[0], Da_previous[1],
                                   0,0};

                    //get SAD from each candidate, there are 4 candidate each time
                    for (int index = 0; index < 5; index++)
                    {
                        bool out_of_boundary_a(false);
						//both ward
                        int eval_center[2] = {center[0] - cs_a[index*2],  center[1] - cs_a[index*2+1]};
						

                        if (eval_center[0] < radius or eval_center[1] < radius or eval_center[0] > src_cols - radius or eval_center[1] > src_rows - radius)
                            out_of_boundary_a = true;
                        if (out_of_boundary_a)
                            continue;

                        if (not out_of_boundary_a)
                        {
                            candidate_a[index*2] = cs_a[index*2];
                            candidate_a[index*2 + 1] = cs_a[index*2 + 1];
                            candidate_sad_a[index] = get_sad(dsrc, ddist, center[0], center[1], eval_center[0], eval_center[1] , src_cols);
							//cout << candidate_sad_a[index] << "	";
                            candidate_index_a[index] = 1;
                        }

                    }
					//cout << endl;
                    //compute penalty from each candidate
                    float min_sad_a(100000.);
                    int tmp_update_a[2] = {0};

                    float max_sad_a = get_max(candidate_sad_a, 5);

                    //compute estimator a
                    for (int i = 0; i < 5; i++)
                    {
                        if (!candidate_index_a[i])
                            continue;
                        float current_sad = candidate_sad_a[i];
                        float penalty(0);
                        switch (i)
                        {
                            case 0:
                                penalty = 0.;
                                break;
                            case 1:
                                penalty = 0.;
                                break;
                            case 2:
                                penalty = 0.004 * max_sad_a * norm(update_a);
                                break;
                            case 3:
                                penalty = 0.008 * max_sad_a;
                                break;
                            case 4:
                                penalty = 0.016 * max_sad_a;
                                break;
                        }

                        current_sad += penalty;
                        if (current_sad < min_sad_a)
                        {
                            min_sad_a = current_sad;
                            tmp_update_a[0] = candidate_a[i*2];
                            tmp_update_a[1] = candidate_a[i*2+1];
                        }
                        //prefer 0 in the case of same SAD
                        else if (min_sad_a + 40. > current_sad and candidate_a[i*2] == 0 and candidate_a[i*2+1] == 0)
                        {
                            tmp_update_a[0] = 0;
                            tmp_update_a[1] = 0;
                        }
                    }

                    //update, if not, counter + 1
                    if (min_sad_a < SAD_a)
                    {
                        SAD_a = min_sad_a;
                        Da_current[0] = tmp_update_a[0];
                        Da_current[1] = tmp_update_a[1];
                        not_update_a = 0;
                    }
                    else
                        not_update_a += 1;

                    //break if any estiminator converge
                    int check_converge = 8;
                    if (not_update_a > check_converge)
                    {
                        break;
                    }
                    //break if out of searching area
                    if (abs(Da_current[0]) > searching_area or abs(Da_current[1]) > searching_area )
                    {
                        break;
                    }
                }


                *(pmm + (r*block_col*2 + c*2)) = Da_current[0];
                *(pmm + (r*block_col*2 + c*2 + 1)) = Da_current[1]; //motion_map[r][c] = Da_current

				//if( Da_current[0] > 10 or  Da_current[1] >10)
					//cout << Da_current[0] << "	"<< Da_current[1] << endl;
            }

        }
    }
}



void draw_arrow(Mat &src, Mat &dist, Mat &draw, int *last_motion)
{
    int block_size(BLOCKSIZE), radius(block_size/2);
    int row(src.rows), col(src.cols);
    int block_row(row/block_size), block_col(col/block_size);

	//draw the arrowed line
    for (int r = 0; r < block_row; r++)
    {
        for (int c = 0; c < block_col; c++)
        {
            if ( *(last_motion + r*block_col*2 + c*2) != 0 or *(last_motion + r*block_col*2 + c*2 + 1) != 0)
            {
                Point motion(*(last_motion + r*block_col*2 + c*2) , *(last_motion + r*block_col*2 + c*2+1 ));
                Point center(c*block_size + radius, r*block_size + radius);
                Point from = center - motion;
                arrowedLine(draw, from, center, Scalar(0,255,0));
            }
        }
    }

}


//imread generate a continous matrix
/*
void tdrs(Mat &src, Mat &dist, Mat &draw, int *last_motion)
{
    int block_size(BLOCKSIZE), radius(block_size/2);
    int row(src.rows), col(src.cols);
    int block_row(row/block_size), block_col(col/block_size);

    int img_size = row * col;
    uchar *dsrc, *ddist;
    dsrc = new uchar[img_size];
    memcpy(dsrc, src.data, img_size*sizeof(uchar));
    ddist = new uchar[img_size];
    memcpy(ddist, dist.data, img_size*sizeof(uchar));


    //initial zero motion map
    int size = block_row * block_col * 2;
    int *motion_map = new int[size]();

    int *pmm, *plmm;
    pmm = new int[size];
    memcpy(pmm, motion_map, size*sizeof(int));
    plmm = new int[size];
    memcpy(plmm, last_motion, size*sizeof(int));

    int total_thread = 1;
    thread ts[total_thread];
    for (int i = 0; i < total_thread; i++)
        ts[i] = thread(tdrs_thread, dsrc, ddist, plmm, pmm, src.rows, src.cols, i, total_thread);
    for (int i = 0; i < total_thread; i++)
        ts[i].join();

    //update last_motion for next frame
    memcpy(last_motion, pmm, size*sizeof(int));

    //free memory
    delete [] pmm; delete [] plmm; delete [] dsrc; delete [] ddist; delete [] motion_map;

	//draw the arrowed line
    for (int r = 0; r < block_row; r++)
    {
        for (int c = 0; c < block_col; c++)
        {
            if ( *(last_motion + r*block_col*2 + c*2) != 0 or *(last_motion + r*block_col*2 + c*2 + 1) != 0)
            {
                Point motion(*(last_motion + r*block_col*2 + c*2) , *(last_motion + r*block_col*2 + c*2 + 1));
				//printf("%d,%d\n",motion.x,motion.y);
                Point center(c*block_size + radius, r*block_size + radius);
				//printf("%d,%d\n",center.x,center.y);
                Point from = center - motion;
                arrowedLine(draw, from, center, Scalar(0,255,0));
            }
        }
    }
	
} 

*/

//imread generate a continous matrix
void tdrs_MC(Mat &src, Mat &dist, Mat &draw, Mat &src_image, Mat &IF_image,int *last_motion)
{
/*
    int block_size(BLOCKSIZE), radius(block_size/2);
    int row(src.rows), col(src.cols);
    int block_row(row/block_size), block_col(col/block_size);
    int img_size = row * col;

    uchar *dsrc, *ddist;
    dsrc = new uchar[img_size];
    memcpy(dsrc, src.data, img_size*sizeof(uchar));
    ddist = new uchar[img_size];
    memcpy(ddist, dist.data, img_size*sizeof(uchar));


    //initial zero motion map
    int size = block_row * block_col * 2;
    int *motion_map = new int[size]();

    int *pmm, *plmm;
    pmm = new int[size];
    memcpy(pmm, motion_map, size*sizeof(int));
    plmm = new int[size];
    memcpy(plmm, last_motion, size*sizeof(int));

    int total_thread = 1;
    thread ts[total_thread];
    for (int i = 0; i < total_thread; i++)
        ts[i] = thread(tdrs_thread, dsrc, ddist, plmm, pmm, src.rows, src.cols, i, total_thread);
    for (int i = 0; i < total_thread; i++)
        ts[i].join();

	//motion vector excluding outliers
	MV_without_abnormal(pmm,row,col);

    //update last_motion for next frame
    memcpy(last_motion, pmm, size*sizeof(int));

    //free memory
    delete [] pmm; delete [] plmm; delete [] dsrc; delete [] ddist; delete [] motion_map;

	//draw the arrowed line

    for (int r = 0; r < block_row; r++)
    {
        for (int c = 0; c < block_col; c++)
        {
            if ( *(last_motion + r*block_col*2 + c*2) != 0 or *(last_motion + r*block_col*2 + c*2 + 1) != 0)
            {
                Point motion(*(last_motion + r*block_col*2 + c*2) , *(last_motion + r*block_col*2 + c*2 + 1));
				//printf("%d,%d\n",motion.x,motion.y);
                Point center(c*block_size + radius, r*block_size + radius);
				//printf("%d,%d\n",center.x,center.y);
                Point from = center - motion;
                arrowedLine(draw, from, center, Scalar(0,255,0));
            }
        }
    }
	
	
	//Forward MC
	for (int r = 0; r< block_row; r++)
	{
		for (int c = 0; c< block_col;c++)
		{
			if (*(last_motion + r*block_col*2 + c*2) != 0 or *(last_motion + r*block_col*2 + c*2 + 1) != 0)	
			{
				//fix this bug!!!! last_motion --> (y,x)
				Point motion(*(last_motion + r*block_col*2 + c*2+1) , *(last_motion + r*block_col*2 + c*2));
				motion.x = int(motion.x*0.5);
				motion.y = int(motion.y*0.5);
				//cout << "motion.x/y is " << motion << endl;
				for (int a = 0; a< block_size; a++)
				{
					for (int b = 0; b < block_size; b++)
					{
						Point loc_dist(r*block_size+a+motion.x,c*block_size+b+motion.y);
						if ((loc_dist.x <0) or (loc_dist.x > row-1) or (loc_dist.y <0 ) or (loc_dist.y > col-1))
						{
							//cout << "out of the edge, x is " << loc_dist.x << "and y is " << loc_dist.y << endl;
							continue;
						}
						else
						{
							Vec3b src_pix = src_image.at<Vec3b>(r*block_size+a,c*block_size+b);
							//cout << src_pix << endl;
							IF_image.at<Vec3b>(r*block_size+a+motion.x,c*block_size+b+motion.y) = src_pix;
							//cout << IF_image.at<Vec3b>(block_row*block_size+a+motion.x,block_col*block_size+b+motion.y) << endl;
						}
 
					}
				}
			}	
		}
	}
	

	Mat next_img = IF_image.clone();
	//Both ward
	for (int r = 0; r< block_row; r++)
	{
		for (int c = 0; c< block_col;c++)
		{
			if (*(last_motion + r*block_col*2 + c*2) != 0 or *(last_motion + r*block_col*2 + c*2 + 1) != 0)	
			{
				//fix this bug!!!! last_motion --> (y,x)
				Point motion(*(last_motion + r*block_col*2 + c*2+1) , *(last_motion + r*block_col*2 + c*2));
				motion.x = int(motion.x*0.5);
				motion.y = int(motion.y*0.5);
				//cout << "motion.x/y is " << motion << endl;
				for (int a = 0; a< block_size; a++)
				{
					for (int b = 0; b < block_size; b++)
					{
						Point loc_dist_prev(r*block_size+a-motion.x,c*block_size+b-motion.y);
						Point loc_dist_next(r*block_size+a+motion.x,c*block_size+b+motion.y);
						if ((loc_dist_prev.x <0) or (loc_dist_prev.x > row-1) or (loc_dist_prev.y <0 ) or (loc_dist_prev.y > col-1) or (loc_dist_next.x <0) or (loc_dist_next.x > row-1) or (loc_dist_next.y <0 ) or (loc_dist_next.y > col-1))
						{
							//cout << loc_dist_prev.x << " " << loc_dist_prev.y << " ; " << loc_dist_next.x << " " << loc_dist_next.y << " ; " << endl;
							continue;
						}
						else
						{
							Vec3b src_pix = src_image.at<Vec3b>(r*block_size+a-motion.x,c*block_size+b-motion.y);
							Vec3b dist_pix = next_img.at<Vec3b>(r*block_size+a+motion.x,c*block_size+b+motion.y);
							for (int i = 0;i <3 ;i++)
							{
								IF_image.at<Vec3b>(r*block_size+a,c*block_size+b)[i] = int(0.5*(src_pix[i]+dist_pix[i]));			
							}
							//cout << src_pix << endl;
							//cout << IF_image.at<Vec3b>(block_row*block_size+a+motion.x,block_col*block_size+b+motion.y) << endl;
						}
 
					}
				}
			}	
		}
	}
*/
} 




//imread generate a continous matrix
void general_IF_MC_Forward(Mat &src_image, Mat &IF_image, Mat &dist_image, int *last_motion)
{
    int block_size(BLOCKSIZE), radius(block_size/2);
    int row(src_image.rows), col(src_image.cols);
    int block_row(row/block_size), block_col(col/block_size);

	//Forward MC
	for (int r = 0; r< block_row; r++)
	{
		for (int c = 0; c< block_col;c++)
		{
			if (*(last_motion + r*block_col*2 + c*2) != 0 or *(last_motion + r*block_col*2 + c*2 + 1) != 0)	
			{
				//fix this bug!!!! last_motion --> (y,x)
				Point motion(*(last_motion + r*block_col*2 + c*2+1) , *(last_motion + r*block_col*2 + c*2));
				motion.x = int(motion.x*0.5);
				motion.y = int(motion.y*0.5);
				//cout << "motion.x/y is " << motion << endl;
				for (int a = 0; a< block_size; a++)
				{
					for (int b = 0; b < block_size; b++)
					{
						Point loc_dist(r*block_size+a+motion.x,c*block_size+b+motion.y);
						if ((loc_dist.x <0) or (loc_dist.x > row-1) or (loc_dist.y <0 ) or (loc_dist.y > col-1))
						{
							//cout << "out of the edge, x is " << loc_dist.x << "and y is " << loc_dist.y << endl;
							continue;
						}
						else
						{
							Vec3b src_pix = src_image.at<Vec3b>(r*block_size+a,c*block_size+b);
							//cout << src_pix << endl;
							IF_image.at<Vec3b>(r*block_size+a+motion.x,c*block_size+b+motion.y) = src_pix;
							//cout << IF_image.at<Vec3b>(block_row*block_size+a+motion.x,block_col*block_size+b+motion.y) << endl;
						}
 
					}
				}
			}	
		}
	}
	
 

}





//imread generate a continous matrix
void general_IF_MC(Mat &src_image, Mat &IF_image, Mat &dist_image, int *last_motion, int *flag_map)
{
    int block_size(BLOCKSIZE), radius(block_size/2);
    int row(src_image.rows), col(src_image.cols);
    int block_row(row/block_size), block_col(col/block_size);
	int num=0;

	//Both ward
	for (int r = 0; r< block_row; r++)
	{
		for (int c = 0; c< block_col;c++)
		{
			if (*(last_motion + r*block_col*2 + c*2) != 0 or *(last_motion + r*block_col*2 + c*2 + 1) != 0)	
			{
				//fix this bug!!!! last_motion --> (y,x)
				Point motion(*(last_motion + r*block_col*2 + c*2+1) , *(last_motion + r*block_col*2 + c*2));
				
				if(*(flag_map + r*block_col*2 + c*2+0) == 1 and *(flag_map + r*block_col*2 + c*2+1) == 1)
				{
					motion.x = 0;
					motion.y = 0;
					//
					num += 1;
				}
				else
				{
					motion.x = int(motion.x*0.5);
					motion.y = int(motion.y*0.5);
				}

				//if(motion.x != 0 and motion.y != 0)
					//cout << "motion.x/y is " << motion.x << "	" << motion.y << endl;

				for (int a = 0; a< block_size; a++)
				{
					for (int b = 0; b < block_size; b++)
					{
						Point loc_dist_prev(r*block_size+a-motion.x,c*block_size+b-motion.y);
						Point loc_dist_next(r*block_size+a+motion.x,c*block_size+b+motion.y);
						if ((loc_dist_prev.x <0) or (loc_dist_prev.x > row-1) or (loc_dist_prev.y <0 ) or (loc_dist_prev.y > col-1) or (loc_dist_next.x <0) or (loc_dist_next.x > row-1) or (loc_dist_next.y <0 ) or (loc_dist_next.y > col-1))
						{
							Vec3b src_pix = src_image.at<Vec3b>(r*block_size+a,c*block_size+b);
							Vec3b dist_pix = dist_image.at<Vec3b>(r*block_size+a,c*block_size+b);
							for (int i = 0;i <3 ;i++)
							{
								IF_image.at<Vec3b>(r*block_size+a,c*block_size+b)[i] = int(0.5*(src_pix[i]+dist_pix[i]));			
							}
						}
						else
						{
							Vec3b src_pix = src_image.at<Vec3b>(r*block_size+a-motion.x,c*block_size+b-motion.y);
							Vec3b dist_pix = dist_image.at<Vec3b>(r*block_size+a+motion.x,c*block_size+b+motion.y);
							for (int i = 0;i <3 ;i++)
							{
								IF_image.at<Vec3b>(r*block_size+a,c*block_size+b)[i] = int(0.5*(src_pix[i]+dist_pix[i]));			
							}
						}
 
					}
				}
			}	
		}
	}

	//debug 
	cout << num << endl;
}



//Select the better SAD
void select_sad(uchar *src, uchar *dist, int *current_motion_forward, int *current_motion_backward, int *sad_motion_map, int *sad_flag_map,int src_row, int src_col)
{
    int block_size(BLOCKSIZE), radius(block_size/2);
    int row(src_row), col(src_col);
    int block_row(row/block_size), block_col(col/block_size);


	for (int r = 0; r< block_row; r++)
	{
		for (int c = 0; c< block_col;c++)
		{
			int val_forward[2] = {int(*(current_motion_forward + r*block_col*2 + c*2)),int(*(current_motion_forward + r*block_col*2 + c*2+1))};
			int val_backward[2] = {int(*(current_motion_backward + r*block_col*2 + c*2)),int(*(current_motion_backward + r*block_col*2 + c*2+1))};

			//the vector of the both direction motion is equal
			if ((fabs(val_forward[0]) == fabs(val_backward[0])) and (fabs(val_forward[1]) == fabs(val_backward[1])))	
			{
				*(sad_motion_map + r*block_col*2 + c*2) = val_forward[0];
				*(sad_motion_map + r*block_col*2 + c*2+1) = val_forward[1];

			}
			else
			{
				//cout << val_forward[0] << "	" << val_forward[1] << "	";
				//cout << val_backward[0] << "	" << val_backward[1] << endl;
				int forward_sad(0), backward_sad(0);

				{

					//forward SAD
					int src_center[2] = {r*block_size+radius-int(val_forward[1]*0.5), c*block_size+radius-int(val_forward[0]*0.5)};
					int dist_center[2] = {r*block_size+radius+int(val_forward[1]*0.5), c*block_size+radius+int(val_forward[0]*0.5)};

                    bool out_of_boundary(false);
                    if (src_center[0] < radius or src_center[1] < radius or src_center[0] > row - radius or src_center[1] > col - radius)
                        out_of_boundary = true;
                    if (dist_center[0] < radius or dist_center[1] < radius or dist_center[0] > row - radius or dist_center[1] > col - radius)
                        out_of_boundary = true;

					if (out_of_boundary == false)
					{
						forward_sad = get_sad(src, dist, src_center[1], src_center[0], dist_center[1], dist_center[0],src_col);
					}
					else
					{
						//cout << "forward out_of_boundary" << endl;
						forward_sad = ERROR_VALUE;
					}
					
					
				}

				{
					//backward SAD
					int src_center[2] = {r*block_size+radius-int(val_backward[1]*0.5), c*block_size+radius-int(val_backward[0]*0.5)};
					int dist_center[2] = {r*block_size+radius+int(val_backward[1]*0.5), c*block_size+radius+int(val_backward[0]*0.5)};

                    bool out_of_boundary(false);
                    if (src_center[0] < radius or src_center[1] < radius or src_center[0] > row - radius or src_center[1] > col - radius)
                        out_of_boundary = true;
                    if (dist_center[0] < radius or dist_center[1] < radius or dist_center[0] > row - radius or dist_center[1] > col - radius)
                        out_of_boundary = true;

					if (out_of_boundary == false)
					{
						backward_sad = get_sad(dist, src, src_center[1], src_center[0], dist_center[1], dist_center[0],src_col);
					}
					else
					{
						//cout << "backward out_of_boundary" << endl;
						backward_sad = ERROR_VALUE;
					}
				}

				if(val_backward[1] !=0 and val_forward[1] != 0)
				//cout << forward_sad << "	" << backward_sad << ";	"<<val_backward[1] <<	"	" << val_forward[1] << "	"<< val_backward[0] <<	"	" << val_forward[0]<<endl;

				if((forward_sad <= backward_sad) )
				{
					*(sad_motion_map + r*block_col*2 + c*2) = val_forward[0];
					*(sad_motion_map + r*block_col*2 + c*2+1) = val_forward[1];

					if(forward_sad > MV_THRESHOLD)
					{
						*(sad_flag_map + r*block_col*2 + c*2) = 1;
						*(sad_flag_map + r*block_col*2 + c*2+1) = 1;
					}
					else
					{
						*(sad_flag_map + r*block_col*2 + c*2) = 0;
						*(sad_flag_map + r*block_col*2 + c*2+1) = 0;
					}
				}
				else if((backward_sad <= forward_sad) )
				{
					*(sad_motion_map + r*block_col*2 + c*2) = -val_backward[0];
					*(sad_motion_map + r*block_col*2 + c*2+1) = -val_backward[1];

					if(backward_sad > MV_THRESHOLD)	
					{
						*(sad_flag_map + r*block_col*2 + c*2) = 1;
						*(sad_flag_map + r*block_col*2 + c*2+1) = 1;
					}
					else
					{
						*(sad_flag_map + r*block_col*2 + c*2) = 0;
						*(sad_flag_map + r*block_col*2 + c*2+1) = 0;
					}
				}

			}

			//cout << *(sad_motion_map + r*block_col*2 + c*2) << "	" << *(sad_motion_map + r*block_col*2 + c*2+1) <<endl;	
		}
	}
}


void post_processConcer_Line(uchar *dsrc, uchar *ddist, int *motion_Mat ,int *motion_candidate, int src_rows, int src_cols, int current_rows, int current_cols)
{
    int block_size(BLOCKSIZE);
    int block_col(src_cols/block_size);
	int row(src_rows),col(src_cols);
	int radius(block_size/2);

	int *cs = new int[6];
	memcpy(cs,motion_candidate, 6*sizeof(int));

	int candidate[6] = {0};
	int candidate_mv_a[2] = {0};
	float candidate_sad[3] = {0};
	float min_sad(9999999.0);

	for (int index = 0; index <3; index++)
	{
        candidate[index*2] = cs[index*2];
        candidate[index*2 + 1] = cs[index*2 + 1];

		int src_center[2] = {current_rows*block_size+radius-int(candidate[1]*0.5), current_cols*block_size+radius-int(candidate[0]*0.5)};
		int dist_center[2] = {current_rows*block_size+radius+int(candidate[1]*0.5), current_cols*block_size+radius+int(candidate[0]*0.5)};

        bool out_of_boundary(false);
        if (src_center[0] < radius or src_center[1] < radius or src_center[0] > row - radius or src_center[1] > col - radius)
            out_of_boundary = true;
        if (dist_center[0] < radius or dist_center[1] < radius or dist_center[0] > row - radius or dist_center[1] > col - radius)
            out_of_boundary = true;

		if (out_of_boundary == false)
		{
			candidate_sad[index] = get_sad(dsrc, ddist, src_center[1], src_center[0], dist_center[1], dist_center[0] , src_cols);

			//get min SAD
			if (candidate_sad[index] < min_sad)
			{
				//(y,x)
				candidate_mv_a[0] = candidate[index*2];
				candidate_mv_a[1] = candidate[index*2+1];
			}
		}
		else
		{
			//cout << "out boundary " <<endl;			
			continue;
		}

        

	}

	//assert value
	*(motion_Mat + current_rows*block_col*2+current_cols*2+0) = candidate_mv_a[0];
	*(motion_Mat + current_rows*block_col*2+current_cols*2+1) = candidate_mv_a[1];	

	delete [] cs;
}


void post_processBlock(uchar *dsrc, uchar *ddist, int *motion_Mat ,int src_rows ,int src_cols , int current_rows, int current_cols)
{
    int block_size(BLOCKSIZE);
    int block_col(src_cols/block_size);
	int row(src_rows),col(src_cols);
	int radius(block_size/2);

	int result[8] = {0};
	int candidate_num = 0;

	//the neiborhood of 8 block is same
	{
		int nb_y[9] = { *(motion_Mat + (current_rows-1)*block_col*2+(current_cols-1)*2+0),
					*(motion_Mat + (current_rows-1)*block_col*2+(current_cols+0)*2+0),
					*(motion_Mat + (current_rows-1)*block_col*2+(current_cols+1)*2+0),
					*(motion_Mat + (current_rows+0)*block_col*2+(current_cols-1)*2+0),
					*(motion_Mat + (current_rows+0)*block_col*2+(current_cols+0)*2+0),
					*(motion_Mat + (current_rows+0)*block_col*2+(current_cols+1)*2+0),
					*(motion_Mat + (current_rows+1)*block_col*2+(current_cols-1)*2+0),
					*(motion_Mat + (current_rows+1)*block_col*2+(current_cols+0)*2+0),
					*(motion_Mat + (current_rows+1)*block_col*2+(current_cols+1)*2+0)
					};

		int nb_x[9] = { *(motion_Mat + (current_rows-1)*block_col*2+(current_cols-1)*2+1),
					*(motion_Mat + (current_rows-1)*block_col*2+(current_cols+0)*2+1),
					*(motion_Mat + (current_rows-1)*block_col*2+(current_cols+1)*2+1),
					*(motion_Mat + (current_rows+0)*block_col*2+(current_cols-1)*2+1),
					*(motion_Mat + (current_rows+0)*block_col*2+(current_cols+0)*2+1),
					*(motion_Mat + (current_rows+0)*block_col*2+(current_cols+1)*2+1),
					*(motion_Mat + (current_rows+1)*block_col*2+(current_cols-1)*2+1),
					*(motion_Mat + (current_rows+1)*block_col*2+(current_cols+0)*2+1),
					*(motion_Mat + (current_rows+1)*block_col*2+(current_cols+1)*2+1)
					};

		if((nb_y[0] == nb_y[1] == nb_y[2] == nb_y[3] == nb_y[4] == nb_y[5] == nb_y[6] == nb_y[7] == nb_y[8]) and (nb_x[0] == nb_x[1] == nb_x[2] == nb_x[3] == nb_x[4] == nb_x[5] == nb_x[6] == nb_x[7] == nb_x[8]))
		{
			//cout << "8 block is same" << endl;
			return;
		}


	}

	for (int mode =0 ; mode<4; mode++)
	{
		if (mode == 0)
		{
			//get the neiborhood block
			int cs[6]= {*(motion_Mat + (current_rows-1)*block_col*2+current_cols*2+0),*(motion_Mat + (current_rows-1)*block_col*2+current_cols*2+1),
						*(motion_Mat + current_rows*block_col*2+(current_cols-1)*2+0),*(motion_Mat + current_rows*block_col*2+(current_cols-1)*2+1),
						*(motion_Mat + (current_rows-1)*block_col*2+(current_cols-1)*2+0),*(motion_Mat + (current_rows-1)*block_col*2+(current_cols-1)*2+1)
						};
					
			//Judging direction
			if(cs[0]*cs[2] >=0 and cs[1]*cs[3] >=0)
			{		
				//y
				result[candidate_num*2]   = (cs[0]+cs[2]+cs[4])/3;
				//x
				result[candidate_num*2+1] = (cs[1]+cs[3]+cs[5])/3;
				candidate_num += 1;	
			}
			
			continue;

		}else if (mode == 1)
		{
			//get the neiborhood block
			int cs[6]= {*(motion_Mat + (current_rows-1)*block_col*2+current_cols*2+0),*(motion_Mat + (current_rows-1)*block_col*2+current_cols*2+1),
						*(motion_Mat + current_rows*block_col*2+(current_cols+1)*2+0),*(motion_Mat + current_rows*block_col*2+(current_cols+1)*2+1),
						*(motion_Mat + (current_rows-1)*block_col*2+(current_cols+1)*2+0),*(motion_Mat + (current_rows-1)*block_col*2+(current_cols+1)*2+1)
						};

			//Judging direction
			if(cs[0]*cs[2] >=0 and cs[1]*cs[3] >=0)
			{		
				//y
				result[candidate_num*2]   = (cs[0]+cs[2]+cs[4])/3;
				//x
				result[candidate_num*2+1] = (cs[1]+cs[3]+cs[5])/3;
				candidate_num += 1;	
			}

			continue;
		}else if (mode == 2)
		{
			//get the neiborhood block
			int cs[6]= {*(motion_Mat + (current_rows+1)*block_col*2+current_cols*2+0),*(motion_Mat + (current_rows+1)*block_col*2+current_cols*2+1),
						*(motion_Mat + current_rows*block_col*2+(current_cols-1)*2+0),*(motion_Mat + current_rows*block_col*2+(current_cols-1)*2+1),
						*(motion_Mat + (current_rows+1)*block_col*2+(current_cols-1)*2+0),*(motion_Mat + (current_rows+1)*block_col*2+(current_cols-1)*2+1)
						};

			//Judging direction
			if(cs[0]*cs[2] >=0 and cs[1]*cs[3] >=0)
			{		
				//y
				result[candidate_num*2]   = (cs[0]+cs[2]+cs[4])/3;
				//x
				result[candidate_num*2+1] = (cs[1]+cs[3]+cs[5])/3;
				candidate_num += 1;	
			}

			continue;
		}else if (mode == 3)
		{
			//get the neiborhood block
			int cs[6]= {*(motion_Mat + (current_rows+1)*block_col*2+current_cols*2+0),*(motion_Mat + (current_rows+1)*block_col*2+current_cols*2+1),
						*(motion_Mat + current_rows*block_col*2+(current_cols+1)*2+0),*(motion_Mat + current_rows*block_col*2+(current_cols+1)*2+1),
						*(motion_Mat + (current_rows+1)*block_col*2+(current_cols+1)*2+0),*(motion_Mat + (current_rows+1)*block_col*2+(current_cols+1)*2+1)
						};

			//Judging direction
			if(cs[0]*cs[2] >=0 and cs[1]*cs[3] >=0)
			{		
				//y
				result[candidate_num*2]   = (cs[0]+cs[2]+cs[4])/3;
				//x
				result[candidate_num*2+1] = (cs[1]+cs[3]+cs[5])/3;
				candidate_num += 1;	
			}

			continue;
		}
	}	

	int candidate[8] = {0};
	int candidate_mv_a[2] = {0};
	float candidate_sad[4] = {0};
	float min_sad(9999999.0);

	for (int index = 0; index <candidate_num; index++)
	{
        candidate[index*2] = result[index*2];
        candidate[index*2 + 1] = result[index*2 + 1];

		int src_center[2] = {current_rows*block_size+radius-int(candidate[index*2 + 1]*0.5), current_cols*block_size+radius-int(candidate[index*2]*0.5)};
		int dist_center[2] = {current_rows*block_size+radius+int(candidate[index*2 + 1]*0.5), current_cols*block_size+radius+int(candidate[index*2]*0.5)};

        bool out_of_boundary(false);
        if (src_center[0] < radius or src_center[1] < radius or src_center[0] > row - radius or src_center[1] > col - radius)
            out_of_boundary = true;
        if (dist_center[0] < radius or dist_center[1] < radius or dist_center[0] > row - radius or dist_center[1] > col - radius)
            out_of_boundary = true;

		if (out_of_boundary == false)
		{
			candidate_sad[index] = get_sad(dsrc, ddist, src_center[1], src_center[0], dist_center[1], dist_center[0] , src_cols);
			//cout << candidate_sad[index] << "	";
			//cout << candidate[index*2] << "	" << candidate[index*2+1] <<endl;
		
			//get min SAD
			if (candidate_sad[index] < min_sad)
			{
				//(y,x)
				candidate_mv_a[0] = candidate[index*2];
				candidate_mv_a[1] = candidate[index*2+1];
				//cout << "result: " << candidate[index*2] << "	" << candidate[index*2+1] <<endl;
			}
		}
		else
		{
			//cout << "out boundary: " << current_rows << "	" << current_cols <<endl;			
			continue;
		}
	}

	//assert value
	*(motion_Mat + current_rows*block_col*2+current_cols*2+0) = candidate_mv_a[0];
	*(motion_Mat + current_rows*block_col*2+current_cols*2+1) = candidate_mv_a[1];	
	
	if(candidate_mv_a[0] !=0 or candidate_mv_a[1] != 0)
	{
		//cout << "result: " << candidate_mv_a[0] << "	" << candidate_mv_a[1] <<endl;
	}
		
}




void post_processME(uchar *dsrc, uchar *ddist, int *motion_Mat, int src_rows, int src_cols)
{
    int block_size(BLOCKSIZE);
    int block_row(src_rows/block_size), block_col(src_cols/block_size);
	int radius(block_size/2);

	for (int r = 0; r < block_row; r++)
	{
		for (int c = 0; c < block_col; c++)
		{
			//case1 and case2
			if (c == 0 or c == block_col-1 or r ==0 or r == block_row-1)
			{
				//case 1: four concers
				if(r==0 and c==0)
				{
					//(y,x)
					int current_motion[6] = {*(motion_Mat + r*block_col*2+c*2+0),*(motion_Mat + r*block_col*2+c*2+1),
											 *(motion_Mat + r*block_col*2+(c+1)*2+0),*(motion_Mat + r*block_col*2+(c+1)*2+1),
											 *(motion_Mat + (r+1)*block_col*2+c*2+0),*(motion_Mat + (r+1)*block_col*2+c*2+1)
											};				
					post_processConcer_Line(dsrc, ddist,  motion_Mat, current_motion,src_rows,src_cols,r,c);
				}else if(r==0 and c==block_col-1)
				{
					//(y,x)
					int current_motion[6] = {*(motion_Mat + r*block_col*2+c*2+0),*(motion_Mat + r*block_col*2+c*2+1),
											 *(motion_Mat + r*block_col*2+(c-1)*2+0),*(motion_Mat + r*block_col*2+(c-1)*2+1),
											 *(motion_Mat + (r+1)*block_col*2+c*2+0),*(motion_Mat + (r+1)*block_col*2+c*2+1)
											};	
					post_processConcer_Line(dsrc, ddist,  motion_Mat, current_motion,src_rows,src_cols,r,c);
				}else if(r==block_row-1 and c==0)
				{
					//(y,x)
					int current_motion[6] = {*(motion_Mat + r*block_col*2+c*2+0),*(motion_Mat + r*block_col*2+c*2+1),
											 *(motion_Mat + r*block_col*2+(c+1)*2+0),*(motion_Mat + r*block_col*2+(c+1)*2+1),
											 *(motion_Mat + (r-1)*block_col*2+c*2+0),*(motion_Mat + (r-1)*block_col*2+c*2+1)
											};	
					post_processConcer_Line(dsrc, ddist,  motion_Mat, current_motion,src_rows,src_cols,r,c);
				}else if(r==block_row-1 and c==block_col-1)
				{
					//(y,x)
					int current_motion[6] = {*(motion_Mat + r*block_col*2+c*2+0),*(motion_Mat + r*block_col*2+c*2+1),
											 *(motion_Mat + r*block_col*2+(c-1)*2+0),*(motion_Mat + r*block_col*2+(c-1)*2+1),
											 *(motion_Mat + (r-1)*block_col*2+c*2+0),*(motion_Mat + (r-1)*block_col*2+c*2+1)
											};	
					post_processConcer_Line(dsrc, ddist,  motion_Mat, current_motion,src_rows,src_cols,r,c);
				}
				else//case 2: four lines
				{
					//h:two lines 
					if(r == 0 or r == block_row-1)
					{
						int current_motion[6] = {*(motion_Mat + r*block_col*2+c*2+0),*(motion_Mat + r*block_col*2+c*2+1),
												 *(motion_Mat + r*block_col*2+(c-1)*2+0),*(motion_Mat + r*block_col*2+(c-1)*2+1),
												 *(motion_Mat + r*block_col*2+(c+1)*2+0),*(motion_Mat + r*block_col*2+(c+1)*2+1)
												};	
						post_processConcer_Line(dsrc, ddist,  motion_Mat, current_motion,src_rows,src_cols,r,c);
					}else if(c == 0 or c == block_col-1)	//v: two lines
					{
						int current_motion[6] = {*(motion_Mat + r*block_col*2+c*2+0),*(motion_Mat + r*block_col*2+c*2+1),
												 *(motion_Mat + (r+1)*block_col*2+c*2+0),*(motion_Mat + (r+1)*block_col*2+c*2+1),
												 *(motion_Mat + (r-1)*block_col*2+c*2+0),*(motion_Mat + (r-1)*block_col*2+c*2+1)
												};	
						post_processConcer_Line(dsrc, ddist,  motion_Mat, current_motion,src_rows,src_cols,r,c);
					}
				}
			}				
			else//normal block
			{
				//
				post_processBlock(dsrc, ddist,  motion_Mat, src_rows, src_cols, r, c);
			}

		
		}
	}
}



//imread generate a continous matrix
void tdrs_both(Mat &src, Mat &dist, Mat &src_image, Mat &IF_image, Mat &dist_image,int *last_motion_forward, int *last_motion_backward)
{
    int block_size(BLOCKSIZE), radius(block_size/2);
    int row(src.rows), col(src.cols);
    int block_row(row/block_size), block_col(col/block_size);

    int img_size = row * col;
    uchar *dsrc, *ddist;
    dsrc = new uchar[img_size];
    memcpy(dsrc, src.data, img_size*sizeof(uchar));
    ddist = new uchar[img_size];
    memcpy(ddist, dist.data, img_size*sizeof(uchar));

    //initial zero motion map
    int size = block_row * block_col * 2;
    int *motion_map_forward = new int[size]();
    int *motion_map_backward = new int[size]();

	//FORWARD:	pmm --> current ;		plmm --> last frame
    int *pmm_f, *plmm_f;
    pmm_f = new int[size];
    memcpy(pmm_f, motion_map_forward, size*sizeof(int));
    plmm_f = new int[size];
    memcpy(plmm_f, last_motion_forward, size*sizeof(int));

	//BACKWARD:	pmm --> current ;		plmm --> last frame
    int *pmm_b, *plmm_b;
    pmm_b = new int[size];
    memcpy(pmm_b, motion_map_backward, size*sizeof(int));
    plmm_b = new int[size];
    memcpy(plmm_b, last_motion_backward, size*sizeof(int));

	int *pmm_better_sad_f, *pmm_better_sad_b , *sad_flag_map;
	pmm_better_sad_f = new int[size];
	pmm_better_sad_b = new int[size];
	sad_flag_map = new int[size];

    int total_thread = CPU_THREAD;
    thread ts_f[total_thread];
	thread ts_b[total_thread];

	//forward
    for (int i = 0; i < total_thread; i++)
	{
        ts_f[i] = thread(tdrs_thread, dsrc, ddist, plmm_f, pmm_f, src.rows, src.cols, i, total_thread);
	}

	//backward
    for (int i = 0; i < total_thread; i++)
	{
        ts_b[i] = thread(tdrs_thread, ddist, dsrc, plmm_b, pmm_b, src.rows, src.cols, i, total_thread);
	}
	
	//thread.join
    for (int i = 0; i < total_thread; i++)
	{
        ts_f[i].join();
        ts_b[i].join();
	}

	//Post-process the ME
	post_processME(dsrc, ddist, pmm_f, row, col);
	post_processME(ddist, dsrc, pmm_b, row, col);


	//select the better SAD
	select_sad(dsrc, ddist, pmm_f, pmm_b, pmm_better_sad_f, sad_flag_map ,row, col);
	
	//mv mean
	//MV_without_abnormal(pmm_better_sad_f,row,col);

	//change the value from pmm_better_sad_f to pmm_better_sad_b
	//memcpy(pmm_better_sad_b, pmm_better_sad_f, size*sizeof(int));
	//for (int i = 0; i < size; i++)
	//{
	//	*(pmm_better_sad_b+i) = -*(pmm_better_sad_b+i);
		//cout << *(pmm_f+i) << "	";
	//}
	

    //update last_motion for next frame
    memcpy(last_motion_forward, pmm_f, size*sizeof(int));
    memcpy(last_motion_backward, pmm_b, size*sizeof(int));

	//IF_bothward
	general_IF_MC(src_image,IF_image,dist_image,pmm_better_sad_f, sad_flag_map);
	//general_IF_MC_Forward(src_image,IF_image,dist_image,last_motion_forward);
	//draw_arrow(src_image, dist_image, IF_image, last_motion_forward);

    //free memory
    delete [] pmm_f; delete [] plmm_f; 
    delete [] pmm_b; delete [] plmm_b;
	delete [] motion_map_forward; delete [] motion_map_backward;
	delete [] pmm_better_sad_f; delete [] pmm_better_sad_b; delete [] sad_flag_map;
	delete [] dsrc; delete [] ddist; 
	
	
} 



//point(x,y) means (col, row)
int main(int argc, char**argv)
{
	printf("start\n");
	//string file_name = "/home/iqiyi/Desktop/ESPCN_shijie/True-Motion-Estimation/Oigin_SDBronze.mp4";
	//string file_name = "/home/iqiyi/Desktop/ESPCN_shijie/True-Motion-Estimation/movie/output.mp4";
	string file_name = "/home/iqiyi/Desktop/ESPCN_shijie/True-Motion-Estimation/movie/FOREMAN_352x288_30_orig_01.avi";
	//string file_name = "/home/iqiyi/Desktop/ESPCN_shijie/True-Motion-Estimation/movie/CITY_704x576_60_orig_01.avi";

    int cnt(0), block_size(BLOCKSIZE);
    VideoCapture cap;
    long VideoTotalFrame;

    if (!cap.open(file_name))
	{
		cout << file_name << "open is fail" << endl;
		exit(1)	;
        return -1;	
	}
	else
	{
		cout << "load video " << file_name << endl;
		VideoTotalFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);
		cout << "the total frame of the Video is " << VideoTotalFrame << endl;
	}	

    Mat src, dist;
    cap >> src;
	// *2 --> to save y and x of the motion
    int *motion_map_forward = new int[src.cols/block_size * src.rows/block_size * 2]();
	int *motion_map_backward = new int[src.cols/block_size * src.rows/block_size * 2]();

    chrono::duration<float, milli> dtn;
    float avg_dtn;

    VideoWriter record("record.avi", CV_FOURCC('M','J','P','G'), cap.get(CV_CAP_PROP_FPS )*2, Size(src.cols, src.rows), true);

	
    while (1)
    {
        chrono::steady_clock::time_point start = chrono::steady_clock::now();
        Mat gsrc, gdist, out, IF_img;

        cap >> dist;
		//empth --> the last frame in the video
		if (dist.empty())
		{
			break;
		}
        out = dist.clone();

        cvtColor(src, gsrc, CV_BGR2GRAY);
        cvtColor(dist, gdist, CV_BGR2GRAY);

		IF_img = src.clone();
		//tdrs_MC(gsrc, gdist, out, src, IF_img, motion_map);
		//bothward: FORWARD and BACKWARD
		tdrs_both(gsrc, gdist, src, IF_img, out,motion_map_forward, motion_map_backward);

		//frame reflash to the next circle
        src = dist.clone();

        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        dtn = end - start;
        avg_dtn = (cnt/float(cnt+1))*avg_dtn + (dtn.count()/float(cnt+1));
        cnt++;
		cout << "cnt is " << cnt << endl;

        string tmp = boost::str(boost::format("%2.2fms / %2.2fms")% dtn.count()  %avg_dtn );
        //putText(out, tmp, cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,200,0), 1, CV_AA);
		putText(IF_img, tmp, cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,200,0), 1, CV_AA);

		//record the video
		record.write(IF_img);
        record.write(out);

		//save the image of the output
		//imwrite(boost::str(boost::format("video/%04d_a.jpg") %cnt).c_str(), IF_img);
		//imwrite(boost::str(boost::format("video/%04d_b.jpg") %cnt).c_str(), out);	

        imshow("motion", IF_img);
		//imshow("motion", out);
		//waitKey(0);

        if (char(waitKey(1)) == 'q')
            break;
    }
    delete [] motion_map_forward; delete [] motion_map_backward;

	cout << "the process has been completed" <<endl;
    return 0;
}
