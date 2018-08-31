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

int BLOCKSIZE = 16;
int CPU_THREAD = 1;

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
int get_sad(uchar *src, uchar *dist, int dx, int dy, int sx, int sy, int col)
{

    int block_size(BLOCKSIZE);
	//left top point of the block
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

	//center
    uchar *pCur, *pRef;
    pCur = (dist + dx*col + dy);
    pRef = (src + sx*col + sy);

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

	//cout << sad_test << endl;
    return sad_test;


}

inline float norm(int *point)
{
    return sqrt( pow(float(*(point) ), 2) + pow(float( *(point + 1) ),2) );
}


//point(x,y) means (row, col)
void oaat(uchar *src, uchar *dist, int *output, int *center, int *block_center, int searching_area, int *updates, int src_rows, int src_cols)
{
    int block_size(BLOCKSIZE), radius(block_size/2);
    output[0] = *(block_center);
    output[1] = *(block_center+1);
	//printf("%d,%d\n",output[0],output[1]);

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
            if (cs[i*2] < radius or cs[i*2+1] < radius or cs[i*2] > src_rows - radius or cs[i*2+1] > src_cols - radius)
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

		//printf("update = %d,%d\n",update[0],update[1]);
        *block_center = update[0];
        *(block_center + 1) = update[1];
    }
}


void tdrs_thread(uchar *dsrc, uchar *ddist, int *plmm, int *pmm, int src_rows, int src_cols, int cur_thread, int num_thread, int direction)
{
    int block_size(BLOCKSIZE);
    int searching_area(block_size*block_size), radius(block_size/2);
    int block_row(src_rows/block_size), block_col(src_cols/block_size);

/*	//debug
	if (direction == 0)
	{
		cout << cur_thread << " thread is working " << " FORWARD" << endl;
	}
	else if (direction == 1)
	{
		cout << cur_thread << " thread is working " << " BACKWARD" << endl;
	}
*/


    for (int r = cur_thread; r < block_row; r += num_thread)
    {
        for (int c = 0; c < block_col; c++)
        {
			//this bug has been fix!!!!!
            int center[2] = {r*block_size + radius,  c*block_size + radius};

            //boundary, use one-at-a-time
            if (r == 0 or c == 0 or c == block_col - 1)
            {
				//printf("cur_t = %d, r=%d\n",cur_thread,r);
                //horizontal search
                int updates_h[6] = {-1,0,0,0,1,0};      //vector<Point> updates_h {Point(-1,0), Point(0,0), Point(1,0)};
                int block_center[2] = {0};
                int init_block_center[2] = {center[0], center[1]};
                oaat(dsrc, ddist, block_center, center, init_block_center, searching_area, updates_h, src_rows, src_cols);
                //vertical search
                int updates_v[6] = {0,-1,0,0,0,1};     //vector<Point> updates_v {Point(0,-1), Point(0,0), Point(0,1)};
                int from_point[2] = {0};
                oaat(dsrc, ddist, from_point, center, block_center, searching_area, updates_v, src_rows, src_cols);

                *(pmm + (r*block_col*2 + c*2 + 0)) = (center[0] - from_point[0]);  //what I trying to do : motion_map[r][c] = (center - from_point);
                *(pmm + (r*block_col*2 + c*2 + 1)) = (center[1] - from_point[1]);
				//printf("pmm = %d,%d\n",*(pmm + (r*block_col*2 + c*2 + 0)),*(pmm + (r*block_col*2 + c*2 + 1)));
            }
            // 3d recursive searching
            else
            {
                //magic number from paper
                int p(8);
                int updates[16] = {1,-1, -1, 1, 0, 2, 0, -2, 4, 0, -4, 0, 5, 2, -5, -2};

                //initial estimation is early cauculated value
                int Da_current[2] = { *(plmm+((r-1)*block_col*2 + (c-1)*2 + 0)), *(plmm + ( (r-1)*block_col*2 + (c-1)*2 + 1))}; //motion_map[r-1][c-1]	//Sa
                int Db_current[2] = { *(plmm+((r-1)*block_col*2 + (c+1)*2 + 0)), *(plmm + ( (r-1)*block_col*2 + (c+1)*2 + 1))}; //motion_map[r-1][c+1]	//Sb

                //inital CAs
                int Da_previous[2] = {0,0};
                int Db_previous[2] = {0,0};

                //if there is CAs
                if (c > 1 and r < block_row -1 and c < block_row -1)
                {
                    Da_previous[0] =  *(plmm + ((r+1)*block_col*2 + (c+1)*2 + 0)); //last_motion[r+1][c+1]	//Ta
                    Da_previous[1] =  *(plmm + ((r+1)*block_col*2 + (c+1)*2 + 1));
                    Db_previous[0] =  *(plmm + ((r+1)*block_col*2 + (c-1)*2 + 0)); //last_motion[r+1][c-1]	//Tb
                    Db_previous[1] =  *(plmm + ((r+1)*block_col*2 + (c-1)*2 + 1));
                }

                int block_cnt_a(r*c), block_cnt_b(r*c+2);
				//printf("block_cnt = %d,%d\n",block_cnt_a,block_cnt_b);
                float SAD_a(100000.), SAD_b(100000.);
                int not_update_a(0), not_update_b(0);

                while (true)
                {
                    int candidate_a[8] = {0};        int candidate_b[8] = {0};
                    float candidate_sad_a[4] = {0};  float candidate_sad_b[4] = {0};
                    bool candidate_index_a[4] = {0};  bool candidate_index_b[4] = {0};

                    int update_a[2] = {updates[(block_cnt_a %p)*2], updates[(block_cnt_a %p)*2 + 1]};
                    int update_b[2] = {updates[(block_cnt_b %p)*2], updates[(block_cnt_b %p)*2 + 1]};

                    block_cnt_a++; block_cnt_b++;

                    //inital candidate set, 1st : ACS, 2nd : CAs, 3th : 0
                    int cs_a[8] = {Da_current[0], Da_current[1],
                                   Da_current[0] + update_a[0], Da_current[1] + update_a[1],
                                   Da_previous[0], Da_previous[1],
                                   0,0};
                    int cs_b[8] = {Db_current[0], Db_current[1],
                                   Db_current[0] + update_b[0], Db_current[1] + update_b[1],
                                   Db_previous[0], Db_previous[1],
                                   0,0};

                    //get SAD from each candidate, there are 4 candidate each time
                    for (int index = 0; index < 4; index++)
                    {
                        bool out_of_boundary_a(false), out_of_boundary_b(false);
                        int eval_center_a[2] = {center[0] - cs_a[index*2],  center[1] - cs_a[index*2 + 1]};
                        int eval_center_b[2] = {center[0] - cs_b[index*2],  center[1] - cs_b[index*2 + 1]};

                        if (eval_center_a[0] < radius or eval_center_a[1] < radius or eval_center_a[0] > src_rows - radius or eval_center_a[1] > src_cols - radius)
                            out_of_boundary_a = true;
                        if (eval_center_b[0] < radius or eval_center_b[1] < radius or eval_center_b[0] > src_rows - radius or eval_center_b[1] > src_cols - radius)
                            out_of_boundary_b = true;
                        if (out_of_boundary_a and out_of_boundary_b)
                            continue;

                        if (not out_of_boundary_a)
                        {
                            candidate_a[index*2] = cs_a[index*2];
                            candidate_a[index*2 + 1] = cs_a[index*2 + 1];
                            candidate_sad_a[index] = get_sad(dsrc, ddist, center[0], center[1], eval_center_a[0], eval_center_a[1] , src_cols);
                            candidate_index_a[index] = 1;
                        }

                        if (not out_of_boundary_b)
                        {
                            candidate_b[index*2] = cs_b[index*2];
                            candidate_b[index*2 + 1] = cs_b[index*2 + 1];
                            candidate_sad_b[index] = get_sad(dsrc, ddist, center[0], center[1], eval_center_b[0], eval_center_b[1] , src_cols);
                            candidate_index_b[index] = 1;
                        }
                    }

                    //compute penalty from each candidate
                    float min_sad_a(100000.), min_sad_b(100000.);
                    int tmp_update_a[2] = {0};
                    int tmp_update_b[2] = {0};

                    float max_sad_a = get_max(candidate_sad_a, 4);
                    float max_sad_b = get_max(candidate_sad_b, 4);

                    //compute estimator a
                    for (int i = 0; i < 4; i++)
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
                                penalty = 0.004 * max_sad_a * norm(update_a);
                                break;
                            case 2:
                                penalty = 0.008 * max_sad_a;
                                break;
                            case 3:
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

                    //compute estimator b
                    for (int i = 0; i < 4; i++)
                    {
                        if (!candidate_index_b[i])
                            continue;
                        float current_sad = candidate_sad_b[i];
                        float penalty(0);
                        switch (i)
                        {
                            case 0:
                                penalty = 0.;
                                break;
                            case 1:
                                penalty = 0.004 * max_sad_b * norm(update_b);
                                break;
                            case 2:
                                penalty = 0.008 * max_sad_b;
                                break;
                            case 3:
                                penalty = 0.016 * max_sad_b;
                                break;
                        }
                        current_sad += penalty;
                        if (current_sad < min_sad_b)
                        {
                            min_sad_b = current_sad;
                            tmp_update_b[0] = candidate_b[i*2];
                            tmp_update_b[1] = candidate_b[i*2+1];
                        }
                        //prefer 0 in the case of same SAD
                        else if (min_sad_b + 40.  > current_sad and candidate_b[i*2] == 0 and candidate_b[i*2+1] == 0)
                        {
                            tmp_update_b[0] = 0;
                            tmp_update_b[1] = 0;
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

                    if (min_sad_b < SAD_b)
                    {
                        SAD_b = min_sad_b;
                        Db_current[0] = tmp_update_b[0];
                        Db_current[1] = tmp_update_b[1];
                        not_update_b = 0;
                    }
                    else
                        not_update_b += 1;

                    //from paper p373, imporve 2nd withdraw
                    float threshold = 5;
                    if (SAD_a > SAD_b + threshold)
                    {
                        SAD_a = SAD_b;
                        Da_current[0] = Db_current[0];
                        Da_current[1] = Db_current[1];
                        not_update_a = 0;
                    }
                    if (SAD_b > SAD_a + threshold)
                    {
                        SAD_b = SAD_a;
                        Db_current[0] = Da_current[0];
                        Db_current[1] = Da_current[1];
                        not_update_b = 0;
                    }

                    //break if any estiminator converge
                    int check_converge = 2;
                    if (not_update_a > check_converge or not_update_b > check_converge)
                    {
                        break;
                    }
                    //break if out of searching area
                    if (abs(Da_current[0]) > searching_area or abs(Da_current[1]) > searching_area or abs(Db_current[0]) > searching_area or abs(Db_current[1]) > searching_area)
                    {
						//cout << "A: " << Da_current[0] << " ; " << Da_current[1] << endl;
						//cout << "B: " << Db_current[0] << " ; " << Db_current[1] << endl;
                        break;
                    }
                }

                if (SAD_a <= SAD_b)
                {
                    *(pmm + (r*block_col*2 + c*2)) = Da_current[0];
                    *(pmm + (r*block_col*2 + c*2 + 1)) = Da_current[1]; //motion_map[r][c] = Da_current
                }
                else
                {
                    *(pmm + (r*block_col*2 + c*2)) = Db_current[0];
                    *(pmm + (r*block_col*2 + c*2 + 1)) = Db_current[1]; //motion_map[r][c] = Db_current
                }
            }

        }
    }
}






//imread generate a continous matrix
void general_IF_MC(Mat &src_image, Mat &IF_image, Mat &dist_image, int *last_motion)
{
    int block_size(BLOCKSIZE), radius(block_size/2);
    int row(src_image.rows), col(src_image.cols);
    int block_row(row/block_size), block_col(col/block_size);

	//MC
	uchar* current = IF_image.ptr<uchar>();
	for (int r = 0; r< block_row; r++)
	{
		for (int c = 0; c< block_col;c++)
		{
			if (*(last_motion + r*block_col*2 + c*2) != 0 or *(last_motion + r*block_col*2 + c*2 + 1) != 0)	
			{
				Point motion(*(last_motion + r*block_col*2 + c*2) , *(last_motion + r*block_col*2 + c*2 + 1));
				motion.x = int(motion.x*0.5);
				motion.y = int(motion.y*0.5);
				//cout << "motion.x/y is " << motion << endl;
				for (int a = 0; a< block_size; a++)
				{
					for (int b = 0; b < block_size; b++)
					{
						Point prev_dist(r*block_size+a-motion.x,c*block_size+b-motion.y);
						Point next_dist(r*block_size+a+motion.x,c*block_size+b+motion.y);
						if ((prev_dist.x <0) or (prev_dist.x > row-1) or (prev_dist.y <0 ) or (prev_dist.y > col-1))
						{
							continue;
						}
						else if ((next_dist.x <0) or (next_dist.x > row-1) or (next_dist.y <0 ) or (next_dist.y > col-1))
						{
							continue;
						}
						else
						{
							//
							Vec3b src_pix_prev = src_image.at<Vec3b>(r*block_size+a-motion.x,c*block_size+b-motion.y);
							Vec3b src_pix_next = dist_image.at<Vec3b>(r*block_size+a+motion.x,c*block_size+b+motion.y);
							//cout << src_pix_prev << " ; " << src_pix_next << endl;
							for (int i = 0;i <3 ;i++)
							{
								IF_image.at<Vec3b>(r*block_size+a,c*block_size+b)[i] = int(0.5*(src_pix_prev[i]+src_pix_next[i]));			
							}
							
							//cout << IF_image.at<Vec3b>(r*block_size+a,c*block_size+b) << endl;
						}
 
					}
				}
			}	
		}
	}
 

}



//Select the better SAD
void select_sad(uchar *src, uchar *dist, int *current_motion_forward, int *current_motion_backward, int *sad_motion_map, int src_row, int src_col)
{
    int block_size(BLOCKSIZE), radius(block_size/2);
    int row(src_row), col(src_col);
    int block_row(row/block_size), block_col(col/block_size);


	for (int r = 0; r< block_row; r++)
	{
		for (int c = 0; c< block_col;c++)
		{
			int val_forward[2] = {int(fabs(*(current_motion_forward + r*block_col*2 + c*2))),int(fabs(*(current_motion_forward + r*block_col*2 + c*2+1)))};
			int val_backward[2] = {int(fabs(*(current_motion_backward + r*block_col*2 + c*2))),int(fabs(*(current_motion_backward + r*block_col*2 + c*2+1)))};

			//the vector of the both direction motion is equal
			if ((val_forward[0] == val_backward[0]) and (val_forward[1] == val_backward[1]))	
			{
				*(sad_motion_map + r*block_col*2 + c*2) = val_forward[0];
				*(sad_motion_map + r*block_col*2 + c*2+1) = val_forward[1];
			}
			else
			{
				int forward_sad(0), backward(0);

				{
					//forward SAD
					int src_center[2] = {r*block_size+radius-int(val_forward[0]*0.5), c*block_size+radius-int(val_forward[1]*0.5)};
					int dist_center[2] = {r*block_size+radius+int(val_forward[0]*0.5), c*block_size+radius+int(val_forward[1]*0.5)};

					forward_sad = get_sad(src, dist, src_center[0], src_center[1], dist_center[0], dist_center[1],src_col);
					//cout << forward_sad <<endl;
				}

				{
					//backward SAD
					int src_center[2] = {r*block_size+radius-int(val_backward[0]*0.5), c*block_size+radius-int(val_backward[1]*0.5)};
					int dist_center[2] = {r*block_size+radius+int(val_backward[0]*0.5), c*block_size+radius+int(val_backward[1]*0.5)};

					backward = get_sad(dist, src, src_center[0], src_center[1], dist_center[0], dist_center[1],src_col);
					//cout << backward <<endl;
				}
			

				if(forward_sad <= backward)
				{
					*(sad_motion_map + r*block_col*2 + c*2) = val_forward[0];
					*(sad_motion_map + r*block_col*2 + c*2+1) = val_forward[1];	
				}
				else
				{
					*(sad_motion_map + r*block_col*2 + c*2) = -val_backward[0];
					*(sad_motion_map + r*block_col*2 + c*2+1) = -val_backward[1];	
				}
			}

			//cout << *(sad_motion_map + r*block_col*2 + c*2) << "	" << *(sad_motion_map + r*block_col*2 + c*2+1) <<endl;	
		}
	}
}


//imread generate a continous matrix
void tdrs_both(Mat &src, Mat &dist, Mat &src_image, Mat &IF_image, int *last_motion_forward, int *last_motion_backward)
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

	int *pmm_better_sad;
	pmm_better_sad = new int[size];

    int total_thread = CPU_THREAD;
    thread ts_f[total_thread];
	thread ts_b[total_thread];

	int forward_direction = 0;
	int backward_direction =1;
	//forward
    for (int i = 0; i < total_thread; i++)
	{
        ts_f[i] = thread(tdrs_thread, dsrc, ddist, plmm_f, pmm_f, src.rows, src.cols, i, total_thread, forward_direction);
	}

	//backward
    for (int i = 0; i < total_thread; i++)
	{
        ts_b[i] = thread(tdrs_thread, ddist, dsrc, plmm_b, pmm_b, src.rows, src.cols, i, total_thread, backward_direction);
	}
	
	//thread.join
    for (int i = 0; i < total_thread; i++)
	{
        ts_f[i].join();
        ts_b[i].join();
	}


	//select the better SAD
	select_sad(dsrc, ddist, pmm_f, pmm_b, pmm_better_sad, row, col);

    //update last_motion for next frame
    memcpy(last_motion_forward, pmm_better_sad, size*sizeof(int));
    memcpy(last_motion_backward, pmm_better_sad, size*sizeof(int));

	//IF_forward
	Mat dist_image = IF_image.clone();
	general_IF_MC(src_image,IF_image,dist_image,last_motion_forward);

    //free memory
    delete [] pmm_f; delete [] plmm_f; 
    delete [] pmm_b; delete [] plmm_b;
	delete [] motion_map_forward; delete [] motion_map_backward;
	delete [] pmm_better_sad;
	delete [] dsrc; delete [] ddist; 
	
	
} 



//point(x,y) means (col, row)
int main(int argc, char**argv)
{
	printf("start\n");
	string file_name = "/home/iqiyi/Desktop/ESPCN_shijie/True-Motion-Estimation/Oigin_SDBronze.mp4";
	//string file_name = "./SDBronze0001.bmp";

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
	// *2 --> to save x and y of the motion
    int *motion_map_forward = new int[src.cols/block_size * src.rows/block_size * 2]();
    int *motion_map_backward = new int[src.cols/block_size * src.rows/block_size * 2]();

    chrono::duration<float, milli> dtn;
    float avg_dtn;

    //VideoWriter record("record.avi", CV_FOURCC('M','J','P','G'), 60., Size(src.cols, src.rows), true);

	
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

		//debug
		//imwrite(boost::str(boost::format("./video/%04d_a.jpg") %cnt).c_str(), src);	
		//imwrite(boost::str(boost::format("./video/%04d_b.jpg") %cnt).c_str(), dist);
		//imshow("src",gsrc);
		//imshow("dist",gdist);

        //tdrs(gsrc, gdist, out, motion_map);
		IF_img = dist.clone();
		
		//bothward: FORWARD and BACKWARD
		tdrs_both(gsrc, gdist, src, IF_img, motion_map_forward, motion_map_backward);

		//frame reflash to the next circle
        src = dist.clone();

        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        dtn = end - start;
        avg_dtn = (cnt/float(cnt+1))*avg_dtn + (dtn.count()/float(cnt+1));
        cnt++;
		//cout << "cnt is " << cnt << endl;

        string tmp = boost::str(boost::format("%2.2fms / %2.2fms")% dtn.count()  %avg_dtn );
        putText(out, tmp, cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,200,0), 1, CV_AA);
		putText(IF_img, tmp, cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,200,0), 1, CV_AA);

		//record the video
		//record.write(IF_img);
        //record.write(out);

		//save the image of the output
		//imwrite(boost::str(boost::format("./video/%04d_a.jpg") %cnt).c_str(), IF_img);
		//imwrite(boost::str(boost::format("./video/%04d_b.jpg") %cnt).c_str(), out);	

        imshow("motion", IF_img);
		//imshow("motion", out);
		//waitKey(0);

        if (char(waitKey(1)) == 'q')
            break;
    }
    delete [] motion_map_forward;
	delete [] motion_map_backward;

	cout << "the process has been completed" <<endl;
    return 0;
}
