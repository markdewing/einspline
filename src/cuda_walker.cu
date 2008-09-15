#include "cuda_walker.h"
#include "determinant_update.h"

cuda_determinant::cuda_determinant() : 
  N(0), A(NULL), Ainv(NULL), Ainv_delta(NULL), Ainv_colk(0),
  new_row(NULL), delta(0)
{

};

cuda_determinant::cuda_determinant(int n)
{
  resize(N);
}

void
cuda_determinant::resize(int n)
{
  N = n;
  cudaMalloc((void**)&A         , N*N*sizeof(float));
  cudaMalloc((void**)&Ainv      , N*N*sizeof(float));
  cudaMalloc((void**)&Ainv_delta, 1*N*sizeof(float));
  cudaMalloc((void**)&Ainv_colk , 1*N*sizeof(float));
  cudaMalloc((void**)&new_row   , 1*N*sizeof(float));
  cudaMalloc((void**)&delta     , 1*N*sizeof(float));
}

void
cuda_walker::resize(int nup, int ndown) 
{
  N[0] = nup; N[1] = ndown;
  dets[0].resize(N[0]);
  dets[1].resize(N[1]);
}



cuda_population::cuda_population() : MaxPop(1000)
{
  A_vec.resize(MaxPop);
  Ainv_vec.resize(MaxPop);
  delta_vec.resize(MaxPop);
  Ainv_delta_vec.resize(MaxPop);
  Ainv_colk_vec.resize(MaxPop);
  ratio_vec.resize(MaxPop);
  pos_vec.resize(3*MaxPop);


  cudaMalloc((void**) &A_list_d,          MaxPop*sizeof(float*));
  cudaMalloc((void**) &Ainv_list_d,       MaxPop*sizeof(float*));
  cudaMalloc((void**) &Ainv_delta_list_d, MaxPop*sizeof(float*));
  cudaMalloc((void**) &Ainv_colk_list_d,  MaxPop*sizeof(float*));
  cudaMalloc((void**) &delta_list_d,      MaxPop*sizeof(float*));
  cudaMalloc((void**) &ratios_d,          MaxPop*sizeof(float));
  cudaMalloc((void**) &pos_d,           4*MaxPop*sizeof(float));
}


__global__ static void
update_inverse_cuda1 (float *Ainv_g[], float *u_g[], float *Ainv_u_g[],
		      float *Ainv_colk_g[], int N, int rowstride, int k);
__global__ static void
update_inverse_cuda2 (float *Ainv_g[], float *u_g[], float *Ainv_u_g[],
		      float *Ainv_colk_g[], int N, int rowstride, int k);


void
cuda_population::calc_new_row(int elec)
{
  int detnum = (elec < num_elecs[0]) ? 0 : 1;
  int N = num_elecs[detnum];
  for (int wi=0; wi<walkers.size(); wi++) {
    cuda_walker &w = walkers[wi];
    cuda_determinant &det = w.dets[detnum];
    pos_vec[4*wi+0] = w.R[3*elec+0];
    pos_vec[4*wi+1] = w.R[3*elec+1];
    pos_vec[4*wi+2] = w.R[3*elec+2];
    delta_vec[wi] = det.delta;
  }
  cudaMemcpy(pos_d, &(pos_vec[0]), 4*walkers.size()*sizeof(float), 
	     cudaMemcpyHostToDevice);
  cudaMemcpy(delta_list_d, &(delta_vec[0]), walkers.size()*sizeof(float*), 
	     cudaMemcpyHostToDevice);

  dim3 dimBlock(SPLINE_BLOCK_SIZE);
  dim3 dimGrid (N/SPLINE_BLOCK_SIZE, walkers.size());
  
  eval_multi_multi_UBspline_3d_s_cuda<<<dimGrid,dimBlock>>>
    (pos_d, multi_spline->gridInv, multi_spline->coefs,
     delta_list_d, multi_spline->stride);

}


void 
cuda_population::update_determinants(int elec)
{
  int index=0;
  int detnum = (elec < num_elecs[0]) ? 0 : 1;
  int N = num_elecs[detnum];
  int row = (elec < num_elecs[0]) ? elec : elec - num_elecs[0];
  for (int wi=0; wi<walkers.size(); wi++) {
    cuda_walker &w = walkers[wi];
    cuda_determinant &det = w.dets[detnum];
    if (w.accept) {
      Ainv_vec[index]       = det.Ainv;
      Ainv_delta_vec[index] = det.Ainv_delta;
      Ainv_colk_vec[index]  = det.Ainv_colk;
      delta_vec[index]      = det.delta;
      index++;
    }
  }
  int num_accept = index;

  cudaMemcpy (Ainv_list_d, &(Ainv_vec[0]), 
	      num_accept*sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy (Ainv_delta_list_d, &(Ainv_delta_vec[0]),
	      num_accept*sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy (Ainv_colk_list_d, &(Ainv_colk_vec[0]), 
	      num_accept*sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy (delta_list_d, &(delta_vec[0]), 
	      num_accept*sizeof(float*), cudaMemcpyHostToDevice);

  dim3 dimBlock(DET_BLOCK_SIZE);
  dim3 dimGrid (N/DET_BLOCK_SIZE, num_accept);
  
  update_inverse_cuda1<<<dimGrid,dimBlock>>>
      (Ainv_list_d, delta_list_d, Ainv_delta_list_d, 
       Ainv_colk_list_d, N, N, row);
  update_inverse_cuda2<<<dimGrid,dimBlock>>>
      (Ainv_list_d, delta_list_d, Ainv_delta_list_d, 
       Ainv_colk_list_d, N, N, row);
};
