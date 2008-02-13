/////////////////////////////////////////////////////////////////////////////
//  einspline:  a library for creating and evaluating B-splines            //
//  Copyright (C) 2007 Kenneth P. Esler, Jr.                               //
//                                                                         //
//  This program is free software; you can redistribute it and/or modify   //
//  it under the terms of the GNU General Public License as published by   //
//  the Free Software Foundation; either version 2 of the License, or      //
//  (at your option) any later version.                                    //
//                                                                         //
//  This program is distributed in the hope that it will be useful,        //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of         //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          //
//  GNU General Public License for more details.                           //
//                                                                         //
//  You should have received a copy of the GNU General Public License      //
//  along with this program; if not, write to the Free Software            //
//  Foundation, Inc., 51 Franklin Street, Fifth Floor,                     //
//  Boston, MA  02110-1301  USA                                            //
/////////////////////////////////////////////////////////////////////////////

#ifndef MULTI_BSPLINE_EVAL_SSE_D_H
#define MULTI_BSPLINE_EVAL_SSE_D_H

#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <stdio.h>
#include <math.h>
extern __m128d *restrict A_d;
#include "multi_bspline_structs.h"

#ifndef _MM_DDOT4_PD
#ifdef HAVE_SSE3
#define _MM_DDOT4_PD(a0, a1, a2, a3, b0, b1, b2, b3, r)               \
do {                                                                  \
   __m128d t0 = _mm_add_pd(_mm_mul_pd (a0, b0),_mm_mul_pd (a1, b1));  \
   __m128d t1 = _mm_add_pd(_mm_mul_pd (a2, b2),_mm_mul_pd (a3, b3));  \
   r = _mm_hadd_pd (t0, t1);                                          \
 } while(0);
#define _MM_DOT4_PD(a0, a1, b0, b1, p)                                \
do {                                                                  \
  __m128d t0 = _mm_add_pd(_mm_mul_pd (a0, b0),_mm_mul_pd (a1, b1));   \
  __m128d t1 = _mm_hadd_pd (t0,t0);                                   \
  _mm_store_sd (&(p), t1);                                            \
 } while (0);
#else
#define _MM_DDOT4_PD(a0, a1, a2, a3, b0, b1, b2, b3, r)               \
do {                                                                  \
   __m128d t0 = _mm_add_pd(_mm_mul_pd (a0, b0),_mm_mul_pd (a1, b1));  \
   __m128d t1 = _mm_add_pd(_mm_mul_pd (a2, b2),_mm_mul_pd (a3, b3));  \
   r = _mm_add_pd(_mm_unpacklo_pd(t0,t1),_mm_unpackhi_pd(t0,t1));     \
 } while(0);
#define _MM_DOT4_PD(a0, a1, b0, b1, p)                                \
do {                                                                  \
  __m128d t0 = _mm_add_pd(_mm_mul_pd (a0, b0),_mm_mul_pd (a1, b1));   \
  __m128d t1 =                                                        \
      _mm_add_pd (_mm_unpacklo_pd(t0,t0), _mm_unpackhi_pd(t0,t0));    \
  _mm_store_d (&(p), t1);                                            \
 } while (0);
#endif
#endif

/************************************************************/
/* 1D double-precision, complex evaulation functions        */
/************************************************************/


/************************************************************/
/* 2D double-precision, complex evaulation functions        */
/************************************************************/


/************************************************************/
/* 3D double-precision, complex evaulation functions        */
/************************************************************/
inline void
eval_multi_UBspline_3d_d (multi_UBspline_3d_d *spline,
			  double x, double y, double z,
			  double* restrict vals)
{
  _mm_prefetch ((const char*) &A_d[0],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[1],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[2],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[3],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[4],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[5],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[6],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[7],_MM_HINT_T0);  

  x -= spline->x_grid.start;
  y -= spline->y_grid.start;  
  z -= spline->z_grid.start;
  double ux = x*spline->x_grid.delta_inv;
  double uy = y*spline->y_grid.delta_inv;
  double uz = z*spline->z_grid.delta_inv;
  ux = fmin (ux, (double)(spline->x_grid.num)-1.0e-5);
  uy = fmin (uy, (double)(spline->y_grid.num)-1.0e-5);
  uz = fmin (uz, (double)(spline->z_grid.num)-1.0e-5);
  double ipartx, iparty, ipartz, tx, ty, tz;
  tx = modf (ux, &ipartx);  int ix = (int) ipartx;
  ty = modf (uy, &iparty);  int iy = (int) iparty;
  tz = modf (uz, &ipartz);  int iz = (int) ipartz;
  
  int xs = spline->x_stride;
  int ys = spline->y_stride;
  int zs = spline->z_stride;
  int N  = spline->num_splines;

  // Now compute the vectors:
  // tpx = [t_x^3 t_x^2 t_x 1]
  // tpy = [t_y^3 t_y^2 t_y 1]
  // tpz = [t_z^3 t_z^2 t_z 1]

  // a  =  A * tpx,   b =  A * tpy,   c =  A * tpz
  // A is 4x4 matrix given by the rows A0, A1, A2, A3
  __m128d tpx01, tpx23, tpy01, tpy23, tpz01, tpz23,
    a01, b01, c01, a23, b23, c23,  
    tmp0, tmp1, r0, r1, i0, i1, val_r, val_i;
  
  tpx01 = _mm_set_pd (tx*tx*tx, tx*tx);
  tpx23 = _mm_set_pd (tx, 1.0);
  tpy01 = _mm_set_pd (ty*ty*ty, ty*ty);
  tpy23 = _mm_set_pd (ty, 1.0);
  tpz01 = _mm_set_pd (tz*tz*tz, tz*tz);
  tpz23 = _mm_set_pd (tz, 1.0);

  // x-dependent vectors
  _MM_DDOT4_PD (A_d[0], A_d[1], A_d[2], A_d[3], tpx01, tpx23, tpx01, tpx23,   a01);
  _MM_DDOT4_PD (A_d[4], A_d[5], A_d[6], A_d[7], tpx01, tpx23, tpx01, tpx23,   a23);
  // y-dependent vectors
  _MM_DDOT4_PD (A_d[0], A_d[1], A_d[2], A_d[3], tpy01, tpy23, tpy01, tpy23,   b01);
  _MM_DDOT4_PD (A_d[4], A_d[5], A_d[6], A_d[7], tpy01, tpy23, tpy01, tpy23,   b23);
  // z-dependent vectors
  _MM_DDOT4_PD (A_d[0], A_d[1], A_d[2], A_d[3], tpz01, tpz23, tpz01, tpz23,   c01);
  _MM_DDOT4_PD (A_d[4], A_d[5], A_d[6], A_d[7], tpz01, tpz23, tpz01, tpz23,   c23);

  // Zero-out values
  __m128d mvals[(N+1)/2];
  for (int n=0; n<N; n++) 
    mvals[n] = _mm_sub_pd (mvals[n], mvals[n]);

  __m128d a[4], b[4], c[4];
  a[0]   = _mm_unpacklo_pd(a01,a01);
  a[1]   = _mm_unpackhi_pd(a01,a01);
  a[2]   = _mm_unpacklo_pd(a23,a23);
  a[3]   = _mm_unpackhi_pd(a23,a23);

  b[0]   = _mm_unpacklo_pd(b01,b01);
  b[1]   = _mm_unpackhi_pd(b01,b01);
  b[2]   = _mm_unpacklo_pd(b23,b23);
  b[3]   = _mm_unpackhi_pd(b23,b23);

  c[0]   = _mm_unpacklo_pd(c01,c01);
  c[1]   = _mm_unpackhi_pd(c01,c01);
  c[2]   = _mm_unpacklo_pd(c23,c23);
  c[3]   = _mm_unpackhi_pd(c23,c23);

  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++) 
      for (int k=0; k<4; k++) {
	__m128d abc = _mm_mul_pd (_mm_mul_pd(a[i], b[j]), c[k]);
	__m128d* restrict coefs = (__m128d*)(spline->coefs + (ix+i)*xs + (iy+j)*ys + (iz+k)*zs);

	for (int n=0; n<(N+1)/2; n++) 
	  mvals[n] = _mm_add_pd (mvals[n], _mm_mul_pd (abc, coefs[n]));
      }
  
  for (int n=0; n<N/2; n++) 
    _mm_storeu_pd((vals+2*n),mvals[n]);
  if (N & 1) 
    _mm_storel_pd(vals+N-1,mvals[N/2]);
}



inline void
eval_multi_UBspline_3d_d_vg (multi_UBspline_3d_d *spline,
			     double x, double y, double z,
			     double* restrict vals,
			     double* restrict grads)
{
  _mm_prefetch ((const char*) &A_d[ 0],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[ 1],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[ 2],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[ 3],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[ 4],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[ 5],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[ 6],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[ 7],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[ 8],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[ 9],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[10],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[11],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[12],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[13],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[14],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[15],_MM_HINT_T0);  

  x -= spline->x_grid.start;
  y -= spline->y_grid.start;  
  z -= spline->z_grid.start;
  double ux = x*spline->x_grid.delta_inv;
  double uy = y*spline->y_grid.delta_inv;
  double uz = z*spline->z_grid.delta_inv;
  ux = fmin (ux, (double)(spline->x_grid.num)-1.0e-5);
  uy = fmin (uy, (double)(spline->y_grid.num)-1.0e-5);
  uz = fmin (uz, (double)(spline->z_grid.num)-1.0e-5);
  double ipartx, iparty, ipartz, tx, ty, tz;
  tx = modf (ux, &ipartx);  int ix = (int) ipartx;
  ty = modf (uy, &iparty);  int iy = (int) iparty;
  tz = modf (uz, &ipartz);  int iz = (int) ipartz;
  
  int xs = spline->x_stride;
  int ys = spline->y_stride;
  int zs = spline->z_stride;
  int N  = spline->num_splines;

  // Now compute the vectors:
  // tpx = [t_x^3 t_x^2 t_x 1]
  // tpy = [t_y^3 t_y^2 t_y 1]
  // tpz = [t_z^3 t_z^2 t_z 1]

  // a  =  A * tpx,   b =  A * tpy,   c =  A * tpz
  // A is 4x4 matrix given by the rows A0, A1, A2, A3
  __m128d tpx01, tpx23, tpy01, tpy23, tpz01, tpz23,
    a01  ,   b01,   c01,   a23,    b23,   c23,  
    da01 ,  db01,  dc01,  da23,   db23,  dc23;

  tpx01 = _mm_set_pd (tx*tx*tx, tx*tx);
  tpx23 = _mm_set_pd (tx, 1.0);
  tpy01 = _mm_set_pd (ty*ty*ty, ty*ty);
  tpy23 = _mm_set_pd (ty, 1.0);
  tpz01 = _mm_set_pd (tz*tz*tz, tz*tz);
  tpz23 = _mm_set_pd (tz, 1.0);

  // x-dependent vectors
  _MM_DDOT4_PD (A_d[ 0], A_d[ 1], A_d[ 2], A_d[ 3], tpx01, tpx23, tpx01, tpx23,   a01);
  _MM_DDOT4_PD (A_d[ 4], A_d[ 5], A_d[ 6], A_d[ 7], tpx01, tpx23, tpx01, tpx23,   a23);
  _MM_DDOT4_PD (A_d[ 8], A_d[ 9], A_d[10], A_d[11], tpx01, tpx23, tpx01, tpx23,  da01);
  _MM_DDOT4_PD (A_d[12], A_d[13], A_d[14], A_d[15], tpx01, tpx23, tpx01, tpx23,  da23);

  // y-dependent vectors
  _MM_DDOT4_PD (A_d[ 0], A_d[ 1], A_d[ 2], A_d[ 3], tpy01, tpy23, tpy01, tpy23,   b01);
  _MM_DDOT4_PD (A_d[ 4], A_d[ 5], A_d[ 6], A_d[ 7], tpy01, tpy23, tpy01, tpy23,   b23);
  _MM_DDOT4_PD (A_d[ 8], A_d[ 9], A_d[10], A_d[11], tpy01, tpy23, tpy01, tpy23,  db01);
  _MM_DDOT4_PD (A_d[12], A_d[13], A_d[14], A_d[15], tpy01, tpy23, tpy01, tpy23,  db23);


  // z-dependent vectors
  _MM_DDOT4_PD (A_d[ 0], A_d[ 1], A_d[ 2], A_d[ 3], tpz01, tpz23, tpz01, tpz23,   c01);
  _MM_DDOT4_PD (A_d[ 4], A_d[ 5], A_d[ 6], A_d[ 7], tpz01, tpz23, tpz01, tpz23,   c23);
  _MM_DDOT4_PD (A_d[ 8], A_d[ 9], A_d[10], A_d[11], tpz01, tpz23, tpz01, tpz23,  dc01);
  _MM_DDOT4_PD (A_d[12], A_d[13], A_d[14], A_d[15], tpz01, tpz23, tpz01, tpz23,  dc23);


  // Zero-out values
  int Nh = (N+1)/2;
  __m128d mvals[Nh], mgrads[3*Nh];
  for (int n=0; n<Nh; n++) {
    mvals[n] = _mm_sub_pd (mvals[n], mvals[n]);
    for (int i=0; i<3; i++)
      mgrads[3*n+i] = _mm_sub_pd (mgrads[3*n+i],mgrads[3*n+i]);
  }

  __m128d a[4], b[4], c[4], da[4], db[4], dc[4];
  a[0]=_mm_unpacklo_pd(a01,a01); da[0]=_mm_unpacklo_pd(da01,da01); 
  a[1]=_mm_unpackhi_pd(a01,a01); da[1]=_mm_unpackhi_pd(da01,da01); 
  a[2]=_mm_unpacklo_pd(a23,a23); da[2]=_mm_unpacklo_pd(da23,da23); 
  a[3]=_mm_unpackhi_pd(a23,a23); da[3]=_mm_unpackhi_pd(da23,da23); 
				 				   
  b[0]=_mm_unpacklo_pd(b01,b01); db[0]=_mm_unpacklo_pd(db01,db01); 
  b[1]=_mm_unpackhi_pd(b01,b01); db[1]=_mm_unpackhi_pd(db01,db01); 
  b[2]=_mm_unpacklo_pd(b23,b23); db[2]=_mm_unpacklo_pd(db23,db23); 
  b[3]=_mm_unpackhi_pd(b23,b23); db[3]=_mm_unpackhi_pd(db23,db23); 
				 				   
  c[0]=_mm_unpacklo_pd(c01,c01); dc[0]=_mm_unpacklo_pd(dc01,dc01); 
  c[1]=_mm_unpackhi_pd(c01,c01); dc[1]=_mm_unpackhi_pd(dc01,dc01); 
  c[2]=_mm_unpacklo_pd(c23,c23); dc[2]=_mm_unpacklo_pd(dc23,dc23); 
  c[3]=_mm_unpackhi_pd(c23,c23); dc[3]=_mm_unpackhi_pd(dc23,dc23); 

  // Main computation loop
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++) 
      for (int k=0; k<4; k++) {
	__m128d abc, d_abc[3];
	
	abc         = _mm_mul_pd (_mm_mul_pd(a[i], b[j]), c[k]);

	d_abc[0]    = _mm_mul_pd (_mm_mul_pd(da[i],  b[j]),  c[k]);
	d_abc[1]    = _mm_mul_pd (_mm_mul_pd( a[i], db[j]),  c[k]);
	d_abc[2]    = _mm_mul_pd (_mm_mul_pd( a[i],  b[j]), dc[k]);

	__m128d* restrict coefs = (__m128d*)(spline->coefs + (ix+i)*xs + (iy+j)*ys + (iz+k)*zs);

	for (int n=0; n<Nh; n++) {
	  mvals[n]      = _mm_add_pd (mvals[n],      _mm_mul_pd (   abc   , coefs[n]));
	  mgrads[3*n+0] = _mm_add_pd (mgrads[3*n+0], _mm_mul_pd ( d_abc[0], coefs[n]));
	  mgrads[3*n+1] = _mm_add_pd (mgrads[3*n+1], _mm_mul_pd ( d_abc[1], coefs[n]));
	  mgrads[3*n+2] = _mm_add_pd (mgrads[3*n+2], _mm_mul_pd ( d_abc[2], coefs[n]));
	}
      }
  
  double dxInv = spline->x_grid.delta_inv;
  double dyInv = spline->y_grid.delta_inv;
  double dzInv = spline->z_grid.delta_inv; 
  
  for (int n=0; n<N/2; n++) {
    _mm_storeu_pd((double*)(vals+2*n),mvals[n]);

    _mm_storel_pd((double*)(grads+6*n+0),mgrads[3*n+0]);
    _mm_storeh_pd((double*)(grads+6*n+3),mgrads[3*n+0]);
    _mm_storel_pd((double*)(grads+6*n+1),mgrads[3*n+1]);
    _mm_storeh_pd((double*)(grads+6*n+4),mgrads[3*n+1]);
    _mm_storel_pd((double*)(grads+6*n+2),mgrads[3*n+2]);
    _mm_storeh_pd((double*)(grads+6*n+5),mgrads[3*n+2]);
  }

  if (N&1) {
    _mm_storel_pd((double*)(vals+N-1),mvals[Nh-1]);

    _mm_storel_pd((double*)(grads+3*(N-1)+0),mgrads[3*(Nh-1)+0]);
    _mm_storel_pd((double*)(grads+3*(N-1)+1),mgrads[3*(Nh-1)+1]);
    _mm_storel_pd((double*)(grads+3*(N-1)+2),mgrads[3*(Nh-1)+2]);
  }

  for (int n=0; n<N; n++) {
    grads[3*n+0] *= dxInv;
    grads[3*n+1] *= dyInv;
    grads[3*n+2] *= dzInv;
  }
}



inline void
eval_multi_UBspline_3d_d_vgl (multi_UBspline_3d_d *spline,
			      double x, double y, double z,
			      double* restrict vals,
			      double* restrict grads,
			      double* restrict lapl)
{
  _mm_prefetch ((const char*) &A_d[ 0],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[ 1],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[ 2],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[ 3],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[ 4],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[ 5],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[ 6],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[ 7],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[ 8],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[ 9],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[10],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[11],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[12],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[13],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[14],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[15],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[16],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[17],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[18],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[19],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[20],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[21],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[22],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[23],_MM_HINT_T0);  

  x -= spline->x_grid.start;
  y -= spline->y_grid.start;  
  z -= spline->z_grid.start;
  double ux = x*spline->x_grid.delta_inv;
  double uy = y*spline->y_grid.delta_inv;
  double uz = z*spline->z_grid.delta_inv;
  ux = fmin (ux, (double)(spline->x_grid.num)-1.0e-5);
  uy = fmin (uy, (double)(spline->y_grid.num)-1.0e-5);
  uz = fmin (uz, (double)(spline->z_grid.num)-1.0e-5);
  double ipartx, iparty, ipartz, tx, ty, tz;
  tx = modf (ux, &ipartx);  int ix = (int) ipartx;
  ty = modf (uy, &iparty);  int iy = (int) iparty;
  tz = modf (uz, &ipartz);  int iz = (int) ipartz;
  
  int xs = spline->x_stride;
  int ys = spline->y_stride;
  int zs = spline->z_stride;
  int N  = spline->num_splines;

  // Now compute the vectors:
  // tpx = [t_x^3 t_x^2 t_x 1]
  // tpy = [t_y^3 t_y^2 t_y 1]
  // tpz = [t_z^3 t_z^2 t_z 1]

  // a  =  A * tpx,   b =  A * tpy,   c =  A * tpz
  // A is 4x4 matrix given by the rows A0, A1, A2, A3
  __m128d tpx01, tpx23, tpy01, tpy23, tpz01, tpz23,
    a01  ,   b01,   c01,   a23,    b23,   c23,  
    da01 ,  db01,  dc01,  da23,   db23,  dc23,  
    d2a01, d2b01, d2c01, d2a23,  d2b23, d2c23;

  tpx01 = _mm_set_pd (tx*tx*tx, tx*tx);
  tpx23 = _mm_set_pd (tx, 1.0);
  tpy01 = _mm_set_pd (ty*ty*ty, ty*ty);
  tpy23 = _mm_set_pd (ty, 1.0);
  tpz01 = _mm_set_pd (tz*tz*tz, tz*tz);
  tpz23 = _mm_set_pd (tz, 1.0);

  // x-dependent vectors
  _MM_DDOT4_PD (A_d[ 0], A_d[ 1], A_d[ 2], A_d[ 3], tpx01, tpx23, tpx01, tpx23,   a01);
  _MM_DDOT4_PD (A_d[ 4], A_d[ 5], A_d[ 6], A_d[ 7], tpx01, tpx23, tpx01, tpx23,   a23);
  _MM_DDOT4_PD (A_d[ 8], A_d[ 9], A_d[10], A_d[11], tpx01, tpx23, tpx01, tpx23,  da01);
  _MM_DDOT4_PD (A_d[12], A_d[13], A_d[14], A_d[15], tpx01, tpx23, tpx01, tpx23,  da23);
  _MM_DDOT4_PD (A_d[16], A_d[17], A_d[18], A_d[19], tpx01, tpx23, tpx01, tpx23, d2a01);
  _MM_DDOT4_PD (A_d[20], A_d[21], A_d[22], A_d[23], tpx01, tpx23, tpx01, tpx23, d2a23);

  // y-dependent vectors
  _MM_DDOT4_PD (A_d[ 0], A_d[ 1], A_d[ 2], A_d[ 3], tpy01, tpy23, tpy01, tpy23,   b01);
  _MM_DDOT4_PD (A_d[ 4], A_d[ 5], A_d[ 6], A_d[ 7], tpy01, tpy23, tpy01, tpy23,   b23);
  _MM_DDOT4_PD (A_d[ 8], A_d[ 9], A_d[10], A_d[11], tpy01, tpy23, tpy01, tpy23,  db01);
  _MM_DDOT4_PD (A_d[12], A_d[13], A_d[14], A_d[15], tpy01, tpy23, tpy01, tpy23,  db23);
  _MM_DDOT4_PD (A_d[16], A_d[17], A_d[18], A_d[19], tpy01, tpy23, tpy01, tpy23, d2b01);
  _MM_DDOT4_PD (A_d[20], A_d[21], A_d[22], A_d[23], tpy01, tpy23, tpy01, tpy23, d2b23);


  // z-dependent vectors
  _MM_DDOT4_PD (A_d[ 0], A_d[ 1], A_d[ 2], A_d[ 3], tpz01, tpz23, tpz01, tpz23,   c01);
  _MM_DDOT4_PD (A_d[ 4], A_d[ 5], A_d[ 6], A_d[ 7], tpz01, tpz23, tpz01, tpz23,   c23);
  _MM_DDOT4_PD (A_d[ 8], A_d[ 9], A_d[10], A_d[11], tpz01, tpz23, tpz01, tpz23,  dc01);
  _MM_DDOT4_PD (A_d[12], A_d[13], A_d[14], A_d[15], tpz01, tpz23, tpz01, tpz23,  dc23);
  _MM_DDOT4_PD (A_d[16], A_d[17], A_d[18], A_d[19], tpz01, tpz23, tpz01, tpz23, d2c01);
  _MM_DDOT4_PD (A_d[20], A_d[21], A_d[22], A_d[23], tpz01, tpz23, tpz01, tpz23, d2c23);


  // Zero-out values
  int Nh = (N+1)/2;
  __m128d mvals[Nh], mgrads[3*Nh], mlapl[3*Nh];
  for (int n=0; n<Nh; n++) {
    mvals[n] = _mm_sub_pd (mvals[n], mvals[n]);
    for (int i=0; i<3; i++) {
      mgrads[3*n+i] = _mm_sub_pd (mgrads[3*n+i],mgrads[3*n+i]);
      mlapl[3*n+i]  = _mm_sub_pd ( mlapl[3*n+i], mlapl[3*n+i]);
    }
  }

  __m128d a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
  a[0]=_mm_unpacklo_pd(a01,a01); da[0]=_mm_unpacklo_pd(da01,da01); d2a[0]=_mm_unpacklo_pd(d2a01,d2a01);
  a[1]=_mm_unpackhi_pd(a01,a01); da[1]=_mm_unpackhi_pd(da01,da01); d2a[1]=_mm_unpackhi_pd(d2a01,d2a01);
  a[2]=_mm_unpacklo_pd(a23,a23); da[2]=_mm_unpacklo_pd(da23,da23); d2a[2]=_mm_unpacklo_pd(d2a23,d2a23);
  a[3]=_mm_unpackhi_pd(a23,a23); da[3]=_mm_unpackhi_pd(da23,da23); d2a[3]=_mm_unpackhi_pd(d2a23,d2a23);
				 				   				  
  b[0]=_mm_unpacklo_pd(b01,b01); db[0]=_mm_unpacklo_pd(db01,db01); d2b[0]=_mm_unpacklo_pd(d2b01,d2b01);
  b[1]=_mm_unpackhi_pd(b01,b01); db[1]=_mm_unpackhi_pd(db01,db01); d2b[1]=_mm_unpackhi_pd(d2b01,d2b01);
  b[2]=_mm_unpacklo_pd(b23,b23); db[2]=_mm_unpacklo_pd(db23,db23); d2b[2]=_mm_unpacklo_pd(d2b23,d2b23);
  b[3]=_mm_unpackhi_pd(b23,b23); db[3]=_mm_unpackhi_pd(db23,db23); d2b[3]=_mm_unpackhi_pd(d2b23,d2b23);
				 				   				  
  c[0]=_mm_unpacklo_pd(c01,c01); dc[0]=_mm_unpacklo_pd(dc01,dc01); d2c[0]=_mm_unpacklo_pd(d2c01,d2c01);
  c[1]=_mm_unpackhi_pd(c01,c01); dc[1]=_mm_unpackhi_pd(dc01,dc01); d2c[1]=_mm_unpackhi_pd(d2c01,d2c01);
  c[2]=_mm_unpacklo_pd(c23,c23); dc[2]=_mm_unpacklo_pd(dc23,dc23); d2c[2]=_mm_unpacklo_pd(d2c23,d2c23);
  c[3]=_mm_unpackhi_pd(c23,c23); dc[3]=_mm_unpackhi_pd(dc23,dc23); d2c[3]=_mm_unpackhi_pd(d2c23,d2c23);

  // Main computation loop
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++) 
      for (int k=0; k<4; k++) {
	__m128d abc, d_abc[3], d2_abc[3];
	
	abc         = _mm_mul_pd (_mm_mul_pd(a[i], b[j]), c[k]);

	d_abc[0]    = _mm_mul_pd (_mm_mul_pd(da[i],  b[j]),  c[k]);
	d_abc[1]    = _mm_mul_pd (_mm_mul_pd( a[i], db[j]),  c[k]);
	d_abc[2]    = _mm_mul_pd (_mm_mul_pd( a[i],  b[j]), dc[k]);

	d2_abc[0]   = _mm_mul_pd (_mm_mul_pd(d2a[i],   b[j]),   c[k]);
	d2_abc[1]   = _mm_mul_pd (_mm_mul_pd(  a[i], d2b[j]),   c[k]);
	d2_abc[2]   = _mm_mul_pd (_mm_mul_pd(  a[i],   b[j]), d2c[k]);
				  

	__m128d* restrict coefs = (__m128d*)(spline->coefs + (ix+i)*xs + (iy+j)*ys + (iz+k)*zs);

	for (int n=0; n<Nh; n++) {
	  mvals[n]      = _mm_add_pd (mvals[n],      _mm_mul_pd (   abc   , coefs[n]));
	  mgrads[3*n+0] = _mm_add_pd (mgrads[3*n+0], _mm_mul_pd ( d_abc[0], coefs[n]));
	  mgrads[3*n+1] = _mm_add_pd (mgrads[3*n+1], _mm_mul_pd ( d_abc[1], coefs[n]));
	  mgrads[3*n+2] = _mm_add_pd (mgrads[3*n+2], _mm_mul_pd ( d_abc[2], coefs[n]));
	  mlapl[3*n+0]  = _mm_add_pd (mlapl[6*n+0],  _mm_mul_pd (d2_abc[0], coefs[n]));
	  mlapl[3*n+1]  = _mm_add_pd (mlapl[6*n+1],  _mm_mul_pd (d2_abc[1], coefs[n]));
	  mlapl[3*n+2]  = _mm_add_pd (mlapl[6*n+2],  _mm_mul_pd (d2_abc[2], coefs[n]));
	}
      }
  
  double dxInv = spline->x_grid.delta_inv;
  double dyInv = spline->y_grid.delta_inv;
  double dzInv = spline->z_grid.delta_inv; 
  
  double lapl3[3*N];
  for (int n=0; n<N/2; n++) {
    _mm_storeu_pd((double*)(vals+2*n),mvals[n]);

    _mm_storel_pd((double*)(grads+6*n+0),mgrads[3*n+0]);
    _mm_storeh_pd((double*)(grads+6*n+3),mgrads[3*n+0]);
    _mm_storel_pd((double*)(grads+6*n+1),mgrads[3*n+1]);
    _mm_storeh_pd((double*)(grads+6*n+4),mgrads[3*n+1]);
    _mm_storel_pd((double*)(grads+6*n+2),mgrads[3*n+2]);
    _mm_storeh_pd((double*)(grads+6*n+5),mgrads[3*n+2]);

    _mm_storel_pd((double*)(lapl3+6*n+0), mlapl [3*n+0]);
    _mm_storeh_pd((double*)(lapl3+6*n+3), mlapl [3*n+0]);
    _mm_storel_pd((double*)(lapl3+6*n+1), mlapl [3*n+1]);
    _mm_storeh_pd((double*)(lapl3+6*n+4), mlapl [3*n+1]);
    _mm_storel_pd((double*)(lapl3+6*n+2), mlapl [3*n+2]);
    _mm_storeh_pd((double*)(lapl3+6*n+5), mlapl [3*n+2]);
  }

  if (N&1) {
    _mm_storel_pd((double*)(vals+N-1),mvals[Nh-1]);

    _mm_storel_pd((double*)(grads+3*(N-1)+0),mgrads[3*(Nh-1)+0]);
    _mm_storel_pd((double*)(grads+3*(N-1)+1),mgrads[3*(Nh-1)+1]);
    _mm_storel_pd((double*)(grads+3*(N-1)+2),mgrads[3*(Nh-1)+2]);

    _mm_storel_pd((double*)(lapl3+3*(N-1)+0),  mlapl [3*(Nh-1)+0]);
    _mm_storel_pd((double*)(lapl3+3*(N-1)+1),  mlapl [3*(Nh-1)+1]);
    _mm_storel_pd((double*)(lapl3+3*(N-1)+2),  mlapl [3*(Nh-1)+2]);
  }

  for (int n=0; n<N; n++) {
    grads[3*n+0] *= dxInv;
    grads[3*n+1] *= dyInv;
    grads[3*n+2] *= dzInv;
    lapl3[3*n+0]  *= dxInv*dxInv;
    lapl3[3*n+1]  *= dyInv*dyInv;
    lapl3[3*n+2]  *= dzInv*dzInv;
    lapl[n] = lapl3[3*n+0] + lapl3[3*n+1] + lapl3[3*n+2];
  }
}


inline void
eval_multi_UBspline_3d_d_vgh (multi_UBspline_3d_d *spline,
			      double x, double y, double z,
			      double* restrict vals,
			      double* restrict grads,
			      double* restrict hess)
{
  _mm_prefetch ((const char*) &A_d[ 0],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[ 1],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[ 2],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[ 3],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[ 4],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[ 5],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[ 6],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[ 7],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[ 8],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[ 9],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[10],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[11],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[12],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[13],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[14],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[15],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[16],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[17],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[18],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[19],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[20],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[21],_MM_HINT_T0);  
  _mm_prefetch ((const char*) &A_d[22],_MM_HINT_T0); _mm_prefetch ((const char*) &A_d[23],_MM_HINT_T0);  

  x -= spline->x_grid.start;
  y -= spline->y_grid.start;  
  z -= spline->z_grid.start;
  double ux = x*spline->x_grid.delta_inv;
  double uy = y*spline->y_grid.delta_inv;
  double uz = z*spline->z_grid.delta_inv;
  ux = fmin (ux, (double)(spline->x_grid.num)-1.0e-5);
  uy = fmin (uy, (double)(spline->y_grid.num)-1.0e-5);
  uz = fmin (uz, (double)(spline->z_grid.num)-1.0e-5);
  double ipartx, iparty, ipartz, tx, ty, tz;
  tx = modf (ux, &ipartx);  int ix = (int) ipartx;
  ty = modf (uy, &iparty);  int iy = (int) iparty;
  tz = modf (uz, &ipartz);  int iz = (int) ipartz;
  
  int xs = spline->x_stride;
  int ys = spline->y_stride;
  int zs = spline->z_stride;
  int N  = spline->num_splines;

  // Now compute the vectors:
  // tpx = [t_x^3 t_x^2 t_x 1]
  // tpy = [t_y^3 t_y^2 t_y 1]
  // tpz = [t_z^3 t_z^2 t_z 1]

  // a  =  A * tpx,   b =  A * tpy,   c =  A * tpz
  // A is 4x4 matrix given by the rows A0, A1, A2, A3
  __m128d tpx01, tpx23, tpy01, tpy23, tpz01, tpz23,
    a01  ,   b01,   c01,   a23,    b23,   c23,  
    da01 ,  db01,  dc01,  da23,   db23,  dc23,  
    d2a01, d2b01, d2c01, d2a23,  d2b23, d2c23;

  tpx01 = _mm_set_pd (tx*tx*tx, tx*tx);
  tpx23 = _mm_set_pd (tx, 1.0);
  tpy01 = _mm_set_pd (ty*ty*ty, ty*ty);
  tpy23 = _mm_set_pd (ty, 1.0);
  tpz01 = _mm_set_pd (tz*tz*tz, tz*tz);
  tpz23 = _mm_set_pd (tz, 1.0);

  // x-dependent vectors
  _MM_DDOT4_PD (A_d[ 0], A_d[ 1], A_d[ 2], A_d[ 3], tpx01, tpx23, tpx01, tpx23,   a01);
  _MM_DDOT4_PD (A_d[ 4], A_d[ 5], A_d[ 6], A_d[ 7], tpx01, tpx23, tpx01, tpx23,   a23);
  _MM_DDOT4_PD (A_d[ 8], A_d[ 9], A_d[10], A_d[11], tpx01, tpx23, tpx01, tpx23,  da01);
  _MM_DDOT4_PD (A_d[12], A_d[13], A_d[14], A_d[15], tpx01, tpx23, tpx01, tpx23,  da23);
  _MM_DDOT4_PD (A_d[16], A_d[17], A_d[18], A_d[19], tpx01, tpx23, tpx01, tpx23, d2a01);
  _MM_DDOT4_PD (A_d[20], A_d[21], A_d[22], A_d[23], tpx01, tpx23, tpx01, tpx23, d2a23);

  // y-dependent vectors
  _MM_DDOT4_PD (A_d[ 0], A_d[ 1], A_d[ 2], A_d[ 3], tpy01, tpy23, tpy01, tpy23,   b01);
  _MM_DDOT4_PD (A_d[ 4], A_d[ 5], A_d[ 6], A_d[ 7], tpy01, tpy23, tpy01, tpy23,   b23);
  _MM_DDOT4_PD (A_d[ 8], A_d[ 9], A_d[10], A_d[11], tpy01, tpy23, tpy01, tpy23,  db01);
  _MM_DDOT4_PD (A_d[12], A_d[13], A_d[14], A_d[15], tpy01, tpy23, tpy01, tpy23,  db23);
  _MM_DDOT4_PD (A_d[16], A_d[17], A_d[18], A_d[19], tpy01, tpy23, tpy01, tpy23, d2b01);
  _MM_DDOT4_PD (A_d[20], A_d[21], A_d[22], A_d[23], tpy01, tpy23, tpy01, tpy23, d2b23);


  // z-dependent vectors
  _MM_DDOT4_PD (A_d[ 0], A_d[ 1], A_d[ 2], A_d[ 3], tpz01, tpz23, tpz01, tpz23,   c01);
  _MM_DDOT4_PD (A_d[ 4], A_d[ 5], A_d[ 6], A_d[ 7], tpz01, tpz23, tpz01, tpz23,   c23);
  _MM_DDOT4_PD (A_d[ 8], A_d[ 9], A_d[10], A_d[11], tpz01, tpz23, tpz01, tpz23,  dc01);
  _MM_DDOT4_PD (A_d[12], A_d[13], A_d[14], A_d[15], tpz01, tpz23, tpz01, tpz23,  dc23);
  _MM_DDOT4_PD (A_d[16], A_d[17], A_d[18], A_d[19], tpz01, tpz23, tpz01, tpz23, d2c01);
  _MM_DDOT4_PD (A_d[20], A_d[21], A_d[22], A_d[23], tpz01, tpz23, tpz01, tpz23, d2c23);


  // Zero-out values
  int Nh = (N+1)/2;
  __m128d mvals[Nh], mgrads[3*Nh], mhess[6*Nh];
  for (int n=0; n<Nh; n++) {
    mvals[n] = _mm_sub_pd (mvals[n], mvals[n]);
    for (int i=0; i<3; i++)
      mgrads[3*n+i] = _mm_sub_pd (mgrads[3*n+i],mgrads[3*n+i]);
    for (int i=0; i<6; i++)
      mhess[6*n+i]  = _mm_sub_pd (mhess[6*n+i], mhess[6*n+i]);
  }

  __m128d a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
  a[0]=_mm_unpacklo_pd(a01,a01); da[0]=_mm_unpacklo_pd(da01,da01); d2a[0]=_mm_unpacklo_pd(d2a01,d2a01);
  a[1]=_mm_unpackhi_pd(a01,a01); da[1]=_mm_unpackhi_pd(da01,da01); d2a[1]=_mm_unpackhi_pd(d2a01,d2a01);
  a[2]=_mm_unpacklo_pd(a23,a23); da[2]=_mm_unpacklo_pd(da23,da23); d2a[2]=_mm_unpacklo_pd(d2a23,d2a23);
  a[3]=_mm_unpackhi_pd(a23,a23); da[3]=_mm_unpackhi_pd(da23,da23); d2a[3]=_mm_unpackhi_pd(d2a23,d2a23);
				 				   				  
  b[0]=_mm_unpacklo_pd(b01,b01); db[0]=_mm_unpacklo_pd(db01,db01); d2b[0]=_mm_unpacklo_pd(d2b01,d2b01);
  b[1]=_mm_unpackhi_pd(b01,b01); db[1]=_mm_unpackhi_pd(db01,db01); d2b[1]=_mm_unpackhi_pd(d2b01,d2b01);
  b[2]=_mm_unpacklo_pd(b23,b23); db[2]=_mm_unpacklo_pd(db23,db23); d2b[2]=_mm_unpacklo_pd(d2b23,d2b23);
  b[3]=_mm_unpackhi_pd(b23,b23); db[3]=_mm_unpackhi_pd(db23,db23); d2b[3]=_mm_unpackhi_pd(d2b23,d2b23);
				 				   				  
  c[0]=_mm_unpacklo_pd(c01,c01); dc[0]=_mm_unpacklo_pd(dc01,dc01); d2c[0]=_mm_unpacklo_pd(d2c01,d2c01);
  c[1]=_mm_unpackhi_pd(c01,c01); dc[1]=_mm_unpackhi_pd(dc01,dc01); d2c[1]=_mm_unpackhi_pd(d2c01,d2c01);
  c[2]=_mm_unpacklo_pd(c23,c23); dc[2]=_mm_unpacklo_pd(dc23,dc23); d2c[2]=_mm_unpacklo_pd(d2c23,d2c23);
  c[3]=_mm_unpackhi_pd(c23,c23); dc[3]=_mm_unpackhi_pd(dc23,dc23); d2c[3]=_mm_unpackhi_pd(d2c23,d2c23);

  // Main computation loop
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++) 
      for (int k=0; k<4; k++) {
	__m128d abc, d_abc[3], d2_abc[6];
	
	abc         = _mm_mul_pd (_mm_mul_pd(a[i], b[j]), c[k]);

	d_abc[0]    = _mm_mul_pd (_mm_mul_pd(da[i],  b[j]),  c[k]);
	d_abc[1]    = _mm_mul_pd (_mm_mul_pd( a[i], db[j]),  c[k]);
	d_abc[2]    = _mm_mul_pd (_mm_mul_pd( a[i],  b[j]), dc[k]);

	d2_abc[0]   = _mm_mul_pd (_mm_mul_pd(d2a[i],   b[j]),   c[k]);
	d2_abc[1]   = _mm_mul_pd (_mm_mul_pd( da[i],  db[j]),   c[k]);
	d2_abc[2]   = _mm_mul_pd (_mm_mul_pd( da[i],   b[j]),  dc[k]);
	d2_abc[3]   = _mm_mul_pd (_mm_mul_pd(  a[i], d2b[j]),   c[k]);
	d2_abc[4]   = _mm_mul_pd (_mm_mul_pd(  a[i],  db[j]),  dc[k]);
	d2_abc[5]   = _mm_mul_pd (_mm_mul_pd(  a[i],   b[j]), d2c[k]);
				  

	__m128d* restrict coefs = (__m128d*)(spline->coefs + (ix+i)*xs + (iy+j)*ys + (iz+k)*zs);

	for (int n=0; n<Nh; n++) {
	  mvals[n]      = _mm_add_pd (mvals[n],      _mm_mul_pd (   abc   , coefs[n]));
	  mgrads[3*n+0] = _mm_add_pd (mgrads[3*n+0], _mm_mul_pd ( d_abc[0], coefs[n]));
	  mgrads[3*n+1] = _mm_add_pd (mgrads[3*n+1], _mm_mul_pd ( d_abc[1], coefs[n]));
	  mgrads[3*n+2] = _mm_add_pd (mgrads[3*n+2], _mm_mul_pd ( d_abc[2], coefs[n]));
	  mhess[6*n+0]  = _mm_add_pd (mhess[6*n+0],  _mm_mul_pd (d2_abc[0], coefs[n]));
	  mhess[6*n+1]  = _mm_add_pd (mhess[6*n+1],  _mm_mul_pd (d2_abc[1], coefs[n]));
	  mhess[6*n+2]  = _mm_add_pd (mhess[6*n+2],  _mm_mul_pd (d2_abc[2], coefs[n]));
	  mhess[6*n+3]  = _mm_add_pd (mhess[6*n+3],  _mm_mul_pd (d2_abc[3], coefs[n]));
	  mhess[6*n+4]  = _mm_add_pd (mhess[6*n+4],  _mm_mul_pd (d2_abc[4], coefs[n]));
	  mhess[6*n+5]  = _mm_add_pd (mhess[6*n+5],  _mm_mul_pd (d2_abc[5], coefs[n]));
	}
      }
  
  double dxInv = spline->x_grid.delta_inv;
  double dyInv = spline->y_grid.delta_inv;
  double dzInv = spline->z_grid.delta_inv; 
  
  for (int n=0; n<N/2; n++) {
    _mm_storeu_pd((double*)(vals+2*n),mvals[n]);

    _mm_storel_pd((double*)(grads+6*n+0),mgrads[3*n+0]);
    _mm_storeh_pd((double*)(grads+6*n+3),mgrads[3*n+0]);
    _mm_storel_pd((double*)(grads+6*n+1),mgrads[3*n+1]);
    _mm_storeh_pd((double*)(grads+6*n+4),mgrads[3*n+1]);
    _mm_storel_pd((double*)(grads+6*n+2),mgrads[3*n+2]);
    _mm_storeh_pd((double*)(grads+6*n+5),mgrads[3*n+2]);

    _mm_storel_pd((double*)(hess+18*n+0),  mhess [6*n+0]);
    _mm_storeh_pd((double*)(hess+18*n+9),  mhess [6*n+0]);
    _mm_storel_pd((double*)(hess+18*n+1),  mhess [6*n+1]);
    _mm_storeh_pd((double*)(hess+18*n+10), mhess [6*n+1]);
    _mm_storel_pd((double*)(hess+18*n+2),  mhess [6*n+2]);
    _mm_storeh_pd((double*)(hess+18*n+11), mhess [6*n+2]);
    _mm_storel_pd((double*)(hess+18*n+4),  mhess [6*n+3]);
    _mm_storeh_pd((double*)(hess+18*n+13), mhess [6*n+3]);
    _mm_storel_pd((double*)(hess+18*n+5),  mhess [6*n+4]);
    _mm_storeh_pd((double*)(hess+18*n+14), mhess [6*n+4]);
    _mm_storel_pd((double*)(hess+18*n+8),  mhess [6*n+5]);
    _mm_storeh_pd((double*)(hess+18*n+17), mhess [6*n+5]);
  }

  if (N&1) {
    _mm_storel_pd((double*)(vals+N-1),mvals[Nh-1]);

    _mm_storel_pd((double*)(grads+3*(N-1)+0),mgrads[3*(Nh-1)+0]);
    _mm_storel_pd((double*)(grads+3*(N-1)+1),mgrads[3*(Nh-1)+1]);
    _mm_storel_pd((double*)(grads+3*(N-1)+2),mgrads[3*(Nh-1)+2]);

    _mm_storel_pd((double*)(hess+9*(N-1)+0),  mhess [6*(Nh-1)+0]);
    _mm_storel_pd((double*)(hess+9*(N-1)+1),  mhess [6*(Nh-1)+1]);
    _mm_storel_pd((double*)(hess+9*(N-1)+2),  mhess [6*(Nh-1)+2]);
    _mm_storel_pd((double*)(hess+9*(N-1)+4),  mhess [6*(Nh-1)+3]);
    _mm_storel_pd((double*)(hess+9*(N-1)+5),  mhess [6*(Nh-1)+4]);
    _mm_storel_pd((double*)(hess+9*(N-1)+8),  mhess [6*(Nh-1)+5]);
  }

  for (int n=0; n<N; n++) {
    grads[3*n+0] *= dxInv;
    grads[3*n+1] *= dyInv;
    grads[3*n+2] *= dzInv;
    hess[9*n+0]  *= dxInv*dxInv;
    hess[9*n+4]  *= dyInv*dyInv;
    hess[9*n+8]  *= dzInv*dzInv;
    hess[9*n+1]  *= dxInv*dyInv;
    hess[9*n+2]  *= dxInv*dzInv;
    hess[9*n+5]  *= dyInv*dzInv;
    // Copy hessian elements into lower half of 3x3 matrix
    hess[9*n+3] = hess[9*n+1];
    hess[9*n+6] = hess[9*n+2];
    hess[9*n+7] = hess[9*n+5];
  }
}


#endif
