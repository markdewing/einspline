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

#ifndef NUBBSPLINE_EVAL_SSE_S_H
#define NUBBSPLINE_EVAL_SSE_S_H

#include <math.h>
#include <stdio.h>
#include "nubspline_structs.h"
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#ifdef __SSE3__
#define _MM_MATVEC4_PS(M0, M1, M2, M3, v, r)                        \
do {                                                                \
  __m128 r0 = _mm_hadd_ps (_mm_mul_ps (M0, v), _mm_mul_ps (M1, v)); \
  __m128 r1 = _mm_hadd_ps (_mm_mul_ps (M2, v), _mm_mul_ps (M3, v)); \
  r = _mm_hadd_ps (r0, r1);                                         \
 } while (0);
#define _MM_DOT4_PS(A, B, p)                                        \
do {                                                                \
  __m128 t  = _mm_mul_ps (A, B);                                    \
  __m128 t1 = _mm_hadd_ps (t,t);                                    \
  __m128 r  = _mm_hadd_ps (t1, t1);                                 \
  _mm_store_ss (&(p), r);                                           \
} while(0);
#else
// Use plain-old SSE instructions
#define _MM_MATVEC4_PS(M0, M1, M2, M3, v, r)                        \
do {                                                                \
  __m128 r0 = _mm_mul_ps (M0, v);                                   \
  __m128 r1 = _mm_mul_ps (M1, v);				    \
  __m128 r2 = _mm_mul_ps (M2, v);                                   \
  __m128 r3 = _mm_mul_ps (M3, v);				    \
  _MM_TRANSPOSE4_PS (r0, r1, r2, r3);                               \
  r = _mm_add_ps (_mm_add_ps (r0, r1), _mm_add_ps (r2, r3));        \
 } while (0);
#define _MM_DOT4_PS(A, B, p)                                        \
do {                                                                \
  __m128 t    = _mm_mul_ps (A, B);                                  \
  __m128 alo  = _mm_shuffle_ps (t, t, _MM_SHUFFLE(0,1,0,1));	    \
  __m128 ahi  = _mm_shuffle_ps (t, t, _MM_SHUFFLE(2,3,2,3));	    \
  __m128 a    = _mm_add_ps (alo, ahi);                              \
  __m128 rlo  = _mm_shuffle_ps (a, a, _MM_SHUFFLE(0,0,0,0));	    \
  __m128 rhi  = _mm_shuffle_ps (a, a, _MM_SHUFFLE(1,1,1,1));	    \
  __m128 r    = _mm_add_ps (rlo, rhi);                              \
  _mm_store_ss (&(p), r);                                           \
} while(0);
#endif




/************************************************************/
/* 1D single-precision, real evaulation functions           */
/************************************************************/

/* Value only */
inline void
eval_NUBspline_1d_s (NUBspline_1d_s * restrict spline, 
		     double x, float* restrict val)
{
  float bfuncs[4];
  int i = get_NUBasis_funcs_s (spline->x_basis, x, bfuncs);
  float* restrict coefs = spline->coefs;
  *val = (coefs[i+0]*bfuncs[0] +coefs[i+1]*bfuncs[1] +
	  coefs[i+2]*bfuncs[2] +coefs[i+3]*bfuncs[3]);
}

/* Value and first derivative */
inline void
eval_NUBspline_1d_s_vg (NUBspline_1d_s * restrict spline, double x, 
			float* restrict val, float* restrict grad)
{
  float bfuncs[4], dbfuncs[4];
  int i = get_NUBasis_dfuncs_s (spline->x_basis, x, bfuncs, dbfuncs);
  float* restrict coefs = spline->coefs;
  *val =  (coefs[i+0]* bfuncs[0] + coefs[i+1]* bfuncs[1] +
	   coefs[i+2]* bfuncs[2] + coefs[i+3]* bfuncs[3]);
  *grad = (coefs[i+0]*dbfuncs[0] + coefs[i+1]*dbfuncs[1] +
	   coefs[i+2]*dbfuncs[2] + coefs[i+3]*dbfuncs[3]);
}

/* Value, first derivative, and second derivative */
inline void
eval_NUBspline_1d_s_vgl (NUBspline_1d_s * restrict spline, double x, 
			float* restrict val, float* restrict grad,
			float* restrict lapl)
{
  float bfuncs[4], dbfuncs[4], d2bfuncs[4];
  int i = get_NUBasis_d2funcs_s (spline->x_basis, x, bfuncs, dbfuncs, d2bfuncs);
  float* restrict coefs = spline->coefs;
  *val =  (coefs[i+0]*  bfuncs[0] + coefs[i+1]*  bfuncs[1] +
	   coefs[i+2]*  bfuncs[2] + coefs[i+3]*  bfuncs[3]);
  *grad = (coefs[i+0]* dbfuncs[0] + coefs[i+1]* dbfuncs[1] +
	   coefs[i+2]* dbfuncs[2] + coefs[i+3]* dbfuncs[3]);
  *lapl = (coefs[i+0]*d2bfuncs[0] + coefs[i+1]*d2bfuncs[1] +
	   coefs[i+2]*d2bfuncs[2] + coefs[i+3]*d2bfuncs[3]);

}

inline void
eval_NUBspline_1d_s_vgh (NUBspline_1d_s * restrict spline, double x, 
			float* restrict val, float* restrict grad,
			float* restrict hess)
{
  eval_NUBspline_1d_s_vgl (spline, x, val, grad, hess);
}

/************************************************************/
/* 2D single-precision, real evaulation functions           */
/************************************************************/

/* Value only */
inline void
eval_NUBspline_2d_s (NUBspline_2d_s * restrict spline, 
		    double x, double y, float* restrict val)
{
  float af[4], bf[4];
  __m128 a, b, bP, tmp0, tmp1, tmp2, tmp3;
  int ix = get_NUBasis_funcs_s (spline->x_basis, x, af);
  int iy = get_NUBasis_funcs_s (spline->y_basis, y, bf);

  float* restrict coefs = spline->coefs;
  int xs = spline->x_stride;
  #define P(i) (spline->coefs+(ix+(i))*xs+iy)
  // Prefetch the data from main memory into cache so it's available
  // when we need to use it.
  _mm_prefetch ((void*)P(0), _MM_HINT_T0);
  _mm_prefetch ((void*)P(1), _MM_HINT_T0);
  _mm_prefetch ((void*)P(2), _MM_HINT_T0);
  _mm_prefetch ((void*)P(3), _MM_HINT_T0);

  a = _mm_loadu_ps (af);
  b = _mm_loadu_ps (bf);

  tmp0 = _mm_loadu_ps (P(0));
  tmp1 = _mm_loadu_ps (P(1));
  tmp2 = _mm_loadu_ps (P(2));
  tmp3 = _mm_loadu_ps (P(3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   b,   bP);
  // Compute value
  _MM_DOT4_PS (a, bP, *val);

#undef P
}


/* Value and gradient */
inline void
eval_NUBspline_2d_s_vg (NUBspline_2d_s * restrict spline, 
		       double x, double y, 
		       float* restrict val, float* restrict grad)
{
  float af[4], bf[4], daf[4], dbf[4];
  __m128 a, b, da, db, bP, dbP, tmp0, tmp1, tmp2, tmp3;
  int ix = get_NUBasis_dfuncs_s (spline->x_basis, x, af, daf);
  int iy = get_NUBasis_dfuncs_s (spline->y_basis, y, bf, dbf);
  float* restrict coefs = spline->coefs;
  int xs = spline->x_stride;
#define P(i) (spline->coefs+(ix+(i))*xs+iy)
  
  // Prefetch the data from main memory into cache so it's available
  // when we need to use it.
  _mm_prefetch ((void*)P(0), _MM_HINT_T0);
  _mm_prefetch ((void*)P(1), _MM_HINT_T0);
  _mm_prefetch ((void*)P(2), _MM_HINT_T0);
  _mm_prefetch ((void*)P(3), _MM_HINT_T0);

  a  = _mm_loadu_ps (af);
  b  = _mm_loadu_ps (bf);
  da = _mm_loadu_ps (daf);
  db = _mm_loadu_ps (dbf);

  tmp0 = _mm_loadu_ps (P(0));
  tmp1 = _mm_loadu_ps (P(1));
  tmp2 = _mm_loadu_ps (P(2));
  tmp3 = _mm_loadu_ps (P(3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   b,   bP);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,  db,  dbP);
  // Compute value
  _MM_DOT4_PS (a, bP, *val);
  // Compute gradient
  _MM_DOT4_PS (da, bP, grad[0]);
  _MM_DOT4_PS (a, dbP, grad[1]);
#undef P
}

/* Value, gradient, and laplacian */
inline void
eval_NUBspline_2d_s_vgl (NUBspline_2d_s * restrict spline, 
			double x, double y, float* restrict val, 
			float* restrict grad, float* restrict lapl)
{
  float af[4], bf[4], daf[4], dbf[4], d2af[4], d2bf[4];
  __m128 a, b, da, db, d2a, d2b, bP, dbP, d2bP, tmp0, tmp1, tmp2, tmp3;
  int ix = get_NUBasis_d2funcs_s (spline->x_basis, x, af, daf, d2af);
  int iy = get_NUBasis_d2funcs_s (spline->y_basis, y, bf, dbf, d2af);
  float* restrict coefs = spline->coefs;
  int xs = spline->x_stride;
#define P(i) (spline->coefs+(ix+(i))*xs+iy)
  
  // Prefetch the data from main memory into cache so it's available
  // when we need to use it.
  _mm_prefetch ((void*)P(0), _MM_HINT_T0);
  _mm_prefetch ((void*)P(1), _MM_HINT_T0);
  _mm_prefetch ((void*)P(2), _MM_HINT_T0);
  _mm_prefetch ((void*)P(3), _MM_HINT_T0);

  a   = _mm_loadu_ps (af);
  b   = _mm_loadu_ps (bf);
  da  = _mm_loadu_ps (daf);
  db  = _mm_loadu_ps (dbf);
  d2a = _mm_loadu_ps (d2af);
  d2b = _mm_loadu_ps (d2bf);

  tmp0 = _mm_loadu_ps (P(0));
  tmp1 = _mm_loadu_ps (P(1));
  tmp2 = _mm_loadu_ps (P(2));
  tmp3 = _mm_loadu_ps (P(3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   b,   bP);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,  db,  dbP);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3, d2b, d2bP);
  // Compute value
  _MM_DOT4_PS (a, bP, *val);
  // Compute gradient
  _MM_DOT4_PS (da, bP, grad[0]);
  _MM_DOT4_PS (a, dbP, grad[1]);
  float sec_derivs[2];
  // Compute laplacian
  _MM_DOT4_PS (d2a, bP, sec_derivs[0]);
  _MM_DOT4_PS (a, d2bP, sec_derivs[1]);
  *lapl = sec_derivs[0] + sec_derivs[1];
#undef P
}

/* Value, gradient, and Hessian */
inline void
eval_NUBspline_2d_s_vgh (NUBspline_2d_s * restrict spline, 
			double x, double y, float* restrict val, 
			float* restrict grad, float* restrict hess)
{
  float af[4], bf[4], daf[4], dbf[4], d2af[4], d2bf[4];
  __m128 a, b, da, db, d2a, d2b, bP, dbP, d2bP, tmp0, tmp1, tmp2, tmp3;
  int ix = get_NUBasis_d2funcs_s (spline->x_basis, x, af, daf, d2af);
  int iy = get_NUBasis_d2funcs_s (spline->y_basis, y, bf, dbf, d2af);
  float* restrict coefs = spline->coefs;
  int xs = spline->x_stride;
#define P(i) (spline->coefs+(ix+(i))*xs+iy)
  
  // Prefetch the data from main memory into cache so it's available
  // when we need to use it.
  _mm_prefetch ((void*)P(0), _MM_HINT_T0);
  _mm_prefetch ((void*)P(1), _MM_HINT_T0);
  _mm_prefetch ((void*)P(2), _MM_HINT_T0);
  _mm_prefetch ((void*)P(3), _MM_HINT_T0);

  a   = _mm_loadu_ps (af);
  b   = _mm_loadu_ps (bf);
  da  = _mm_loadu_ps (daf);
  db  = _mm_loadu_ps (dbf);
  d2a = _mm_loadu_ps (d2af);
  d2b = _mm_loadu_ps (d2bf);

  tmp0 = _mm_loadu_ps (P(0));
  tmp1 = _mm_loadu_ps (P(1));
  tmp2 = _mm_loadu_ps (P(2));
  tmp3 = _mm_loadu_ps (P(3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   b,   bP);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,  db,  dbP);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3, d2b, d2bP);
  // Compute value
  _MM_DOT4_PS (a, bP, *val);
  // Compute gradient
  _MM_DOT4_PS (da, bP, grad[0]);
  _MM_DOT4_PS (a, dbP, grad[1]);
  float sec_derivs[2];
  // Compute hessian
  // Compute hessian
  _MM_DOT4_PS (d2a, bP, hess[0]);
  _MM_DOT4_PS (a, d2bP, hess[3]);
  _MM_DOT4_PS (da, dbP, hess[1]);
  // Copy hessian element into lower half of 2x2 matrix
  hess[2] = hess[1];
#undef P
}


/************************************************************/
/* 3D single-precision, real evaulation functions           */
/************************************************************/

/* Value only */
inline void
eval_NUBspline_3d_s (NUBspline_3d_s * restrict spline, 
		    double x, double y, double z,
		    float* restrict val)
{
  float af[4], bf[4], cf[4];
  
  __m128 a, b, c, cP[4], bcP, tmp0, tmp1, tmp2, tmp3;
  int ix = get_NUBasis_funcs_s (spline->x_basis, x, af);
  int iy = get_NUBasis_funcs_s (spline->y_basis, y, bf);
  int iz = get_NUBasis_funcs_s (spline->z_basis, z, cf);
  
  int xs = spline->x_stride;
  int ys = spline->y_stride;
  float* restrict coefs = spline->coefs;
#define P(i,j) (coefs+(ix+(i))*xs+(iy+(j))*ys+(iz))
  _mm_prefetch ((void*)P(0,0), _MM_HINT_T0);
  _mm_prefetch ((void*)P(0,1), _MM_HINT_T0);
  _mm_prefetch ((void*)P(0,2), _MM_HINT_T0);
  _mm_prefetch ((void*)P(0,3), _MM_HINT_T0);
  _mm_prefetch ((void*)P(1,0), _MM_HINT_T0);
  _mm_prefetch ((void*)P(1,1), _MM_HINT_T0);
  _mm_prefetch ((void*)P(1,2), _MM_HINT_T0);
  _mm_prefetch ((void*)P(1,3), _MM_HINT_T0);
  _mm_prefetch ((void*)P(2,0), _MM_HINT_T0);
  _mm_prefetch ((void*)P(2,1), _MM_HINT_T0);
  _mm_prefetch ((void*)P(2,2), _MM_HINT_T0);
  _mm_prefetch ((void*)P(2,3), _MM_HINT_T0);
  _mm_prefetch ((void*)P(3,0), _MM_HINT_T0);
  _mm_prefetch ((void*)P(3,1), _MM_HINT_T0);
  _mm_prefetch ((void*)P(3,2), _MM_HINT_T0);
  _mm_prefetch ((void*)P(3,3), _MM_HINT_T0);
  
  a   = _mm_loadu_ps (af);
  b   = _mm_loadu_ps (bf);
  c   = _mm_loadu_ps (cf);
  
  // Compute cP, dcP, and d2cP products 1/4 at a time to maximize
  // register reuse and avoid rerereading from memory or cache.
  // 1st quarter
  tmp0 = _mm_loadu_ps (P(0,0));
  tmp1 = _mm_loadu_ps (P(0,1));
  tmp2 = _mm_loadu_ps (P(0,2));
  tmp3 = _mm_loadu_ps (P(0,3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[0]);
  // 2nd quarter
  tmp0 = _mm_loadu_ps (P(1,0));
  tmp1 = _mm_loadu_ps (P(1,1));
  tmp2 = _mm_loadu_ps (P(1,2));
  tmp3 = _mm_loadu_ps (P(1,3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[1]);
  // 3rd quarter
  tmp0 = _mm_loadu_ps (P(2,0));
  tmp1 = _mm_loadu_ps (P(2,1));
  tmp2 = _mm_loadu_ps (P(2,2));
  tmp3 = _mm_loadu_ps (P(2,3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[2]);
  // 4th quarter
  tmp0 = _mm_loadu_ps (P(3,0));
  tmp1 = _mm_loadu_ps (P(3,1));
  tmp2 = _mm_loadu_ps (P(3,2));
  tmp3 = _mm_loadu_ps (P(3,3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[3]);
  
  // Now compute bcP, dbcP, bdcP, d2bcP, bd2cP, and dbdc products
  _MM_MATVEC4_PS (  cP[0],   cP[1],   cP[2],   cP[3],   b,   bcP);
  // Compute value
  _MM_DOT4_PS (a, bcP, *val);

#undef P
}

/* Value and gradient */
inline void
eval_NUBspline_3d_s_vg (NUBspline_3d_s * restrict spline, 
			double x, double y, double z,
			float* restrict val, float* restrict grad)
{
  float af[4], bf[4], cf[4], daf[4], dbf[4], dcf[4];
  
  __m128 a, b, c, da, db, dc, cP[4], dcP[4], bcP, dbcP, 
    dbP, bdcP, tmp0, tmp1, tmp2, tmp3;
  int ix = get_NUBasis_dfuncs_s (spline->x_basis, x, af, daf);
  int iy = get_NUBasis_dfuncs_s (spline->y_basis, y, bf, dbf);
  int iz = get_NUBasis_dfuncs_s (spline->z_basis, z, cf, dcf);
  
  int xs = spline->x_stride;
  int ys = spline->y_stride;
  float* restrict coefs = spline->coefs;
#define P(i,j) (coefs+(ix+(i))*xs+(iy+(j))*ys+(iz))
  _mm_prefetch ((void*)P(0,0), _MM_HINT_T0);
  _mm_prefetch ((void*)P(0,1), _MM_HINT_T0);
  _mm_prefetch ((void*)P(0,2), _MM_HINT_T0);
  _mm_prefetch ((void*)P(0,3), _MM_HINT_T0);
  _mm_prefetch ((void*)P(1,0), _MM_HINT_T0);
  _mm_prefetch ((void*)P(1,1), _MM_HINT_T0);
  _mm_prefetch ((void*)P(1,2), _MM_HINT_T0);
  _mm_prefetch ((void*)P(1,3), _MM_HINT_T0);
  _mm_prefetch ((void*)P(2,0), _MM_HINT_T0);
  _mm_prefetch ((void*)P(2,1), _MM_HINT_T0);
  _mm_prefetch ((void*)P(2,2), _MM_HINT_T0);
  _mm_prefetch ((void*)P(2,3), _MM_HINT_T0);
  _mm_prefetch ((void*)P(3,0), _MM_HINT_T0);
  _mm_prefetch ((void*)P(3,1), _MM_HINT_T0);
  _mm_prefetch ((void*)P(3,2), _MM_HINT_T0);
  _mm_prefetch ((void*)P(3,3), _MM_HINT_T0);
  
  a   = _mm_loadu_ps (af);
  b   = _mm_loadu_ps (bf);
  c   = _mm_loadu_ps (cf);
  da  = _mm_loadu_ps (daf);
  db  = _mm_loadu_ps (dbf);
  dc  = _mm_loadu_ps (dcf);
  
  // Compute cP, dcP, and d2cP products 1/4 at a time to maximize
  // register reuse and avoid rerereading from memory or cache.
  // 1st quarter
  tmp0 = _mm_loadu_ps (P(0,0));
  tmp1 = _mm_loadu_ps (P(0,1));
  tmp2 = _mm_loadu_ps (P(0,2));
  tmp3 = _mm_loadu_ps (P(0,3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[0]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,  dc,  dcP[0]);
  // 2nd quarter
  tmp0 = _mm_loadu_ps (P(1,0));
  tmp1 = _mm_loadu_ps (P(1,1));
  tmp2 = _mm_loadu_ps (P(1,2));
  tmp3 = _mm_loadu_ps (P(1,3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[1]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,  dc,  dcP[1]);
  // 3rd quarter
  tmp0 = _mm_loadu_ps (P(2,0));
  tmp1 = _mm_loadu_ps (P(2,1));
  tmp2 = _mm_loadu_ps (P(2,2));
  tmp3 = _mm_loadu_ps (P(2,3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[2]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,  dc,  dcP[2]);
  // 4th quarter
  tmp0 = _mm_loadu_ps (P(3,0));
  tmp1 = _mm_loadu_ps (P(3,1));
  tmp2 = _mm_loadu_ps (P(3,2));
  tmp3 = _mm_loadu_ps (P(3,3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[3]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,  dc,  dcP[3]);
  
  // Now compute bcP, dbcP, bdcP, d2bcP, bd2cP, and dbdc products
  _MM_MATVEC4_PS (  cP[0],   cP[1],   cP[2],   cP[3],   b,   bcP);
  _MM_MATVEC4_PS (  cP[0],   cP[1],   cP[2],   cP[3],  db,  dbcP);
  _MM_MATVEC4_PS ( dcP[0],  dcP[1],  dcP[2],  dcP[3],   b,  bdcP);
  // Compute value
  _MM_DOT4_PS (a, bcP, *val);
  // Compute gradient
  _MM_DOT4_PS (da, bcP, grad[0]);
  _MM_DOT4_PS (a, dbcP, grad[1]);
  _MM_DOT4_PS (a, bdcP, grad[2]);

#undef P
}



/* Value, gradient, and laplacian */
inline void
eval_NUBspline_3d_s_vgl (NUBspline_3d_s * restrict spline, 
			 double x, double y, double z,
			 float* restrict val, float* restrict grad, float* restrict lapl)
{
  float af[4], bf[4], cf[4], daf[4], dbf[4], dcf[4], d2af[4], d2bf[4], d2cf[4]; 
  
  __m128 a, b, c, da, db, dc, d2a, d2b, d2c, cP[4], dcP[4], d2cP[4], bcP, dbcP,
    d2bcP, dbdcP, bd2cP, bdcP, tmp0, tmp1, tmp2, tmp3;
  int ix = get_NUBasis_d2funcs_s (spline->x_basis, x, af, daf, d2af);
  int iy = get_NUBasis_d2funcs_s (spline->y_basis, y, bf, dbf, d2bf);
  int iz = get_NUBasis_d2funcs_s (spline->z_basis, z, cf, dcf, d2cf);
  
  int xs = spline->x_stride;
  int ys = spline->y_stride;
  float* restrict coefs = spline->coefs;
#define P(i,j) (coefs+(ix+(i))*xs+(iy+(j))*ys+(iz))
  _mm_prefetch ((void*)P(0,0), _MM_HINT_T0);
  _mm_prefetch ((void*)P(0,1), _MM_HINT_T0);
  _mm_prefetch ((void*)P(0,2), _MM_HINT_T0);
  _mm_prefetch ((void*)P(0,3), _MM_HINT_T0);
  _mm_prefetch ((void*)P(1,0), _MM_HINT_T0);
  _mm_prefetch ((void*)P(1,1), _MM_HINT_T0);
  _mm_prefetch ((void*)P(1,2), _MM_HINT_T0);
  _mm_prefetch ((void*)P(1,3), _MM_HINT_T0);
  _mm_prefetch ((void*)P(2,0), _MM_HINT_T0);
  _mm_prefetch ((void*)P(2,1), _MM_HINT_T0);
  _mm_prefetch ((void*)P(2,2), _MM_HINT_T0);
  _mm_prefetch ((void*)P(2,3), _MM_HINT_T0);
  _mm_prefetch ((void*)P(3,0), _MM_HINT_T0);
  _mm_prefetch ((void*)P(3,1), _MM_HINT_T0);
  _mm_prefetch ((void*)P(3,2), _MM_HINT_T0);
  _mm_prefetch ((void*)P(3,3), _MM_HINT_T0);
  
  a   = _mm_loadu_ps (af);
  b   = _mm_loadu_ps (bf);
  c   = _mm_loadu_ps (cf);
  da  = _mm_loadu_ps (daf);
  db  = _mm_loadu_ps (dbf);
  dc  = _mm_loadu_ps (dcf);
  d2a = _mm_loadu_ps (d2af);
  d2b = _mm_loadu_ps (d2bf);
  d2c = _mm_loadu_ps (d2cf);
  
  // Compute cP, dcP, and d2cP products 1/4 at a time to maximize
  // register reuse and avoid rerereading from memory or cache.
  // 1st quarter
  tmp0 = _mm_loadu_ps (P(0,0));
  tmp1 = _mm_loadu_ps (P(0,1));
  tmp2 = _mm_loadu_ps (P(0,2));
  tmp3 = _mm_loadu_ps (P(0,3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[0]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,  dc,  dcP[0]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3, d2c, d2cP[0]);
  // 2nd quarter
  tmp0 = _mm_loadu_ps (P(1,0));
  tmp1 = _mm_loadu_ps (P(1,1));
  tmp2 = _mm_loadu_ps (P(1,2));
  tmp3 = _mm_loadu_ps (P(1,3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[1]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,  dc,  dcP[1]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3, d2c, d2cP[1]);
  // 3rd quarter
  tmp0 = _mm_loadu_ps (P(2,0));
  tmp1 = _mm_loadu_ps (P(2,1));
  tmp2 = _mm_loadu_ps (P(2,2));
  tmp3 = _mm_loadu_ps (P(2,3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[2]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,  dc,  dcP[2]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3, d2c, d2cP[2]);
  // 4th quarter
  tmp0 = _mm_loadu_ps (P(3,0));
  tmp1 = _mm_loadu_ps (P(3,1));
  tmp2 = _mm_loadu_ps (P(3,2));
  tmp3 = _mm_loadu_ps (P(3,3));
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[3]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,  dc,  dcP[3]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3, d2c, d2cP[3]);
  
  // Now compute bcP, dbcP, bdcP, d2bcP, bd2cP, and dbdc products
  _MM_MATVEC4_PS (  cP[0],   cP[1],   cP[2],   cP[3],   b,   bcP);
  _MM_MATVEC4_PS (  cP[0],   cP[1],   cP[2],   cP[3],  db,  dbcP);
  _MM_MATVEC4_PS ( dcP[0],  dcP[1],  dcP[2],  dcP[3],   b,  bdcP);
  _MM_MATVEC4_PS (  cP[0],   cP[1],   cP[2],   cP[3], d2b, d2bcP);
  _MM_MATVEC4_PS (d2cP[0], d2cP[1], d2cP[2], d2cP[3],   b, bd2cP);
  _MM_MATVEC4_PS ( dcP[0],  dcP[1],  dcP[2],  dcP[3],  db, dbdcP);
  // Compute value
  _MM_DOT4_PS (a, bcP, *val);
  // Compute gradient
  _MM_DOT4_PS (da, bcP, grad[0]);
  _MM_DOT4_PS (a, dbcP, grad[1]);
  _MM_DOT4_PS (a, bdcP, grad[2]);
  // Compute laplacian
  float sec_derivs[3];

  _MM_DOT4_PS (d2a, bcP, sec_derivs[0]);
  _MM_DOT4_PS (a, d2bcP, sec_derivs[1]);
  _MM_DOT4_PS (a, bd2cP, sec_derivs[2]);
  *lapl = sec_derivs[0] + sec_derivs[1] + sec_derivs[2];
#undef P
}



typedef union { float scalars[4]; __m128 v; } vec4;

/* Value, gradient, and Hessian */
inline void
eval_NUBspline_3d_s_vgh (NUBspline_3d_s * restrict spline, 
			 double x, double y, double z,
			 float* restrict val, float* restrict grad, float* restrict hess)
{
  __m128 a, b, c, da, db, dc, d2a, d2b, d2c, cP[4], dcP[4], d2cP[4], bcP, dbcP,
    d2bcP, dbdcP, bd2cP, bdcP, tmp0, tmp1, tmp2, tmp3;
//  float af[4], bf[4], cf[4], daf[4], dbf[4], dcf[4], d2af[4], d2bf[4], d2cf[4]; 
//    vec4 av, bv, cv, dav, dbv, dcv, d2av, d2bv, d2cv;
//   int ix = get_NUBasis_d2funcs_s (spline->x_basis, x, av.scalars, dav.scalars, d2av.scalars);
//   int iy = get_NUBasis_d2funcs_s (spline->y_basis, y, bv.scalars, dbv.scalars, d2bv.scalars);
//   int iz = get_NUBasis_d2funcs_s (spline->z_basis, z, cv.scalars, dcv.scalars, d2cv.scalars);
//   a   =   av.v;   b =   bv.v;   c =   cv.v;
//   da  =  dav.v;  db =  dbv.v;  dc =  dcv.v;
//   d2a = d2av.v; d2b = d2bv.v; d2c = d2cv.v;
//   int ix = get_NUBasis_d2funcs_s (spline->x_basis, x, af, daf, d2af);
//   int iy = get_NUBasis_d2funcs_s (spline->y_basis, y, bf, dbf, d2bf);
//   int iz = get_NUBasis_d2funcs_s (spline->z_basis, z, cf, dcf, d2cf);
//   a = _mm_loadu_ps (af);  da = _mm_loadu_ps (daf);  d2a = _mm_loadu_ps (d2af);
//   b = _mm_loadu_ps (bf);  db = _mm_loadu_ps (dbf);  d2b = _mm_loadu_ps (d2bf);
//   c = _mm_loadu_ps (cf);  dc = _mm_loadu_ps (dcf);  d2c = _mm_loadu_ps (d2cf);
  int ix = get_NUBasis_d2funcs_sse_s (spline->x_basis, x, &a, &da, &d2a);
  int iy = get_NUBasis_d2funcs_sse_s (spline->y_basis, y, &b, &db, &d2b);
  int iz = get_NUBasis_d2funcs_sse_s (spline->z_basis, z, &c, &dc, &d2c);

  int xs = spline->x_stride;
  int ys = spline->y_stride;
  int ys2 = 2*ys;
  int ys3 = 3*ys;
  float* restrict coefs = spline->coefs;
#define P(i,j) (coefs+(ix+(i))*xs+(iy+(j))*ys+(iz))
  float *restrict p = P(0,0);
  _mm_prefetch ((char*)(p    ), _MM_HINT_T0);
  _mm_prefetch ((char*)(p+ys ), _MM_HINT_T0);
  _mm_prefetch ((char*)(p+ys2), _MM_HINT_T0);
  _mm_prefetch ((char*)(p+ys3), _MM_HINT_T0);
  p+= xs;
  _mm_prefetch ((char*)(p    ), _MM_HINT_T0);
  _mm_prefetch ((char*)(p+ys ), _MM_HINT_T0);
  _mm_prefetch ((char*)(p+ys2), _MM_HINT_T0);
  _mm_prefetch ((char*)(p+ys3), _MM_HINT_T0);
  p+= xs;
  _mm_prefetch ((char*)(p    ), _MM_HINT_T0);
  _mm_prefetch ((char*)(p+ys ), _MM_HINT_T0);
  _mm_prefetch ((char*)(p+ys2), _MM_HINT_T0);
  _mm_prefetch ((char*)(p+ys3), _MM_HINT_T0);
  p+= xs;
  _mm_prefetch ((char*)(p    ), _MM_HINT_T0);
  _mm_prefetch ((char*)(p+ys ), _MM_HINT_T0);
  _mm_prefetch ((char*)(p+ys2), _MM_HINT_T0);
  _mm_prefetch ((char*)(p+ys3), _MM_HINT_T0);

  // Compute cP, dcP, and d2cP products 1/4 at a time to maximize
  // register reuse and avoid rerereading from memory or cache.
  // 1st quarter
  p = P(0,0);
  tmp0 = _mm_loadu_ps (p    );
  tmp1 = _mm_loadu_ps (p+ys );
  tmp2 = _mm_loadu_ps (p+ys2);
  tmp3 = _mm_loadu_ps (p+ys3);
  p += xs;
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[0]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,  dc,  dcP[0]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3, d2c, d2cP[0]);
  // 2nd quarter
  tmp0 = _mm_loadu_ps (p    );
  tmp1 = _mm_loadu_ps (p+ys );
  tmp2 = _mm_loadu_ps (p+ys2);
  tmp3 = _mm_loadu_ps (p+ys3);
  p += xs;
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[1]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,  dc,  dcP[1]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3, d2c, d2cP[1]);
  // 3rd quarter
  tmp0 = _mm_loadu_ps (p    );
  tmp1 = _mm_loadu_ps (p+ys );
  tmp2 = _mm_loadu_ps (p+ys2);
  tmp3 = _mm_loadu_ps (p+ys3);
  p += xs;
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[2]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,  dc,  dcP[2]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3, d2c, d2cP[2]);
  // 4th quarter
  tmp0 = _mm_loadu_ps (p    );
  tmp1 = _mm_loadu_ps (p+ys );
  tmp2 = _mm_loadu_ps (p+ys2);
  tmp3 = _mm_loadu_ps (p+ys3);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,   c,   cP[3]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3,  dc,  dcP[3]);
  _MM_MATVEC4_PS (tmp0, tmp1, tmp2, tmp3, d2c, d2cP[3]);
  
  // Now compute bcP, dbcP, bdcP, d2bcP, bd2cP, and dbdc products
  _MM_MATVEC4_PS (  cP[0],   cP[1],   cP[2],   cP[3],   b,   bcP);
  _MM_MATVEC4_PS (  cP[0],   cP[1],   cP[2],   cP[3],  db,  dbcP);
  _MM_MATVEC4_PS ( dcP[0],  dcP[1],  dcP[2],  dcP[3],   b,  bdcP);
  _MM_MATVEC4_PS (  cP[0],   cP[1],   cP[2],   cP[3], d2b, d2bcP);
  _MM_MATVEC4_PS (d2cP[0], d2cP[1], d2cP[2], d2cP[3],   b, bd2cP);
  _MM_MATVEC4_PS ( dcP[0],  dcP[1],  dcP[2],  dcP[3],  db, dbdcP);
  // Compute value
  _MM_DOT4_PS (a, bcP, *val);
  // Compute gradient
  _MM_DOT4_PS (da, bcP, grad[0]);
  _MM_DOT4_PS (a, dbcP, grad[1]);
  _MM_DOT4_PS (a, bdcP, grad[2]);
  // Compute hessian
  _MM_DOT4_PS (d2a, bcP, hess[0]);
  _MM_DOT4_PS (a, d2bcP, hess[4]);
  _MM_DOT4_PS (a, bd2cP, hess[8]);
  _MM_DOT4_PS (da, dbcP, hess[1]);
  _MM_DOT4_PS (da, bdcP, hess[2]);
  _MM_DOT4_PS (a, dbdcP, hess[5]);

  // Copy hessian elements into lower half of 3x3 matrix
  hess[3] = hess[1];
  hess[6] = hess[2];
  hess[7] = hess[5];

#undef P
}

#undef _MM_MATVEC4_PS
#undef _MM_DOT4_PS

#endif
