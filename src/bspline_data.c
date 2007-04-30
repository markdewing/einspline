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

/*****************
/*   SSE Data    */
/*****************/

#ifdef __SSE__
// Single-precision version of matrices
#include <xmmintrin.h>
__m128  A0, A1, A2, A3, dA0, dA1, dA2, dA3, d2A0, d2A1, d2A2, d2A3;
#endif

#ifdef __SSE2__
// Double-precision version of matrices
#include <emmintrin.h>
__m128d A0_01, A0_23, A1_01, A1_23, A2_01, A2_23, A3_01, A3_23,
  dA0_01, dA0_23, dA1_01, dA1_23, dA2_01, dA2_23, dA3_01, dA3_23,
  d2A0_01, d2A0_23, d2A1_01, d2A1_23, d2A2_01, d2A2_23, d2A3_01, d2A3_23;
#endif 

#ifdef USE_ALTIVEC
vector float A0   = (vector float) ( -1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0);
vector float A1   = (vector float) (  3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0);
vector float A2   = (vector float) ( -3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0);
vector float A3   = (vector float) (  1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0);
/* vector float A0   = (vector float) ( -1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0); */
/* vector float A1   = (vector float) (  3.0/6.0, -6.0/6.0,  3.0/6.0, 0.0/6.0); */
/* vector float A2   = (vector float) ( -3.0/6.0,  0.0/6.0,  3.0/6.0, 0.0/6.0); */
/* vector float A3   = (vector float) (  1.0/6.0,  4.0/6.0,  1.0/6.0, 0.0/6.0); */
/* vector float A0   = (vector float) ( 1.0/6.0, -3.0/6.0,  3.0/6.0, -1.0/6.0); */
/* vector float A1   = (vector float) ( 4.0/6.0,  0.0/6.0, -6.0/6.0,  3.0/6.0); */
/* vector float A2   = (vector float) ( 1.0/6.0,  3.0/6.0,  3.0/6.0, -3.0/6.0); */
/* vector float A3   = (vector float) ( 0.0/6.0,  0.0/6.0,  0.0/6.0,  1.0/6.0); */
vector float dA0  = (vector float) ( 0.0, -0.5,  1.0, -0.5 );
vector float dA1  = (vector float) ( 0.0,  1.5, -2.0,  0.0 );
vector float dA2  = (vector float) ( 0.0, -1.5,  1.0,  0.5 );
vector float dA3  = (vector float) ( 0.0,  0.5,  0.0,  0.0 );
vector float d2A0 = (vector float) ( 0.0,  0.0, -1.0,  1.0 );
vector float d2A1 = (vector float) ( 0.0,  0.0,  3.0, -2.0 );
vector float d2A2 = (vector float) ( 0.0,  0.0, -3.0,  1.0 );
vector float d2A3 = (vector float) ( 0.0,  0.0,  1.0,  0.0 );
#endif

/*****************/
/* Standard Data */
/*****************/

//////////////////////
// Single precision //
//////////////////////
const float A44f[16] = 
  { -1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0,
     3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0,
    -3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0,
     1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0 };
const float* restrict Af = A44f;

const float dA44f[16] =
  {  0.0, -0.5,  1.0, -0.5,
     0.0,  1.5, -2.0,  0.0,
     0.0, -1.5,  1.0,  0.5,
     0.0,  0.5,  0.0,  0.0 };
const float* restrict dAf = dA44f;

const float d2A44f[16] = 
  {  0.0, 0.0, -1.0,  1.0,
     0.0, 0.0,  3.0, -2.0,
     0.0, 0.0, -3.0,  1.0,
     0.0, 0.0,  1.0,  0.0 };
const float* restrict d2Af = d2A44f;


//////////////////////
// Double precision //
//////////////////////
const double A44d[16] = 
  { -1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0,
     3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0,
    -3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0,
     1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0 };
const double* restrict Ad = A44d;

const double dA44d[16] =
  {  0.0, -0.5,  1.0, -0.5,
     0.0,  1.5, -2.0,  0.0,
     0.0, -1.5,  1.0,  0.5,
     0.0,  0.5,  0.0,  0.0 };
const double* restrict dAd = dA44d;

const double d2A44d[16] = 
  {  0.0, 0.0, -1.0,  1.0,
     0.0, 0.0,  3.0, -2.0,
     0.0, 0.0, -3.0,  1.0,
     0.0, 0.0,  1.0,  0.0 };
const double* restrict d2Ad = d2A44d;

