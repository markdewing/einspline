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

#ifndef MULTI_BSPLINE_EVAL_STD_Z_H
#define MULTI_BSPLINE_EVAL_STD_Z_H

#include <math.h>
#include <stdio.h>
#include "multi_bspline_structs.h"

extern const double* restrict   Ad;
extern const double* restrict  dAd;
extern const double* restrict d2Ad;

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
eval_multi_UBspline_3d_z (multi_UBspline_3d_z *spline,
			  double x, double y, double z,
			  complex_double* restrict vals)
{
  x -= spline->x_grid.start;
  y -= spline->y_grid.start;
  z -= spline->z_grid.start;
  double ux = x*spline->x_grid.delta_inv;
  double uy = y*spline->y_grid.delta_inv;
  double uz = z*spline->z_grid.delta_inv;
  double ipartx, iparty, ipartz, tx, ty, tz;
  tx = modf (ux, &ipartx);  int ix = (int) ipartx;
  ty = modf (uy, &iparty);  int iy = (int) iparty;
  tz = modf (uz, &ipartz);  int iz = (int) ipartz;
  
  double tpx[4], tpy[4], tpz[4], a[4], b[4], c[4];
  tpx[0] = tx*tx*tx;  tpx[1] = tx*tx;  tpx[2] = tx;  tpx[3] = 1.0;
  tpy[0] = ty*ty*ty;  tpy[1] = ty*ty;  tpy[2] = ty;  tpy[3] = 1.0;
  tpz[0] = tz*tz*tz;  tpz[1] = tz*tz;  tpz[2] = tz;  tpz[3] = 1.0;
  complex_double* restrict coefs = spline->coefs;

  a[0] = (Ad[ 0]*tpx[0] + Ad[ 1]*tpx[1] + Ad[ 2]*tpx[2] + Ad[ 3]*tpx[3]);
  a[1] = (Ad[ 4]*tpx[0] + Ad[ 5]*tpx[1] + Ad[ 6]*tpx[2] + Ad[ 7]*tpx[3]);
  a[2] = (Ad[ 8]*tpx[0] + Ad[ 9]*tpx[1] + Ad[10]*tpx[2] + Ad[11]*tpx[3]);
  a[3] = (Ad[12]*tpx[0] + Ad[13]*tpx[1] + Ad[14]*tpx[2] + Ad[15]*tpx[3]);

  b[0] = (Ad[ 0]*tpy[0] + Ad[ 1]*tpy[1] + Ad[ 2]*tpy[2] + Ad[ 3]*tpy[3]);
  b[1] = (Ad[ 4]*tpy[0] + Ad[ 5]*tpy[1] + Ad[ 6]*tpy[2] + Ad[ 7]*tpy[3]);
  b[2] = (Ad[ 8]*tpy[0] + Ad[ 9]*tpy[1] + Ad[10]*tpy[2] + Ad[11]*tpy[3]);
  b[3] = (Ad[12]*tpy[0] + Ad[13]*tpy[1] + Ad[14]*tpy[2] + Ad[15]*tpy[3]);

  c[0] = (Ad[ 0]*tpz[0] + Ad[ 1]*tpz[1] + Ad[ 2]*tpz[2] + Ad[ 3]*tpz[3]);
  c[1] = (Ad[ 4]*tpz[0] + Ad[ 5]*tpz[1] + Ad[ 6]*tpz[2] + Ad[ 7]*tpz[3]);
  c[2] = (Ad[ 8]*tpz[0] + Ad[ 9]*tpz[1] + Ad[10]*tpz[2] + Ad[11]*tpz[3]);
  c[3] = (Ad[12]*tpz[0] + Ad[13]*tpz[1] + Ad[14]*tpz[2] + Ad[15]*tpz[3]);

  double abc[64];
  abc[ 0]=a[0]*b[0]*c[0]; abc[ 1]=a[0]*b[0]*c[1]; abc[ 2]=a[0]*b[0]*c[2]; abc[ 3]=a[0]*b[0]*c[3];
  abc[ 4]=a[0]*b[1]*c[0]; abc[ 5]=a[0]*b[1]*c[1]; abc[ 6]=a[0]*b[1]*c[2]; abc[ 7]=a[0]*b[1]*c[3];
  abc[ 8]=a[0]*b[2]*c[0]; abc[ 9]=a[0]*b[2]*c[1]; abc[10]=a[0]*b[2]*c[2]; abc[11]=a[0]*b[2]*c[3];
  abc[12]=a[0]*b[3]*c[0]; abc[13]=a[0]*b[3]*c[1]; abc[14]=a[0]*b[3]*c[2]; abc[15]=a[0]*b[3]*c[3];

  abc[16]=a[1]*b[0]*c[0]; abc[17]=a[1]*b[0]*c[1]; abc[18]=a[1]*b[0]*c[2]; abc[19]=a[1]*b[0]*c[3];
  abc[20]=a[1]*b[1]*c[0]; abc[21]=a[1]*b[1]*c[1]; abc[22]=a[1]*b[1]*c[2]; abc[23]=a[1]*b[1]*c[3];
  abc[24]=a[1]*b[2]*c[0]; abc[25]=a[1]*b[2]*c[1]; abc[26]=a[1]*b[2]*c[2]; abc[27]=a[1]*b[2]*c[3];
  abc[28]=a[1]*b[3]*c[0]; abc[29]=a[1]*b[3]*c[1]; abc[30]=a[1]*b[3]*c[2]; abc[31]=a[1]*b[3]*c[3];

  abc[32]=a[2]*b[0]*c[0]; abc[33]=a[2]*b[0]*c[1]; abc[34]=a[2]*b[0]*c[2]; abc[35]=a[2]*b[0]*c[3];
  abc[36]=a[2]*b[1]*c[0]; abc[37]=a[2]*b[1]*c[1]; abc[38]=a[2]*b[1]*c[2]; abc[39]=a[2]*b[1]*c[3];
  abc[40]=a[2]*b[2]*c[0]; abc[41]=a[2]*b[2]*c[1]; abc[42]=a[2]*b[2]*c[2]; abc[43]=a[2]*b[2]*c[3];
  abc[44]=a[2]*b[3]*c[0]; abc[45]=a[2]*b[3]*c[1]; abc[46]=a[2]*b[3]*c[2]; abc[47]=a[2]*b[3]*c[3];

  abc[48]=a[3]*b[0]*c[0]; abc[49]=a[3]*b[0]*c[1]; abc[50]=a[3]*b[0]*c[2]; abc[51]=a[3]*b[0]*c[3];
  abc[52]=a[3]*b[1]*c[0]; abc[53]=a[3]*b[1]*c[1]; abc[54]=a[3]*b[1]*c[2]; abc[55]=a[3]*b[1]*c[3];
  abc[56]=a[3]*b[2]*c[0]; abc[57]=a[3]*b[2]*c[1]; abc[58]=a[3]*b[2]*c[2]; abc[59]=a[3]*b[2]*c[3];
  abc[60]=a[3]*b[3]*c[0]; abc[61]=a[3]*b[3]*c[1]; abc[62]=a[3]*b[3]*c[2]; abc[63]=a[3]*b[3]*c[3];

//   int offsets[64];
//   int index = 0;
  int xs = spline->x_stride;
  int ys = spline->y_stride;
//   for (int i=0; i<4; i++)
//     for (int j=0; j<4; j++)
//       for (int k=0; k<4; k++) {
// 	offsets[index]= (ix+i)*xs +(iy+j)*ys + (iz+k);
// 	index++;
//       }
	
  for (int n=0; n<spline->num_splines; n++) {
    complex_double* restrict coefs = spline->coefs + n*spline->spline_stride;
    vals[n] = 0.0;
    for (int i=0; i<4; i++)
      for (int j=0; j<4; j++)
	for (int k=0; k<4; k++)
	  vals[n] += a[i]*b[j]*c[k]*coefs[(ix+i)*xs + (iy+j)*ys + (iz+k)];
//     for (int j=0; j<64; j++) 
//       vals[i] += coefs[offsets[j]]*abc[j];
  }
}
			  


#endif
