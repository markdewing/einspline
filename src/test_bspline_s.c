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

#include "bspline.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433
#endif

typedef struct
{
  double kcut;
  double *Gvecs;
  float *coefs;
  int numG;
} periodic_func_s;

void
int_periodic_func (periodic_func_s *func, double kcut)
{
  func->kcut = kcut;
  func->numG = 0;
  int imax = (int) ceil (kcut/(2.0*M_PI));
  for (int ix=-imax; ix<=imax; ix++) {
    double kx = 2.0*M_PI * ix;
    for (int iy=-imax; iy<=imax; iy++) {
      double ky = 2.0*M_PI * iy;
      for (int iz=-imax; iz<=imax; iz++) {
	double kz = 2.0*M_PI * iz;
	if ((kx*kx + ky*ky + kz*kz) < (kcut*kcut))
	  func->numG++;
      }
    }
  }
  func->Gvecs = (double*) malloc (3*sizeof(double)*func->numG);
  func->coefs = (float*)  malloc (2*sizeof(float) *func->numG);

  int iG = 0;
  for (int ix=-imax; ix<=imax; ix++) {
    double kx = 2.0*M_PI * ix;
    for (int iy=-imax; iy<=imax; iy++) {
      double ky = 2.0*M_PI * iy;
      for (int iz=-imax; iz<=imax; iz++) {
	double kz = 2.0*M_PI * iz;
	if ((kx*kx + ky*ky + kz*kz) < (kcut*kcut)) {
	  func->Gvecs[3*iG+0] = kx;
	  func->Gvecs[3*iG+1] = ky;
	  func->Gvecs[3*iG+2] = kz;
	  func->coefs[2*iG+0] = 2.0*(drand48()-0.5);
	  func->coefs[2*iG+1] = 2.0*(drand48()-0.5);
	  iG++;
	}
      }
    }
  }
}

void
eval_periodic_func_s (periodic_func_s* restrict func,
		      double x, double y, double z,
		      float *restrict val, float *restrict grad,
		      float *restrict hess)
{
  *val = 0.0;
  for (int i=0; i<3; i++)    grad[i] = 0.0;
  for (int i=0; i<9; i++)    hess[i] = 0.0;

  for (int iG=0; iG<func->numG; iG++) {
    double kx = func->Gvecs[3*iG+0];
    double ky = func->Gvecs[3*iG+1];
    double kz = func->Gvecs[3*iG+2];
    double phase = x*kx + y+ky + z*kz;
    double re, im;
    sincos(phase, &im, &re);
    double c_re = func->coefs[2*iG+0];
    double c_im = func->coefs[2*iG+1];
    *val    += re*c_re - im*c_im;
    grad[0] += -kx*(re*c_im + im*c_re);
    grad[1] += -ky*(re*c_im + im*c_re);
    grad[2] += -kz*(re*c_im + im*c_re);
    hess[0] += -kx*kx*(re*c_re - im*c_im);
    hess[1] += -kx*ky*(re*c_re - im*c_im);
    hess[2] += -kx*kz*(re*c_re - im*c_im);
    hess[3] += -ky*kx*(re*c_re - im*c_im);
    hess[4] += -ky*ky*(re*c_re - im*c_im);
    hess[5] += -ky*kz*(re*c_re - im*c_im);
    hess[6] += -kz*kx*(re*c_re - im*c_im);
    hess[7] += -kz*ky*(re*c_re - im*c_im);
    hess[8] += -kz*kz*(re*c_re - im*c_im);
  }
}


void
test_bspline_3d_s()
{
  double kcut = 2.0*M_PI * 5.0;
  int Nspline = 15;
  Ugrid x_grid, y_grid, z_grid;
  x_grid.start = 0.0; x_grid.end = 1.0; x_grid.num = Nspline;
  y_grid.start = 0.0; y_grid.end = 1.0; y_grid.num = Nspline;
  z_grid.start = 0.0; z_grid.end = 1.0; z_grid.num = Nspline;
  BCtype_s xBC, yBC, zBC;
  xBC.lCode = xBC.rCode = PERIODIC;
  yBC.lCode = yBC.rCode = PERIODIC;
  zBC.lCode = zBC.rCode = PERIODIC;

  float *data = malloc (sizeof(float)*Nspline*Nspline*Nspline);
  periodic_func_s func;


  UBspline_3d_s *spline =
    create_UBspline_3d_s (x_grid, y_grid, z_grid, xBC, yBC, zBC,
			  data);
  

  

}

main()
{
  test_bspline_3d_s();
}
