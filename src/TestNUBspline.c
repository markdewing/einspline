#include "nubspline_create.h"
#include "nubspline_eval_std_s.h"
#include "nubspline_eval_std_d.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void
PrintPassFail(bool pass)
{
  if (pass)
    // Print green "Passed"
    fprintf (stderr, "%c[32mPassed%c[0m\n", 0x1B, 0x1B);
  else
    // Print red "Failed"
    fprintf (stderr, "%c[31mFailed%c[0m\n", 0x1B, 0x1B);
}


void
TestCenterGrid()
{
  fprintf (stderr, "Testing CenterGrid:   ");
  bool passed = true;
  NUgrid* grid = create_center_grid (-5.0, 7.0, 6.0, 200);

/*   for (int i=0; i<200; i++) */
/*     fprintf (stderr, "%16.12f\n", grid->points[i]); */
  for (int i=0; i<10000; i++) {
    double x = -5.0+12.0*drand48();
    int lo = (*grid->reverse_map)(grid, x);
    assert (x >= grid->points[lo]);
    assert (x <= grid->points[lo+1]);
  }
  PrintPassFail (passed);
}


void
TestGeneralGrid()
{
  fprintf (stderr, "Testing GeneralGrid:  ");
  bool passed = true;
  NUgrid* centgrid = create_center_grid (-5.0, 7.0, 6.0, 200);
  NUgrid* grid = create_general_grid (centgrid->points, 200);
/*   for (int i=0; i<200; i++) */
/*     fprintf (stderr, "%16.12f\n", grid->points[i]); */
  for (int i=0; i<10000; i++) {
    double x = -5.0+12.0*drand48();
    int lo = (*grid->reverse_map)(grid, x);
    passed = passed && (x >= grid->points[lo]);
    passed = passed && (x <= grid->points[lo+1]);
  }
  PrintPassFail (passed);
}

void
GridSpeedTest()
{
  NUgrid* centgrid = create_center_grid (-5.0, 7.0, 6.0, 2000);
  NUgrid* gengrid = create_general_grid (centgrid->points, 2000);
  int centsum=0, gensum=0;
  
  clock_t rstart, rend, cstart, cend, gstart, gend;
  
  rstart = clock();
  for (int i=0; i<100000000; i++) {
    double x = -5.0 + 12.0*drand48();
  }
  rend = clock();

  cstart = clock();
  for (int i=0; i<100000000; i++) {
    double x = -5.0 + 12.0*drand48();
    centsum += (*centgrid->reverse_map)(centgrid, x);
  }
  cend = clock();

  gstart = clock();
  for (int i=0; i<100000000; i++) {
    double x = -5.0 + 12.0*drand48();
    gensum += (*gengrid->reverse_map)(gengrid, x);
  }
  gend = clock();
  
  double cent_time = (double)(cend-cstart+rstart-rend)/(double)CLOCKS_PER_SEC;
  double gen_time  = (double)(gend-gstart+rstart-rend)/(double)CLOCKS_PER_SEC;
  fprintf (stderr, "%d %d\n", centsum, gensum);
  fprintf (stderr, "center_grid  time = %1.3f s.\n", cent_time);
  fprintf (stderr, "general_grid time = %1.3f s.\n", gen_time);
}

void
TestNUBasis()
{
  NUgrid* centgrid = create_center_grid (-5.0, 7.0, 10.0, 20);
  NUBasis* basis = create_NUBasis (centgrid, true);

  double bfuncs[4];
  for (double x=-5.0; x<=7.0; x+=0.001) {
    get_NUBasis_funcs_d (basis, x, bfuncs);
    fprintf (stderr, "%1.12f %1.12f %1.12f %1.12f %1.12f\n",
	     x, bfuncs[0], bfuncs[1], bfuncs[2], bfuncs[3]);
  }
}

void
TestNUBspline()
{
  NUgrid* centgrid = create_center_grid (-5.0, 7.0, 10.0, 20);
  NUBasis* basis = create_NUBasis (centgrid, true);
  float data[20];
  for (int i=0; i<20; i++) {
    double x = centgrid->points[i];
    double angle = (x+5.0)/12.0 * 2.0*M_PI;
    data[i] = sin(angle);
  }
  BCtype_s bc;
  //  bc.lCode = PERIODIC;  bc.rCode = PERIODIC;
  //  bc.lCode = DERIV1; bc.lVal = 2.0*M_PI/12.0;
  //  bc.rCode = DERIV1; bc.rVal = 2.0*M_PI/12.0;
  bc.lCode = NATURAL;  bc.rCode = FLAT;
  NUBspline_1d_s *spline = create_NUBspline_1d_s (centgrid, bc, data);
  for (double x=-5.0; x<=7.0; x+=0.001) {
    float val;
    eval_NUBspline_1d_s (spline, x, &val);
    double angle = (x+5.0)/12.0 * 2.0*M_PI;
    fprintf (stderr, "%1.16e %1.16e %1.16e\n", x, val, sin(angle));
  }
}

void
TestNUB_2d_s()
{
  int Mx=30, My=35;
  NUgrid *x_grid = create_center_grid (-3.0, 4.0, 7.5, Mx);
  NUgrid *y_grid = create_center_grid (-1.0, 9.0, 3.5, My);
  float data[Mx*My];
  for (int ix=0; ix<Mx; ix++)
    for (int iy=0; iy<My; iy++)
      data[ix*My+iy] = -1.0+2.0*drand48();
  
  BCtype_s xBC, yBC;
  xBC.lCode = PERIODIC;
  yBC.lCode = PERIODIC;
//   xBC.lCode = FLAT;  xBC.rCode = FLAT;
//   yBC.lCode = FLAT;  yBC.rCode = FLAT;



  NUBspline_2d_s *spline = create_NUBspline_2d_s (x_grid, y_grid, xBC, yBC, data);
  
  int xFine = 400;
  int yFine = 400;
  FILE *fout = fopen ("2d_s.dat", "w");
  double xi = x_grid->start;
  double xf = x_grid->end;// + x_grid->points[1] - x_grid->points[0];
  double yi = y_grid->start;
  double yf = y_grid->end;// + y_grid->points[1] - y_grid->points[0];
  for (int ix=0; ix<xFine; ix++) {
    double x = xi+ (double)ix/(double)(xFine)*(xf-xi);
    for (int iy=0; iy<yFine; iy++) {
      double y = yi + (double)iy/(double)(yFine)*(yf-yi);
      float val;
      eval_NUBspline_2d_s (spline, x, y, &val);
      fprintf (fout, "%1.16e ", val);
    }
    fprintf (fout, "\n");
  }
  fclose (fout);
}

void
TestNUB_3d_s()
{
  int Mx=20, My=27, Mz=23;
  NUgrid *x_grid = create_center_grid (-3.0, 4.0,  7.5, Mx);
  NUgrid *y_grid = create_center_grid (-1.0, 9.0,  3.5, My);
  NUgrid *z_grid = create_center_grid (-1.8, 2.0,  2.8, Mz);
  float data[Mx*My*Mz];
  for (int ix=0; ix<Mx; ix++)
    for (int iy=0; iy<My; iy++)
      for (int iz=0; iz<Mz; iz++) {
	data[(ix*My+iy)*Mz+iz] = -1.0+2.0*drand48();
	//data[(ix*My+iy)*Mz+iz] = data[((ix%(Mx-1))*My+(iy%(My-1)))*Mz+(iz%(Mz-1))];
      }
  
  BCtype_s xBC, yBC, zBC;
//   xBC.lCode = PERIODIC;
//   yBC.lCode = PERIODIC;
  xBC.lCode = PERIODIC;  xBC.rCode = PERIODIC;
  yBC.lCode = PERIODIC;  yBC.rCode = PERIODIC;
  zBC.lCode = PERIODIC;  zBC.rCode = PERIODIC;

  NUBspline_3d_s *spline = create_NUBspline_3d_s (x_grid, y_grid, z_grid, xBC, yBC, zBC, data);
  
  int xFine = 200, yFine = 200, zFine=200;
  FILE *fout = fopen ("3d_s.dat", "w");
  double xi = x_grid->start;  double xf = x_grid->end;
  double yi = y_grid->start;  double yf = y_grid->end;
  double zi = z_grid->start;  double zf = z_grid->end;
  for (int ix=0; ix<xFine; ix++) {
    double x = xi+ (double)ix/(double)(xFine)*(xf-xi);
    for (int iy=0; iy<yFine; iy++) {
      double y = yi + (double)iy/(double)(yFine)*(yf-yi);
      for (int iz=0; iz<zFine; iz++) {
	double z = zi + (double)iz/(double)(zFine)*(zf-zi);
	float val;
	eval_NUBspline_3d_s (spline, x, y, z, &val);
	fprintf (fout, "%1.16e ", val);
      }
    }
    fprintf (fout, "\n");
  }
  fclose (fout);
}

void
TestNUB_2d_d()
{
  int Mx=30, My=35;
  NUgrid *x_grid = create_center_grid (-3.0, 4.0, 7.5, Mx);
  NUgrid *y_grid = create_center_grid (-1.0, 9.0, 3.5, My);
  double data[Mx*My];
  for (int ix=0; ix<Mx; ix++)
    for (int iy=0; iy<My; iy++)
      data[ix*My+iy] = -1.0+2.0*drand48();
  
  BCtype_d xBC, yBC;
  xBC.lCode = PERIODIC;
  yBC.lCode = PERIODIC;
//   xBC.lCode = FLAT;  xBC.rCode = FLAT;
//   yBC.lCode = FLAT;  yBC.rCode = FLAT;



  NUBspline_2d_d *spline = create_NUBspline_2d_d (x_grid, y_grid, xBC, yBC, data);
  
  int xFine = 400;
  int yFine = 400;
  FILE *fout = fopen ("2d_d.dat", "w");
  double xi = x_grid->start;
  double xf = x_grid->end;// + x_grid->points[1] - x_grid->points[0];
  double yi = y_grid->start;
  double yf = y_grid->end;// + y_grid->points[1] - y_grid->points[0];
  for (int ix=0; ix<xFine; ix++) {
    double x = xi+ (double)ix/(double)(xFine)*(xf-xi);
    for (int iy=0; iy<yFine; iy++) {
      double y = yi + (double)iy/(double)(yFine)*(yf-yi);
      double val;
      eval_NUBspline_2d_d (spline, x, y, &val);
      fprintf (fout, "%1.16e ", val);
    }
    fprintf (fout, "\n");
  }
  fclose (fout);
}

int main()
{
  // TestCenterGrid();
  // TestGeneralGrid();
  // GridSpeedTest();
  // TestNUBasis();
  // TestNUBasis();
  // TestNUBspline();
  // TestNUB_2d_s();
  TestNUB_2d_d();
  //TestNUB_3d_s();
}
