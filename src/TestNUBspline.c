#include "nubspline_create.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>

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



main()
{
//   TestCenterGrid();
//   TestGeneralGrid();
//   GridSpeedTest();
  TestNUBasis();
}
