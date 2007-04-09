#include "nubspline_create.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

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

main()
{
  TestCenterGrid();
  TestGeneralGrid();
}
