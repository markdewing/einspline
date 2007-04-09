#include "nubspline_create.h"
#include <stdio.h>
#include <assert.h>

void
TestCenterGrid()
{
  NUgrid* grid = create_center_grid (-5.0, 7.0, 6.0, 200);

/*   for (int i=0; i<200; i++) */
/*     fprintf (stderr, "%16.12f\n", grid->points[i]); */
  for (int i=0; i<10000; i++) {
    double x = -5.0+12.0*drand48();
    int lo = (*grid->reverse_map)(grid, x);
    assert (x >= grid->points[lo]);
    assert (x <= grid->points[lo+1]);
  }
}

main()
{
  TestCenterGrid();
}
