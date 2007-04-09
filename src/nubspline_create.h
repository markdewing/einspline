#ifndef NUBSPLINE_CREATE_H
#define NUBSPLINE_CREATE_H

#include "nubspline_base.h"

NUgrid* 
create_center_grid (double start, double end, double ratio, int num_points);

NUgrid*
create_general_grid (double *points, int num_points);


#endif
