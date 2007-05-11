#include "blip_create.h"
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#define _XOPEN_SOURCE 600
#include <stdlib.h>


inline double dot (double a[3], double b[3])
{
  return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
}

// This function creates a single-precision real blip function from a
// set of plane-wave coefficients.  lattice is a 3x3 array specifying
// the lattice vectors.  The first lattice vector is given
// contiguously at latice[0], the second at lattice[3], and the third
// at lattice[6].  The next is a list of 3D G-vectors in the format:
// G_x[0] G_y[0] G_z[0], G_x[1], G_y[1], G_z[1],...
// Next, complex plane-wave coefficents are given, one for each
// G-vector.  Next, the number of G-vectors is given, followed by
// a factor which increases the density of the real-space grid.  A
// factor of 1.0 uses the minimum density to avoid aliasing.  Finally,
// the last parameter specifies whether to take the real or imaginary part.
UBspline_3d_s
create_blip_3d_s (double *lattice, double *Gvecs, 
		  complex_float *coefs, int numG,
		  double factor, bool useReal)
{
  int max_ix=0, max_iy=0, max_iz=0;
  int Nx, Ny, Nz;
  double twoPiInv = 1.0/(2.0*M_PI);
  for (int i=0; i<numG; i++) {
    double *G = Gvecs+3*i;
    int ix = round (twoPiInv * dot (lattice+0, G));
    int iy = round (twoPiInv * dot (lattice+3, G));
    int iz = round (twoPiInv * dot (lattice+6, G));
    if (abs(ix) > max_ix)   max_ix = ix;
    if (abs(iy) > max_iy)   max_iy = iy;
    if (abs(iz) > max_iz)   max_iz = iz;
  }
  Nx = 2*max_ix + 1;
  Ny = 2*max_ix + 1;
  Nz = 2*max_ix + 1;
  Nx = (int) ceil(factor*Nx);
  Ny = (int) ceil(factor*Ny);
  Nz = (int) ceil(factor*Nz);

  // FFTs are a little faster with even dimensions.
  if ((Nx%2)==1) Nx++;
  if ((Ny%2)==1) Ny++;
  if ((Nz%2)==1) Nz++;

  // Now allocate space for FFT box
  complex_float *fft_box;
  posix_memalign ((void**)&fft_box, (size_t)16, sizeof(complex_float)*Nx*Ny*Nz);

  // Create FFTW plan
  fftwf_plan_dft_3d (Nx, Ny, Nz, (fftwf_complex*)fft_box, (fftwf_complex*)fft_box, 1,
		     FFTW_ESTIMATE);

  
  // Zero-out fft-box
  for (int i=0; i<Nx*Ny*Nz; i++)
    fft_box[i] = (complex_float)0.0f;
  
  // Now fill in fft box with coefficients in the right places
  for (int i=0; i<numG; i++) {
    double *G = Gvecs+3*i;
    int ix = round (twoPiInv * dot (lattice+0, G));
    int iy = round (twoPiInv * dot (lattice+3, G));
    int iz = round (twoPiInv * dot (lattice+6, G));
    ix = (ix + Nx)%Nx;
    iy = (iy + Ny)%Ny;
    iz = (iz + Nz)%Nz;
    double gamma = 1.0;
    if (fabs(G[0]) > 1.0e-12)
      gamma *= (3.0/(G[0]*G[0]*G[0]*G[0])*(3.0 - 4.0*cos(G[0]) + cos(2.0*G[0])));
    else
      gamma *= 1.5;
    if (fabs(G[1]) > 1.0e-12)
      gamma *= (3.0/(G[1]*G[1]*G[1]*G[1])*(3.0 - 4.0*cos(G[1]) + cos(2.0*G[1])));
    else
      gamma *= 1.5;
    if (fabs(G[2]) > 1.0e-12)
      gamma *= (3.0/(G[2]*G[2]*G[2]*G[2])*(3.0 - 4.0*cos(G[2]) + cos(2.0*G[2])));
    else
      gamma *= 1.5;
    fft_box[ix*(Ny+iy)*Nz+iz] = coefs[i]/gamma;
  }
  
  // Execute the FFTW plan
  

  free (fft_box);
}
