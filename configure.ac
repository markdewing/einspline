#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
AC_PREREQ(2.61)
AC_INIT(einspline, 0.5, kesler@ciw.edu, einspline)
AM_INIT_AUTOMAKE(einspline, 0.6)
AC_CONFIG_SRCDIR([src/bspline.h])
AC_CONFIG_HEADER([src/config.h])


# Checks for programs.
AC_PROG_CC_STDC
AC_PROG_CXX
AC_PROG_LIBTOOL
PKG_PROG_PKG_CONFIG

# Optimal compiler flags
AX_CC_MAXOPT
AX_CXX_MAXOPT
AX_F77_MAXOPT


AC_CHECK_LIB([m], [sqrt])
AC_MSG_CHECKING([LIBS = $LIBS])
# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([stdlib.h sys/time.h unistd.h])


AC_MSG_CHECKING([optimal compiler flags])
CFLAGS="${CFLAGS} -g"
AC_MSG_RESULT([${CFLAGS}])

AC_ARG_ENABLE(altivec,[  --enable-altivec        enable PowerPC SIMD extensions])
if test "x$enable_altivec" = "xyes"; then
   CFLAGS="$CFLAGS -faltivec -DUSE_ALTIVEC"
   FFLAGS="$FFLAGS -DUSE_ALTIVEC"
fi

AC_ARG_ENABLE(sse2, [  --enable-sse2           enable SSE2 SIMD instructions])
if test "x$enable_sse2" = "xyes"; then
  AC_CHECK_HEADER([xmmintrin.h],
                  [AC_DEFINE([HAVE_XMMINTRIN_H],[1],
                   [xmmintrin.h is available.])],
                  [AC_MSG_ERROR([Cannot utilize SSE2 extensions without xmmintrin.h])])
  AC_CHECK_HEADER([emmintrin.h],
                  [AC_DEFINE([HAVE_EMMINTRIN_H],[1],
                   [emmintrin.h is available.])],
                  [AC_MSG_ERROR([Cannot utilize SSE2 extensions without emmintrin.h])])
  AC_DEFINE([HAVE_SSE2], [1], [SSE2 instructions are available])
fi

AC_ARG_ENABLE(sse3, [  --enable-sse3           enable SSE3 SIMD instructions])
if test "x$enable_sse3" = "xyes"; then
  AC_CHECK_HEADER([xmmintrin.h],
                  [AC_DEFINE([HAVE_XMMINTRIN_H],[1],
                   [xmmintrin.h is available.])],
                  [AC_MSG_ERROR([Cannot utilize SSE2 extensions without xmmintrin.h])])
  AC_CHECK_HEADER([emmintrin.h],
                  [AC_DEFINE([HAVE_EMMINTRIN_H],[1],
                   [emmintrin.h is available.])],
                  [AC_MSG_ERROR([Cannot utilize SSE2 extensions without emmintrin.h])])
  AC_CHECK_HEADER([pmmintrin.h],
                  [AC_DEFINE([HAVE_PMMINTRIN_H],[1],
                   [pmmintrin.h is available.])],
                  [AC_MSG_ERROR([Cannot utilize SSE2 extensions without pmmintrin.h])])
  AC_DEFINE([HAVE_SSE2], [1], [SSE2 instructions are available])
  AC_DEFINE([HAVE_SSE3], [1], [SSE3 instructions are available])
fi

AC_ARG_ENABLE(blips,   [  --enable-blips          enable routines for creating BLIPS])
AM_CONDITIONAL([WANT_BLIPS], [test "$enable_blips" = "yes"])
if test "x$enable_blips" = "xyes"; then
  PKG_CHECK_MODULES(FFTW3,  fftw3,  fftw3_ok=yes,  fftw3_ok=no)
  PKG_CHECK_MODULES(FFTW3F, fftw3f, fftw3f_ok=yes, fftw3f_ok=no)
  if test "x$fftw3_ok" = "xyes"; then
    AC_DEFINE([HAVE_FFTW3], [1], [FFTW3 library is available])
  else
    AC_MSG_ERROR([Blip construction routines require that FFTW3 be installed.])
  fi
  if test "x$fftw3f_ok" = "xyes"; then
    AC_DEFINE([HAVE_FFTW3F], [1], [FFTW3F library is available])
  else
    AC_MSG_ERROR([Blip construction routines require that FFTW3 be installed.])
  fi
fi

AC_ARG_ENABLE(profile,[  --enable-profile        instrument code with profiling information])
if test "x$enable_profile" = "xyes"; then
   CFLAGS="-pg $CFLAGS"
   FFLAGS="-pg $FFLAGS"
   LDFLAGS="-pg $LDFLAGS"
fi

AC_ARG_ENABLE(debug,[  --enable-debug          enable code for debugging checks])
if test "x$enable_debug" = "xyes"; then
   CFLAGS="-DDEBUG $CFLAGS"
   FFLAGS="-DDEBUG $FFLAGS"
fi


AC_F77_WRAPPERS
AC_F77_LIBRARY_LDFLAGS
AC_F77_DUMMY_MAIN

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_FUNC_STRTOD
AC_CHECK_FUNCS([clock_gettime floor pow sqrt strtol])

AC_SUBST(PKGDATADEF)

AC_CONFIG_FILES([ Makefile                  \
		  src/Makefile              \
                  doc/Makefile              \
		  www/Makefile              \
                  einspline.pc  ])
AC_OUTPUT
