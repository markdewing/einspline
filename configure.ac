#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
AC_PREREQ(2.59)
AC_INIT(libbspline, 0.5, kesler@ciw.edu, libbspline)
AM_INIT_AUTOMAKE(libbspline, 0.5)
AC_CONFIG_SRCDIR([src/bspline.h])
AC_CONFIG_HEADER([config.h])


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
AC_MSG_RESULT([${CFLAGS}])

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

AC_ARG_WITH(boost,    [  --with-boost=/path/to/my/boost	include boost libraries; needed for qmcpack])
if (test "x$with_boost" != "x"); then
	 	BOOST_LIB="-I$with_boost/include -L$with_boost/lib"
else # don't link with boost
		BOOST_LIB=" "
fi

AC_SUBST(BOOST_LIB)
AC_F77_WRAPPERS

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
		  src/Makefile              ])
AC_OUTPUT
