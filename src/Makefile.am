check_PROGRAMS =  TestBspline TestNUBspline

TestBspline_SOURCES   = TestBspline.c
TestNUBspline_SOURCES = TestNUBspline.c

LDADD = 
AM_CCFLAGS = -g 

TestBspline_LDADD   = libbspline.la $(FLIBS)
TestNUBspline_LDADD = libbspline.la $(FLIBS)

lib_LTLIBRARIES = libbspline.la
libbspline_la_SOURCES =                               \
                       bspline_base.h                 \
                       bspline_create.c               \
                       bspline_create.h               \
                       bspline_structs.h              \
                       bspline_data.c                 \
                       bspline_eval_std_s.h           \
                       bspline_eval_sse_s.h           \
                       bspline_eval_std_c.h           \
                       bspline_eval_sse_c.h           \
                       bspline_eval_std_d.h           \
                       bspline_eval_sse_d.h           \
                       bspline_eval_std_z.h           \
                       bspline_eval_sse_z.h           \
                       nubspline_base.h               \
                       nubspline_create.c             \
                       nubspline_create.h             
libbspline_la_LIBADD = $(FLIBS)


includedir=$(prefix)/include/bspline


#if WITH_F77
#     libsprng_la_SOURCES += common/fwrap_mpi.c
#endif

#if HAVE_GMP
#     libsprng_la_SOURCES += pmlcg/pmlcg.h pmlcg/pmlcg.c pmlcg/info.h pmlcg/basic.h 
#endif


include_HEADERS = bspline_base.h bspline_create.h bspline_eval_sse_c.h        \
                  bspline_eval_sse_d.h bspline_eval_sse_s.h                   \
                  bspline_eval_sse_z.h bspline_eval_std_c.h                   \
                  bspline_eval_std_d.h bspline_eval_std_s.h                   \
                  bspline_eval_std_z.h bspline.h bspline_structs.h


EXTRA_DIST = 
