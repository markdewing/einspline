if WANT_BLIPS
   MAYBE_BLIPS = blip_create.c blip_create.h
   CHECK_BLIPS = test_blip
endif


check_PROGRAMS =  TestBspline TestNUBspline test_fbspline $(CHECK_BLIPS)

TestBspline_SOURCES   = TestBspline.c
TestNUBspline_SOURCES = TestNUBspline.c
test_fbspline_SOURCES = test_fbspline.f
if WANT_BLIPS
   test_blip_SOURCES     = test_blip.c
endif

LDADD = 
AM_CCFLAGS = -g 

TestBspline_LDADD   = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
TestNUBspline_LDADD = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
test_fbspline_LDADD = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
test_blip_LDADD     = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)


lib_LTLIBRARIES = libeinspline.la
libeinspline_la_SOURCES =                             \
                       aligned_alloc.h                \
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
                       nubspline_create.h             \
                       nubspline_eval_sse_s.h         \
                       nubspline_eval_std_s.h         \
                       nubspline_eval_sse_c.h         \
                       nubspline_eval_std_c.h         \
                       nubspline_eval_sse_d.h         \
                       nubspline_eval_std_d.h         \
                       nubspline_eval_sse_z.h         \
                       nubspline_eval_std_z.h         \
                       nubspline_structs.h            \
                       nubasis.h                      \
                       nubasis.c                      \
                       nugrid.h                       \
                       nugrid.c                       \
                       fbspline.c                     \
                       fbspline.h                     \
                       fnubspline.c                   \
                       fnubspline.h                   \
                       $(MAYBE_BLIPS)


libeinspline_la_LIBADD = $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)


includedir=$(prefix)/include/einspline


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
                  bspline_eval_std_z.h bspline.h bspline_structs.h            \
                  fbspline.h fnubspline.h                                     \
                  nubspline_base.h nubspline_create.h                         \
                  nubspline_eval_sse_s.h nubspline_eval_std_s.h               \
	          nubspline_eval_sse_c.h nubspline_eval_std_c.h               \
                  nubspline_eval_sse_d.h nubspline_eval_std_d.h               \
	          nubspline_eval_sse_z.h nubspline_eval_std_z.h               \
                  nubspline_structs.h nubasis.h nubspline.h nugrid.h


EXTRA_DIST = 
