if WANT_BLIPS
   MAYBE_BLIPS = blip_create.c blip_create.h
   CHECK_BLIPS = test_blip
endif


check_PROGRAMS =  TestBspline TestNUBspline test_fbspline \
                  test_bspline_s test_bspline_d test_multi \
                  test_fmulti_bspline $(CHECK_BLIPS)


TestBspline_SOURCES         = TestBspline.c
TestNUBspline_SOURCES       = TestNUBspline.c
test_fbspline_SOURCES       = test_fbspline.f
test_fmulti_bspline_SOURCES = test_fmulti_bspline.f
if WANT_BLIPS
   test_blip_SOURCES     = test_blip.c
endif
test_bspline_s_SOURCES = test_bspline_s.c
test_bspline_d_SOURCES = test_bspline_d.c
test_multi_SOURCES     = test_multi.c

LDADD = 
AM_CCFLAGS = -g 

TestBspline_LDADD    = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
TestNUBspline_LDADD  = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
test_fbspline_LDADD  = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
test_fmulti_bspline_LDADD = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
test_blip_LDADD      = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
test_bspline_s_LDADD = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
test_bspline_d_LDADD = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)
test_multi_LDADD     = libeinspline.la $(FLIBS) $(FFTW3_LIBS) $(FFTW3F_LIBS)


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
                       multi_bspline_create.c         \
                       multi_bspline_create.h         \
                       multi_bspline_eval_sse_c.h     \
                       multi_bspline_eval_std_c.h     \
                       multi_bspline_eval_sse_d.h     \
                       multi_bspline_eval_std_d.h     \
                       multi_bspline_eval_sse_s.h     \
                       multi_bspline_eval_std_s.h     \
                       multi_bspline_eval_sse_z.h     \
                       multi_bspline_eval_std_z.h     \
                       multi_bspline_structs.h        \
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
                       fmulti_bspline.c               \
                       fmulti_bspline.h               \
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
                  fbspline.h fmulti_bspline.h  fnubspline.h                   \
                  multi_bspline_create.h      multi_bspline_structs.h         \
                  multi_bspline_eval_sse_c.h  multi_bspline_eval_sse_c.h      \
                  multi_bspline_eval_sse_d.h  multi_bspline_eval_sse_d.h      \
                  multi_bspline_eval_sse_s.h  multi_bspline_eval_sse_s.h      \
                  multi_bspline_eval_sse_z.h  multi_bspline_eval_std_z.h      \
                  nubspline_base.h nubspline_create.h                         \
                  nubspline_eval_sse_s.h nubspline_eval_std_s.h               \
	          nubspline_eval_sse_c.h nubspline_eval_std_c.h               \
                  nubspline_eval_sse_d.h nubspline_eval_std_d.h               \
	          nubspline_eval_sse_z.h nubspline_eval_std_z.h               \
                  nubspline_structs.h nubasis.h nubspline.h nugrid.h          \
                  config.h


EXTRA_DIST = 
