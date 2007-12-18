Summary:  einspline B-spline library
Name: einspline
Version: 0.7.7
Release: 1
License: GNU Public License
Group: System Environment/Libraries
URL: http://einspline.sf.net
Source0: %{name}-%{version}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-root

%description
Einspline is a C library for the creation and evaluation of 1D, 2D,
and 3D cubic B-splines.  It creates interpolating B-splines with
flexible boundary conditions.  It supports real and complex data with
single or double precision.  It also allows both uniform and
nonuniform grid spacing.  Evaluation routines include standard
versions and those optimized for SSE SIMD instructions.

%package devel
Summary: Header files, libraries and development documentation for %{name}
Group: Development/Libraries
Requires: %{name} = %{version}-%{release}

%description devel
This package contains the header files, static libraries and development
documentation for %{name}. If you like to develop programs using %{name},
you will need to install %{name}-devel.

%prep
%setup -q

%build
%configure

%install
%{__rm} -rf %{buildroot}
%makeinstall

%clean
%{__rm} -rf %{buildroot}

%files
%defattr(-, root, root, 0755)
%doc AUTHORS ChangeLog COPYING COPYRIGHT NEWS README* TODO
%{_libdir}/*.so.*

%files devel
%defattr(-, root, root, 0755)
%doc www
%doc %{_mandir}/man?/*
%doc %{_infodir}/*.info*
%{_bindir}/*
%{_includedir}/*.h
%{_includedir}/*.f
%{_libdir}/*.a
%{_libdir}/*.so
%{_libdir}/pkgconfig/*.pc
#exclude %{_libdir}/*.la
%doc

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%changelog
* Tue Nov 20 2007 Ken Esler <kesler@ciw.edu> - 
- Initial build.

