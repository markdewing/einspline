#!/bin/env python
from pylab import *

data = load ('2d_s.dat');
d2 = reshape (data, (400,400));

#pcolor (d3[:,:,0], shading='flat');
x = concatenate ((d2[:,20], d2[:,20]));
y = concatenate ((d2[20,:], d2[20,:]));

plot (x, 'r');
hold(True);
plot (y, 'g');
show();
