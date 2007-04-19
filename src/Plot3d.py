#!/bin/env python
from pylab import *

data = load ('3d_d.dat');
d3 = reshape (data, (200, 200, 200));

#pcolor (d3[:,:,0], shading='flat');
x = concatenate ((d3[:,20,20], d3[:,20,20]));
y = concatenate ((d3[20,:,20], d3[20,:,20]));
z = concatenate ((d3[20,20,:], d3[20,20,:]));

plot (x, 'r');
hold(True);
plot (y, 'g');
plot (z, 'b');
show();
