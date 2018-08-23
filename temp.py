# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from LightPipes import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

wavelength=800*nm
N = 200
size = 0.005


z_start=0.001*cm; z_end= 150*cm;
steps=50;
delta_z=(z_end-z_start)/steps
z=z_start

F=Begin(size,wavelength,N);
F=GaussAperture(0.001, 0, 0, 1, F)

#F=Fresnel(f,F);
F=Lens(0.8,0,0,F);
I01 = Intensity(0, F)

Fin=F

intensities = []
for i in range(0, steps):
    F=Fresnel(z, Fin);
    I=Intensity(0, F);
    intensities.append(I)
    z=z+delta_z
    
# z x y
Iswapped = swapaxes(intensities, 0, 2) # y x z
peak_intensity = numpy.max(Iswapped[N//2])
plt.imshow(numpy.log(numpy.add(Iswapped[N//2], 1)), cmap='hot', vmin=0, vmax= log(1+peak_intensity), 
           aspect= size/z_end*steps/N*70);
plt.show()
        



