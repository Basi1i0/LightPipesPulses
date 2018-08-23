from LightPipes import *
import LightPipes as lp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy

"""
    Young's experiment.
    Two holes with radii, R, separated by, d, in a screen are illuminated by a plane wave. The interference pattern
    at a distance, z, behind the screen is calculated.
"""

wavelength=3*um
size=10.0*mm
N=100
z=20*cm
R=0.2*mm
d=3*mm

#plt.rcParams["figure.figsize"] = [6,4]

#for wavelength in numpy.linspace(0.1, 10, 10)*um:
LP1 = LP(size,wavelength,N)
LP2 = LP(size,2*wavelength,N)
    
LP1.GaussAperture(R/2.0,  d/2.0, 0, 1)
LP2.GaussAperture(R/2.0,  d/2.0, 0, 1)
    
#axisField = []

#for z in numpy.linspace(9.998, 10, 100)*cm:
#    print (z)
        
LP1.Fresnel(z)
LP2.Fresnel(z)


LP1.Fresnel(z)
LP2.Fresnel(z)

I1=LP1.Intensity(2)
plt.imshow(I1, cmap='hot'); plt.axis('off');plt.title('intensity pattern')
plt.show()

I2=LP2.Intensity(2)
plt.imshow(I2, cmap='hot'); plt.axis('off');plt.title('intensity pattern')
plt.show()
#    
Fprop = numpy.add(LP1.Field(), LP2.Field())
#   
##axisField.append(Fprop[N//2][N//2])
## 
Isum= LP1.IntensityExt(2, Fprop)
plt.imshow(Isum, cmap='hot'); plt.axis('off'); plt.title('intensity pattern')
plt.show()
#
##plt.plot(numpy.real(axisField))
