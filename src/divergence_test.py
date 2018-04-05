import numpy
import matplotlib.pyplot as plt
from ftle import *


def testDivergence(pLength, pHeight):
    test = doubleGyreFTLE( length=98, 
                        height=98, 
                        maxx=1, 
                        maxy=1,
                        percentLength=pLength, 
                        percentHeight=pHeight, 
                        estimateVelocity=True  )

    coords = []
    for i in range(1,99):
        for j in range(1,99):
            coords.append((round(i*0.01,2),round(j*0.01,2)))

    vfield = dict( (coord, (test.getEstimatedVelocity(coord[0], coord[1], 20)[0], \
                            test.getEstimatedVelocity(coord[0], coord[1], 20)[1]) ) for coord in coords)

    dx, dy = 0.01, 0.01
    sum_div = 0
    for key in vfield.keys():
        # print( 'checking key: ', key )
        #print( 'for x direction, we check: ', (round(key[0]-dx,2), key[1]), (round(key[0]+dx), key[1]))
        try:
            dvx_dx = (vfield[(round(key[0]+dx, 2),key[1])][0] - vfield[(round(key[0]-dx,2),key[1])][0])/(2*dx)
            dvy_dy = (vfield[(key[0],round(key[1]+dy,2))][1] - vfield[(key[0],round(key[1]-dy,2))][1])/(2*dy)
            # print( 'dvx_dx = ', dvx_dx )
            # print( 'dvy_dy = ', dvy_dy )
            # print( 'div = ', dvx_dx + dvy_dy )
            sum_div += (dvx_dx + dvy_dy)**2
        except KeyError:
            pass
            # print( 'cant find ', (key[0]-dx, key[1]), (key[0]+dx, key[1]) )

    avg_div = np.sqrt(sum_div[0])/len(vfield.keys())
    print( 'for {}% info, avg div = {}'.format(pLength*100, avg_div) )
    
    return avg_div 


results = []
for i in range(20,100,5):
    pLength = i*0.01
    pHeight = i*0.01
    results.append( testDivergence(pLength, pHeight) )

fig = plt.figure()
ax = plt.subplot(111)
plt.title('Divergence calculations w.r.t percentage of interpolated velocity data')
ax.set_xlabel('Interpolated velocity data (%)')
ax.set_ylabel('Approximated Divergence')
plt.plot( [100-x for x in range(20,100,5)], results )
plt.tight_layout()
plt.grid(ls='--')
plt.savefig('../figures/divergenceTest.png', format='png')



