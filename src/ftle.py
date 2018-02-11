import numpy as np
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import axes3d, Axes3D
import time as time
import pdb

class FTLE():
    def __init__(self, **kwargs):
        self.length = 100
        self.height = 50
        self.percentLength = 1  # set resolution of descretization
        self.percentHeight = 1
        self.minx = 0.01
        self.miny = 0.01
        self.maxx = 2
        self.maxy = 1
        self.delta = 1e-5
        self.earthRotation = 1
        self.phi = np.pi*40.7128/180  # lattitude
        self.coriolis = 2*self.earthRotation*np.sin(self.phi)
        self.dimensions = 2
        self.state = np.zeros((self.length,4))
        self.dt = 0.1
        self.elapsedTime = 20
        self.mappingfile = None
        self.estimateVelocity = False

        self.length=kwargs['length'] if 'length' in kwargs else self.length
        self.height=kwargs['height'] if 'height' in kwargs else self.height
        self.percentLength=kwargs['percentLength'] if 'percentLength' in kwargs else self.percentLength 
        self.percentHeight=kwargs['percentHeight'] if 'percentHeight' in kwargs else self.percentHeight 
        self.delta=kwargs['delta'] if 'delta' in kwargs else self.delta
        self.earthRotation=kwargs['earthRotation'] if 'earthRotation' in kwargs else self.earthRotation 
        self.phi=kwargs['phi'] if 'phi' in kwargs else self.phi 
        self.coriolis=kwargs['coriolis'] if 'coriolis' in kwargs else self.coriolis 
        self.dimensions=kwargs['dimensions'] if 'dimensions' in kwargs else self.dimensions 
        self.state=kwargs['state'] if 'state' in kwargs else self.state 
        self.dt=kwargs['dt'] if 'dt' in kwargs else self.dt 
        self.elapsedTime=kwargs['elapsedTime'] if 'elapsedTime' in kwargs else self.elapsedTime 
        self.mappingfile=kwargs['mappingfile'] if 'mappingfile' in kwargs else self.mappingfile 
        self.estimateVelocity=kwargs['estimateVelocity'] if 'estimateVelocity' in kwargs else self.estimateVelocity 


    def doubleGyreVelocity(self, x, y, t, a = 0.1, epsilon = 0.25, w = 0.2*np.pi, dims = 2):
        f = epsilon*np.sin(w*t)*x**2 + (1-2*epsilon*np.sin(w*t))*x
        if dims == 2:
            _vx = a*np.pi*np.sin(np.pi*f)*np.cos(np.pi*y)
            _vy = -1*a*np.pi*np.cos(np.pi*f)*np.sin(np.pi*y)*(2*epsilon*np.sin(w*t)*x 
                + (1-2*epsilon*np.sin(w*t)))
        elif dims == 3:
            print( "not sure if we can do that yet!")
        # return np.vstack([vx,vy])
        return _vx, _vy

    # def getEstimatedVelocity(self, x, y, t):
    #     def find_nearest(array,values):
    #         idxs = [(np.abs(array-value)).argmin() for value in values]
    #         return np.array([array[idx] for idx in idxs])

    #     est_x = find_nearest(np.linspace(self.minx,self.maxx,self.length*self.percentLength), x)
    #     est_y = find_nearest(np.linspace(self.miny,self.maxy,self.height*self.percentHeight), y)

    #     return self.doubleGyreVelocity(est_x, est_y, t)

    # def getEstimatedVelocity(self, x, y, t):
    #     _xarr = np.linspace(self.minx, self.maxx, self.length*self.percentLength)
    #     _yarr = np.linspace(self.miny, self.maxy, self.height*self.percentHeight)
    #     xx_, yy_ = np.meshgrid(_xarr,_yarr)
    #     vx_, vy_ = self.doubleGyreVelocity(xx_, yy_, t)[0], self.doubleGyreVelocity(xx_, yy_, t)[1] 
    #     vxfunc, vyfunc = interpolate.interp2d(_xarr, _yarr, vx_), interpolate.interp2d(_xarr, _yarr, vy_)
    #     vx, vy = vxfunc(x, y)[0], vyfunc(x, y)[0]
    #     # print(vx)
    #     return vx, vy


    def getEstimatedVelocity(self, x, y, t):
        # pdb.set_trace()
        _xarr = np.linspace(self.minx, self.maxx, self.length*self.percentLength)
        _yarr = np.linspace(self.miny, self.maxy, self.height*self.percentHeight)
        xdata = self.doubleGyreVelocity(*np.meshgrid(_xarr, _yarr, indexing='ij', sparse=True), t)[0]
        ydata = self.doubleGyreVelocity(*np.meshgrid(_xarr, _yarr, indexing='ij', sparse=True), t)[1]
        # vxfunc,vyfunc = interpolate.interp2d(_xarr, _yarr, xdata), interpolate.interp2d(_xarr, _yarr, ydata)
        vxfunc = RegularGridInterpolator((_xarr, _yarr), xdata, method='nearest')
        vyfunc = RegularGridInterpolator((_xarr, _yarr), ydata, method='nearest')
        x, y = np.clip(x,self.minx,self.maxx), np.clip(y,self.miny,self.maxy)
        points = np.transpose(np.array([x, y]))
        outxv, outxy = vxfunc(points), vyfunc(points)
        return outxv, outxy


    def getVelocity(self, x, y, t):
        # print(self.doubleGyreVelocity(x, y, t)[0])
        return self.doubleGyreVelocity(x, y, t)


    def update(self, state, t):
        x = state[:,0]
        y = state[:,1]
        if self.estimateVelocity:
            vx, vy = self.getEstimatedVelocity(x, y, t)
        else:        
            vx, vy = self.getVelocity(x, y, t)
        return np.column_stack((-vx,-vy))
        # noise = B*np.random.normal(u,sigma,(L,2))
        # v = accel(state[:,0],state[:,1],r,t) + noise
        # return np.hstack((r,v))

    def rk4(self, state, t, dt=None):
        if dt is None:
            dt = self.dt
        tmp_state = state[:,0:2]
        k1 = dt*self.update(tmp_state,t)
        k2 = dt*self.update(tmp_state+0.5*k1,t+0.5*dt)
        k3 = dt*self.update(tmp_state+0.5*k2,t+0.5*dt)
        k4 = dt*self.update(tmp_state+k3,t+dt)
        tmp_state += (k1+2*k2+2*k3+k4)/6
        state[:,0] = np.clip(tmp_state[:,0],self.minx,self.maxx)
        state[:,1] = np.clip(tmp_state[:,1],self.miny,self.maxy)

        #noise = B*np.random.normal(u,sigma,(L,2))
        #state[:,4:6] += noise
        return state

    def createMapping(self, mappingfile=None, dt=None, elapsedTime=None):
        # pdb.set_trace()
        if dt is None:
            dt = self.dt
        if elapsedTime is None:
            elapsedTime = self.elapsedTime
        if mappingfile is None:
            mappingfile = self.mappingfile
        else:
            self.mappingfile = mappingfile

        # output = open('data/kevin_output_steps.txt','ab')
        output = open(mappingfile,'w')  #overwrite file!!
        for i in np.linspace(0.01, 0.99, self.height, np.float64):
            self.state = np.zeros((self.length, 2))  # (xposition, yposition)
            self.state[:,0] = np.linspace(0.01, 1.99, self.length, np.float64)
            self.state[:,1] = i*np.ones(self.length, np.float64)

            for t in np.arange(0,elapsedTime,dt):
                self.state = self.rk4(self.state,t,dt)
                
            np.savetxt(output,self.state)

    def plotter(self, filename=None, title=None, display=True):
 
        def Jacobian(X,Y):
            dx = 1.98/(self.length-1) # horizontal grid difference
            dy = 0.98/(self.height-1) # verticle grid difference
            J = np.empty([2,2],float)
            FTLE = np.empty([self.height-2,self.length-2],float)
            
            for i in range(0,self.height-2):
                for j in range(0,self.length-2):
                    J[0][0] = (X[(1+i)*self.length+2+j]-X[(1+i)*self.length+j])/(2*dx)
                    J[0][1] = (X[(2+i)*self.length+1+j]-X[i*self.length+1+j])/(2*dx)
                    J[1][0] = (Y[(1+i)*self.length+2+j]-Y[(1+i)*self.length+j])/(2*dy)
                    J[1][1] = (Y[(2+i)*self.length+1+j]-Y[i*self.length+1+j])/(2*dy)
                    
                    D = np.dot(np.transpose(J),J) # Green-Cauchy tensor
                    lamda = np.linalg.eigvals(D) # its largest eigenvalue
                    FTLE[i][j] = max(lamda)
            return FTLE

        start_time = time.time()
        mappingfile = open(self.mappingfile,'r')
        X,Y = np.loadtxt(mappingfile,unpack = True)
        mappingfile.close()

        # scatter plot
        plt.figure(1)
        plt.scatter(X,Y)

        # FTLE plot
        fig = plt.figure(2)
        FTLE = Jacobian(X,Y)
        FTLE = np.log(FTLE)

        ax = plt.subplot(111)
        im = ax.imshow(FTLE)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",size = "5%",pad = 0.1)
        plt.colorbar(im, cax = cax)
        #plt.gca().invert_yaxis()
        self.total_time = format(time.time()-start_time, '2.2f')

        if title is None:
            ax.set_title("res: {}x{}, seconds: {}".format(self.length, self.height, self.total_time))
        else:
            ax.set_title("{}".format(title))
        if filename is None:
            fig.savefig('../figures/{}x{}.pdf'.format(self.length, self.height), format='pdf')
        else:
            fig.savefig('../figures/{}'.format(filename), format='pdf')

        if display:
            plt.show()

    def testVelocityDiff(self):
        x = np.random.rand(100)
        y = np.random.rand(100)
        print( np.mean(np.subtract(self.getVelocity(x,y,20), self.getEstimatedVelocity(x,y,20))) ) 

def main():

    datafolder = '../data'
    mappingfile = datafolder + '/kevin_output_steps.txt'
    test = FTLE(length=100, height=50, percentLength=0.1, percentHeight=0.1, estimateVelocity=True)
    # test.createMapping(mappingfile)
    # test.plotter(display=True, title='test', filename='TESTS')
    test.testVelocityDiff()

if __name__ == '__main__':
    main()