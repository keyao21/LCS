import numpy as np

class FTLE():
    def __init__(self):
        self.delta = 1e-5
        self.earthRotation = 1
        # lattitude
        self.phi = np.pi*40.7128/180
        self.coriolis = 2*self.earthRotation*np.sin(self.phi)
        self.dimensions = 2
        self.length = 100
        self.height = 50
        self.state = np.zeros((self.length,4))
        self.dt = 0.1
        self.elapsedTime = 20
        self.mappingfile = None

    def velocity(self, x, y, t, a = 0.1, epsilon = 0.25, w = 0.2*np.pi, dims = 2):
        f = epsilon*np.sin(w*t)*x**2 + (1-2*epsilon*np.sin(w*t))*x
        if dims == 2:
            velocity_x = a*np.pi*np.sin(np.pi*f)*np.cos(np.pi*y)
            velocity_y = -1*a*np.pi*np.cos(np.pi*f)*np.sin(np.pi*y)*(2*epsilon*np.sin(w*t)*x 
                + (1-2*epsilon*np.sin(w*t)))
        elif dims == 3:
            print( "not sure if we can do that yet!")
        # return np.vstack([velocity_x,velocity_y])
        return velocity_x, velocity_y

    def update(self, state, t):
        x = state[:,0]
        y = state[:,1]
        vx, vy = self.velocity(x, y, t)
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
        state[:,0] = np.clip(tmp_state[:,0],0.00001,1.99999)
        state[:,1] = np.clip(tmp_state[:,1],0.00001,0.99999)

        #noise = B*np.random.normal(u,sigma,(L,2))
        #state[:,4:6] += noise
        return state

    def createMapping(self, mappingfile=None, dt=None, elapsedTime=None):
        if dt is None:
            dt = self.dt
        if elapsedTime is None:
            elapsedTime = self.elapsedTime
        if mappingfile is None:
            mappingfile = self.mappingfile
        else:
            self.mappingfile = mappingfile

        # output = open('data/kevin_output_steps.txt','ab')
        output = open(mappingfile,'ab')

        for i in np.linspace(0.01, 0.99, self.height, np.float64):
            self.state = np.zeros((self.length, 2))  # (xposition, yposition)
            self.state[:,0] = np.linspace(0.01, 1.99, self.length, np.float64)
            self.state[:,1] = i*np.ones(self.length, np.float64)

            for t in np.arange(0,elapsedTime,dt):
                self.state = self.rk4(self.state,t,dt)
                
            np.savetxt(output,self.state)

    def plotter(self):
 
        def Jacobian(X,Y):
            dx = 1.98/(self.length-1) # horizontal grid difference
            dy = 0.98/(self.height-1) # verticle grid difference
            J = np.empty([2,2],float)
            FTLE = np.empty([H-2,L-2],float)
            
            for i in range(0,H-2):
                for j in range(0,L-2):
                    J[0][0] = (X[(1+i)*L+2+j]-X[(1+i)*L+j])/(2*dx)
                    J[0][1] = (X[(2+i)*L+1+j]-X[i*L+1+j])/(2*dx)
                    J[1][0] = (Y[(1+i)*L+2+j]-Y[(1+i)*L+j])/(2*dy)
                    J[1][1] = (Y[(2+i)*L+1+j]-Y[i*L+1+j])/(2*dy)
                    
                    D = np.dot(np.transpose(J),J) # Green-Cauchy tensor
                    lamda = LA.eigvals(D) # its largest eigenvalue
                    FTLE[i][j] = max(lamda)
            return FTLE

        mappingfile = open(self.mappingfile,'r')
        X,Y = np.loadtxt(mappingfile,unpack = True)
        mappingfile.close()

        # scatter plot
        plt.figure(1)
        plt.scatter(X,Y)

        # FTLE plot
        plt.figure(2)
        FTLE = Jacobian(X,Y)
        FTLE = np.log(FTLE)

        ax = plt.subplot(111)
        im = ax.imshow(FTLE)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",size = "5%",pad = 0.1)
        plt.colorbar(im, cax = cax)
        #plt.gca().invert_yaxis()

        print(time.time()-start_time)
        plt.show()


def main():
    datafolder = 'data'
    mappingfile = datafolder + '/kevin_output_steps.txt'
    test = FTLE()
    test.createMapping(mappingfile)

if __name__ == '__main__':
    main()