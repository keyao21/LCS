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

    def createMapping(self, dt=None, elapsedTime=None):
        if dt is None:
            dt = self.dt
        if elapsedTime is None:
            elapsedTime = self.elapsedTime

        output = open('data/kevin_output_steps.txt','ab')

        for i in np.linspace(0.01, 0.99, self.height, np.float64):
            self.state = np.zeros((self.length, 2))  # (xposition, yposition)
            self.state[:,0] = np.linspace(0.01, 1.99, self.length, np.float64)
            self.state[:,1] = i*np.ones(self.length, np.float64)

            for t in np.arange(0,elapsedTime,dt):
                self.state = self.rk4(self.state,t,dt)
                
            np.savetxt(output,self.state)

def main():
    test = FTLE()
    test.createMapping()
    test.state

if __name__ == '__main__':
    main()