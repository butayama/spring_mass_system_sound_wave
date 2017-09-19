import numpy as np
import matplotlib.pyplot as plt
"""
A program to simulate to see the vibration/traveling wave of N masses connected with springs
with a forced harmonic motion force at the 0 mass.

author = Carlos Vargas Aguero
email = carlos.vargasaguero@gmail.com
"""

"""
Please, run the program on Python 3, not on Python 2, in Python 2 sometimes gives some problems with the RK4 method, I don't know where. Maybe the line t_steps = int(t_f/delta_t), but who knows. It send the masses to position e265 sometimes, 
and Python 3 with the same variables made it right.
"""

#Functions of the program

def read_variables():
    """
    read the variable of the problem in the file input.txt
    """
    f = open('input.txt','r')
    f.readline() #Skip the first line, which is an instruction on the txt file.
    var_list = [float(line.split(' ')[2]) for line in f.readlines()]
    f.close()
    return var_list

def f_n_in(x_n,x_next,x_before):
    """
    To calculate the force on the n mass, with 0 < n < n_balls
    The k_spr is not included because it was factored out
    """
    return -2*x_n + x_next + x_before

def f_n_out(x_n,x_bef):
    """
    To calculate the force on the last ball in the system
    The k_spr is not included because it was factored out
    """
    return -x_n + x_bef

def rk4_mass_spring_system(amp,omega,k_spr_m,n_balls,t_f,delta_t):
    """
    To obtain the motion of a coupled n_balls mass spring system
    of k_spr of the constant of the spring and mass m_b.
    There the first ball is forced in a harmonic motions
    with harmonic motion of amplitud amp and omega frecuency.
    The displacement from equilibrium position and velocities,
    x_n and v_n are all 0 at t = 0.
    The motion is made from t = 0 to tf in steps of delta_t
    Runge Kutta 4 method is used to produce the position x(t)
    The final mass is free, without a spring at the end
    """

    t_steps = int(t_f/delta_t)

    t = np.arange(0,t_f,delta_t)
    x = np.empty([n_balls, t_steps])
    v = np.empty([n_balls, t_steps])

    #k factors of Runge Kutta 4
    kx = np.empty([4,n_balls])
    kv = np.empty([4,n_balls])

    #Initial Conditions
    x[:,0] = 0.0
    v[:,0] = 0.0

    #Motion of the 0 mass
    x[0,:] = amp*np.sin(omega*t)*(1-0.5*(np.sign(t-5)+1.0))
    #    v[0,:] = omega*amp*np.sin(omega*t)

    #Only the proportion between k_spr and m appears, not k_spr or m_b alone
    #    k_spr_m = k_spr/m_b

    for jt in range(t_steps-1):

        #k1 factors
        for n in range(1,n_balls):
            if n <= (n_balls-2):
                kx[0,n] = delta_t*v[n,jt]
                kv[0,n] = delta_t*(k_spr_m)*f_n_in(x[n,jt], x[n+1,jt], x[n-1,jt])
            elif n == (n_balls-1):
                kx[0,n] = delta_t*v[n,jt]
                kv[0,n] = delta_t*(k_spr_m)*f_n_out(x[n,jt], x[n-1,jt])

        #k2 factors
        for n in range(1,n_balls):
            if n <= (n_balls-2):
                kx[1,n] = delta_t*(v[n,jt]+kv[0,n])
                kv[1,n] = delta_t* (k_spr_m)*f_n_in(x[n,jt]+0.5*kx[0,n], x[n+1,jt]+0.5*kx[0,n+1], x[n-1,jt]+0.5*kx[0,n-1])
            elif n == (n_balls-1):
                kx[1,n] = delta_t*(v[n,jt]+kv[0,n])
                kv[1,n] = delta_t*(k_spr_m)*f_n_out(x[n,jt]+0.5*kx[0,n], x[n-1,jt]+0.5*kx[0,n-1])

        #k3 factors
        for n in range(1,n_balls):
            if n <= (n_balls-2):
                kx[2,n] = delta_t*(v[n,jt]+kv[1,n])
                kv[2,n] = delta_t* (k_spr_m)*f_n_in(x[n,jt]+0.5*kx[1,n], x[n+1,jt]+0.5*kx[1,n+1], x[n-1,jt]+0.5*kx[1,n-1])
            elif n == (n_balls-1):
                kx[2,n] = delta_t*(v[n,jt]+kv[1,n])
                kv[2,n] = delta_t* (k_spr_m)*f_n_out(x[n,jt]+0.5*kx[1,n],x[n-1,jt]+0.5*kx[1,n-1])

        #k4 factors
        for n in range(1,n_balls):
            if n <= (n_balls-2):
                kx[3,n] = delta_t*(v[n,jt]+kv[2,n])
                kv[3,n] = delta_t* (k_spr_m)*f_n_in(x[n,jt]+kx[2,n],x[n+1,jt]+0.5*kx[2,n+1],x[n-1,jt]+0.5*kx[2,n-1])
            elif n == (n_balls-1):
                kx[3,n] = delta_t* (v[n,jt]+kv[2,n])
                kv[3,n] = delta_t* (k_spr_m)*f_n_out(x[n,jt]+kx[2,n],x[n-1,jt]+kx[2,n-1])

        #next position/velocity

        for n in range(1,n_balls):
            x[n,jt+1] = x[n,jt] + (kx[0,n]+2*kx[1,n]+2*kx[2,n]+kx[3,n])/6.0
            v[n,jt+1] = v[n,jt] + (kv[0,n]+2*kv[1,n]+2*kv[2,n]+kv[3,n])/6.0

    del(kx,kv,v)
    return t_steps,t,x

def plot_masses(t,x,n_balls,k_spr_m):
    for i in range(n_balls):
        plt.plot(t, x[i,:])
    plt.xlabel('time')
    plt.ylabel('Position')
    plt.title('Position of the masses')
    plt.savefig('positions_k_spr_m_{}.svg'.format(k_spr_m))

def main():
    #The x obtained is the displacement, when making the animation
    #we need the real position, not displacement
    #dist_balls is the distance between the equilibrium position
    #between neighbor balls
    
    #The variables are always read in that order
    amp, dist_balls, omega, k_spr_m, n_balls, t_f, delta_t = read_variables()
    # amp        =  Amplitude of the forced simple armonic motion x0 ball
    # dist_balls =  The distance between each neighbor ball. It is used in the animation
    # omega      =  Angular frecuency of the forced simple armonic motion x0 ball
    # k_spr_m    =  Relation between k of the spring and mass (Only the proportion appears) Not individually
    # n_balls    =  Number of balls
    # t_f        =  Final time
    # delta_t    =  Time spacing of the RK4 method

    n_balls = int(n_balls)
    t_steps, t,x = rk4_mass_spring_system(amp,omega,k_spr_m,n_balls,t_f,delta_t)

    #For plotting, it is used the real position, not the displacement from equilibrium
    #x is now going to be the real position instead of displacement
    for i in range(t_steps):
        x[:,i] = x[:,i] + dist_balls * np.arange(0,n_balls,1)
    plot_masses(t,x,n_balls,k_spr_m)

if __name__ == '__main__':
    main()
