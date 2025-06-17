import matplotlib.pyplot as plt
import numpy as np
import math

def get_plane_equation(p):
    #input plane, return coefficients for plane equation in the form ax+by+cz=d
    p0, p1, p2 = p
    p0=np.asarray(p0)
    p1=np.asarray(p1)
    p2=np.asarray(p2)

    # These two vectors are in the plane
    v1 = p2 - p0
    v2 = p1 - p0

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p2)

    print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))

    return a, b, c, d


def get_changeofbasis(p):
    #input plane, return tranlsation and rotation matrices
    #https://math.stackexchange.com/questions/1167717/transform-a-plane-to-the-xy-plane
    a,b,c,d=get_plane_equation(p)

    t=np.zeros((3,3))
    if(c!=0):
        t[:,2]=np.transpose(np.array([-d/c, -d/c, -d/c]))
    else:
        t[:,2]=np.transpose(np.array([-d, -d, -d]))
    
    co=c/math.sqrt(a**2+b**2+c**2)
    si=math.sqrt((a**2+b**2)/(a**2+b**2+c**2))
    u1=b/math.sqrt(a**2+b**2)
    u2=-a/math.sqrt(a**2+b**2)

    R=np.array([[co+u1**2*(1-co), u1*u2*(1-co), u2*si],
            [u1*u2*(1-co), co+u2**2*(1-co), -u1*si],
            [-u2*si, u1*si, co]])

    return t, R


def plot_3d(points):
#Plots a 3d plane given 3 points to define the plane

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    i=0
    col=['red', 'blue', 'green', 'pink']

    for p in points:
        p0, p1, p2, label = p
        p=[p0,p1,p2]

        #set new basis for main plane, everything else will move around it
        if i == 0:
            t, R = get_changeofbasis(p)

        #translate
        p=p-t
        #rotate
        p=p@R

        #reassign coordinates in new basis
        p0, p1, p2 = p

        #get new equation for plane
        a, b, c, d = get_plane_equation(p)

        p=np.asarray(p)
        x = np.linspace(p[:,0].min(), p[:,0].max(), 5)
        y = np.linspace(p[:,1].min(), p[:,1].max(), 5)
        X, Y = np.meshgrid(x, y)

        if(c != 0):
            Z = (d - a * X - b * Y) / c
        else:
            Z = (d - a * X - b * Y)
        
        ax.plot_surface(X, Y, Z, alpha=0.75, color=col[i])
        ax.plot(*zip(p0, p1, p2), color=col[i], label=label, linestyle=' ', marker='o')
        
        i+=1
        
    ax.view_init(0, 22)
    plt.tight_layout()
    plt.show()
    ax.legend()