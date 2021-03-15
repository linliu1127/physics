#this method is from griffith problem 2.61
import numpy as np
import matplotlib.pyplot as plt
N=100
H=np.zeros((N,N))
vj=8/((N+1)**2)
l=(N+1)**2/(np.pi**2)
for i in range(0,N):
    for k in range(0,N):
        if i==k:
            if i<=(N+1)/2 :
                H[i][k]=2+vj
            else:
                H[i][k]=2
        if k==i-1 or k==i+1:
            H[i][k]=-1
E,v=np.linalg.eig(H) 

D=v.T@H@v
grounds=np.argmin(E)
print(E[grounds],grounds)
groundv=v[:,grounds]
x=np.arange(N)
p1=plt.plot(x,-groundv[0:],label="numerical")
perturb=(2/100)**(0.5)*(np.sin(np.pi*x/N)+(32/np.pi**3)*((-1/3**2)*np.sin(2*np.pi*x/N)+(2/15**2)*np.sin(4*np.pi*x/N)-(3/35**2)*np.sin(6*np.pi*x/N)))
nperturb=(2/100)**(0.5)*(np.sin(np.pi*x/N))
p2=plt.plot(x,nperturb,label="nonperturbed")
p3=plt.plot(x,perturb,label="perturbed")
plt.legend(loc="upper left")
plt.show()