import random as r
import math as m
r.seed(2)
def q2(x, y, z):#defining the ellipsoid
	a=1
	b=1.5
	c=2
	f=(x/a)**2+(y/b)**2+(z/c)**2
	return f
def monte_carlo(f,n):
	count=0
	points=[[],[],[]]
	for i in range(n):
		x=-1+2*r.random()#x can be any number between [-1,1]
		y=-1.5+3*r.random()#y can be any number between [-1.5,1.5]
		z=-2+4*r.random()#z can be any number between [-2,2]
		if f(x,y,z)<=1:
			points[0].append(x)
			points[1].append(y)
			points[2].append(z)
		else:
			continue
	volume=24*len(points[0])/n#volume=p(a point lies in the ellipsoid)*volume of cuboid	
	return volume,points#returning volume


N=[100,500,1000,5000,10000,15000,20000,30000,40000,50000]
arr=[[],[],[]]#arr=[[number of steps],[volume found],[fractional error]]
for i in range(len(N)):
	volume, data = monte_carlo(q2,N[i])
	frac_err=abs(4*m.pi-volume)/4*m.pi
	arr[0].append(N[i])
	arr[1].append(volume)
	arr[2].append(frac_err)
	
with open("result.txt", 'w') as f:
	for i in range(len(arr[0])):
		f.write(str(arr[0][i])+'	'+str(arr[1][i])+'	'+str(arr[2][i])+'\n')
		
volume,data=monte_carlo(q2,30000)		
with open("point.txt", 'w') as f:
	for i in range(len(data[0])):
		f.write(str(data[0][i])+'	'+str(data[1][i])+'	'+str(data[2][i]))
		f.write('\n')
		
		
print("Analytical volume of the ellipsoid =",4*m.pi,"\n")	
print("Monte Carlo values:")    
for i in  range(len(N)):
	print("With",N[i],"points","volume is",arr[1][i])
	 
	
