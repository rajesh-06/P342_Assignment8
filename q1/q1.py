import random as r
import math as m
import mymodule as mm
r.seed(3)
xdata=[0]
ydata=[0]

def write(filename,x,y):
	a=[]
	for i in range(5):#choosing 5 random walks out of 100
		a.append(r.randint(0,99))
	with open(filename,"w") as f:
		for i in range(len(x[0])):
			for j in range(5):
				f.write(str(x[a[j]][i])+"	"+str(y[a[j]][i])+"	")
			f.write("\n")
	return a
	
N=[250,500,1000,2500,5000]#5 step taken
filename=["250.txt","500.txt","1000.txt","2500.txt","5000.txt"]
rx=[0 for i in range(5)]
ry=[0 for i in range(5)]
for i in range(5):
	x1,y1,rx[i],ry[i]=mm.creat_walk(1,N[i])
	a1=write(filename[i],x1,y1)

r=[[],[],[],[],[]]
r_rms2=[0 for i in range(5)]
rms=[0 for i in range(5)]
d=[0 for i in range(5)]
for j in range(5):
	for i in range(100):
		d=(rx[j][i]**2+ry[j][i]**2)**0.5
		r[j].append(d)
		r_rms2[j]+=d**2/100

sqrtN=[]
for i in range(5):
	rms[i]=r_rms2[i]**0.5
	sqrtN.append(N[i]**0.5)
with open("rms.txt","w") as f:
	for i in range(5):
		f.write(str(sqrtN[i])+"	"+str(rms[i])+"\n")


for i in range(5):
	print("For random walk for",N[i],"steps")
	print("R_rms = ",rms[i])
	print("Average x displacement = ",sum(rx[i])/100)
	print("Average y displacement = ",sum(ry[i])/100)
	print("Average distance = ",sum(r[i])/100,"\n")
	
