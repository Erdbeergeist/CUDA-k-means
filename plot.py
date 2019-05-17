import numpy as np
import matplotlib.pyplot as plt

data = []

x = []
y = []


with open("log.txt", "r") as ins:
    for line in ins:
        data.append(line)

loopnr = 0
clusternr = 0
for i in range (len(data)):
	if (data[i][0]=="#"): 
		loopnr = int(data[i][8:])
		if (loopnr>1):
			plt.savefig("Plots/L"+str(loopnr-1)+".png")
			x = []
			y = []
		continue
	if (data[i][0]=="C"):
		clusternr = int(data[i][8:])
		continue
	if (data[i][0]=="P"):
		clusterx  = float(data[i].split()[1])
		clustery  = float(data[i].split()[2])
		plt.plot(clusterx,clustery,marker="x",markersize=10,color="black")
		plt.xlim(0,100)
		plt.ylim(0,100)
		continue	
	if (data[i][:4]=="Done"):
		plt.plot(x,y,marker=".",linestyle ="none")
		plt.xlim(0,100)
		plt.ylim(0,100)
		plt.savefig("Plots/L"+str(loopnr)+".png")
		break
		
	if (data[i][0]=="!"):
		plt.plot(x,y,marker=".",linestyle ="none")
		plt.xlim(0,100)
		plt.ylim(0,100)
		x = []
		y = []
		continue
	print(data[i])	
	x.append(data[i].split()[0])
	y.append(data[i].split()[1])
			


