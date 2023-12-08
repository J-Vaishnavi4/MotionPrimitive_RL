import matplotlib.pyplot as plt 
import csv 
  
rew = [] 
x=[]
i=0
  
with open('./reward/reward_BMP.csv','r') as csvfile: 
    plots = csv.reader(csvfile, delimiter = ',') 
    
    for row in plots: 
        rew.append(float(row[0]))
        x.append(i)
        i+=1

print(i)
print(len(rew))
plt.plot(i,rew) 
plt.show()