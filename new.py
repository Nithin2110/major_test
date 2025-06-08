import random
l=[]
for i in range(100):
    x=random.random()
    l.append(round(x*100))
d={}
for i in l:
    if i not in d:
        d[i]=1
    else:
        d[i]+=1
popular=max(d,key=d.get)
print("frequent number is ",popular)

print("next version")

#version 1-1
#cc2b7fd00adaed2061d2df2a9d6c080931e23230