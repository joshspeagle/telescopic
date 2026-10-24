import math, random
random.seed(2)
def direct(hX,hY,k,m,tau,L):
    tot=0.0
    for t in range(k,m+1):
        s=hX[t]; j=1
        while t+j*L<=tau-1:
            s+=hX[t+j*L]-hY[t+(j-1)*L]; j+=1
        tot+=s
    return tot/(m-k+1)
def floor_form(hX,hY,k,m,tau,L):
    mc=sum(hX[t] for t in range(k,m+1))/(m-k+1)
    return mc+sum(min(1.0, math.floor((s-k)/L)/(m-k+1))*(hX[s]-hY[s-L]) for s in range(k+L,tau))
e_small=0; e_big=0
for trial in range(4000):
    L=random.randint(1,7); k=random.randint(0,15); m=k+random.randint(0,30); tau=L+1+random.randint(0,60)
    T=max(tau,m)+5*L+5
    hX=[random.gauss(0,1) for _ in range(T+1)]; hY=[random.gauss(0,1) for _ in range(T+1)]
    for t in range(tau,T+1):
        hX[t]=hY[t-L]
    err=abs(direct(hX,hY,k,m,tau,L)-floor_form(hX,hY,k,m,tau,L))
    if tau<=m+L+1: e_small=max(e_small,err)
    else: e_big=max(e_big,err)
print("floor form error when tau <= m+L+1:", e_small)
print("floor form error when tau >  m+L+1:", e_big)
