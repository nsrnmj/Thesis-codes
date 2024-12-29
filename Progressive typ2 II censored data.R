‎n=20;m=15;a=1.7;b=1;c=1.5;R=c(n-m,c(rep(0,m-1)));theta=c(a,b,c);V=U=c()‎

‎for(j in 1:m){‎
‎for(i in 1:m){‎
‎W=sort(runif(m,min=0,max=1))‎
‎V[i]=W[i]^(1/(i+(sum(R[(m-i+1):m]))))}‎
‎U[j]=1-prod(V[(m-j+1):m])}‎
‎y=((-1/b)*log(1-(U^(1/a))))^(1/c)‎
‎print(y)‎
