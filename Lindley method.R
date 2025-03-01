‎L1=function(theta){‎
‎a=theta[1];b=theta[2];c=theta[3]‎
‎for(j in 1:m){‎ 
‎for(i in 1:m){‎
‎W=sort(runif(m,min=0,max=1))‎
‎V[i]=W[i]^(1/(i+(sum(R[(m-i+1):m]))))}‎
‎U[j]=1-prod(V[(m-j+1):m])}‎
‎y=((-1/b)*log(1-(U^(1/a))))^(1/c)‎
‎aa1=(m*log(a))+(m*log(b))+(m*log(c))+((c-1)*sum(log(y)))-(b*sum(y^c))+((a-1)*‎
‎sum(log(1-exp(-b*(y^c)))))+sum(R*log(1-(1-exp(-b*(y^c)))^a))‎
‎return(-aa1)}‎

‎dL1=function(theta){‎
‎a=theta[1];b=theta[2];c=theta[3]‎
‎for(j in 1:m){‎
‎for(i in 1:m){‎
‎W=sort(runif(m,min=0,max=1))‎
‎V[i]=W[i]^(1/(i+(sum(R[(m-i+1):m]))))}‎
‎U[j]=1-prod(V[(m-j+1):m])}‎
‎y=((-1/b)*log(1-(U^(1/a))))^(1/c)‎
‎a1=m/a‎ + ‎sum(-(R * (1‎ - ‎exp(-(b * y^c)))^a * log(1‎ - ‎exp(-(b * y^c)))/(1‎ - 
    ‎(1‎ - ‎exp(-(b * y^c)))^a)))‎ + ‎sum(log(1‎ - ‎exp(-(b * y^c))))‎
‎a2=(a‎ - ‎1) * sum(y^c * exp(-(b * y^c))/(1‎ - ‎exp(-(b * y^c))))‎ + 
    ‎m/b‎ + ‎sum(-(a * R * y^c * (1‎ - ‎exp(-(b * y^c)))^(a‎ - ‎1) *‎ 
    ‎exp(-(b * y^c))/(1‎ - ‎(1‎ - ‎exp(-(b * y^c)))^a)))‎ - ‎sum(y^c)‎
‎a3=(a‎ - ‎1) * sum(b * y^c * exp(-(b * y^c)) * log(y)/(1‎ - ‎exp(-(b *‎ 
    ‎y^c))))‎ + ‎m/c‎ + ‎sum(-(a * b * R * y^c * (1‎ - ‎exp(-(b * y^c)))^(a‎ - 
    ‎1) * exp(-(b * y^c)) * log(y)/(1‎ - ‎(1‎ - ‎exp(-(b * y^c)))^a)))‎ + 
    ‎sum(log(y))‎ - ‎b * sum(y^c * log(y))‎
‎c(a1,a2,a3)}‎

‎lindley=function(){‎
‎a=theta[1];b=theta[2];c=theta[3]‎
‎for(j in 1:m){‎
‎for(i in 1:m){‎
‎W=sort(runif(m,min=0,max=1))‎
‎V[i]=W[i]^(1/(i+(sum(R[(m-i+1):m]))))}‎
‎U[j]=1-prod(V[(m-j+1):m])}‎
‎y=((-1/b)*log(1-(U^(1/a))))^(1/c)‎
‎a=optim(par=theta,fn=L1,gr=dL1)$par[1];b=optim(par=theta,fn=L1,gr=dL1)$par[2]‎
‎c=optim(par=theta,fn=L1,gr=dL1)$par[3]‎
‎aa=optim(par=theta,fn=L1,hessian=TRUE)$hessian‎
‎#aa=optimHess(par=theta,fn=L1)‎
‎rt=solve(aa);rt=-rt‎
‎f <‎- ‎function(a,b,c) (m*log(a))+(m*log(b))+(m*log(c))+((c-1)*‎
                             ‎sum(log(y)))-(b*sum(y^c))+((a-1)*sum(log(1-exp(-b*(y^c)))))‎+
                             ‎sum(R*log(1-(1-exp(-b*(y^c)))^a))‎
‎L1=Deriv(f‎, ‎"a",‎ ‎cache.exp=FALSE);L2=Deriv(f‎, ‎"b",‎ ‎cache.exp=FALSE)‎
‎L3=Deriv(f‎, ‎"c",‎ ‎cache.exp=FALSE)‎
‎L11=Deriv(L1‎, ‎"a",‎ ‎cache.exp=FALSE);L22=Deriv(L2‎, ‎"b",‎ ‎cache.exp=FALSE)‎
‎L33=Deriv(L3‎, ‎"c",‎ ‎cache.exp=FALSE);L12=Deriv(L1‎, ‎"b",‎ ‎cache.exp=FALSE);‎
‎L13=Deriv(L1‎, ‎"c",‎ ‎cache.exp=FALSE);L23=Deriv(L2‎, ‎"c",‎ ‎cache.exp=FALSE)‎
‎L111=Deriv(L11‎, ‎"a",‎ ‎cache.exp=FALSE);L112=Deriv(L11‎, ‎"b",‎ ‎cache.exp=FALSE)‎
‎L113=Deriv(L11‎, ‎"c",‎ ‎cache.exp=FALSE);L122=Deriv(L12‎, ‎"b",‎ ‎cache.exp=FALSE)‎
‎L123=Deriv(L12‎, ‎"c",‎ ‎cache.exp=FALSE);L133=Deriv(L13‎, ‎"c",‎ ‎cache.exp=FALSE)‎
‎L222=Deriv(L22‎, ‎"b",‎ ‎cache.exp=FALSE);L223=Deriv(L22‎, ‎"c",‎ ‎cache.exp=FALSE)‎
‎L233=Deriv(L23‎, ‎"c",‎ ‎cache.exp=FALSE);L333=Deriv(L33‎, ‎"c",‎ ‎cache.exp=FALSE)‎
‎L1=L1(a,b,c);L2=L2(a,b,c);L3=L3(a,b,c);L11=L11(a,b,c);L22=L22(a,b,c);L33=L33(a,b,c)‎
‎L12=L21=L12(a,b,c);L13=L31=L13(a,b,c);L23=L32=L23(a,b,c);L111=L111(a,b,c)‎
‎L112=L121=L211=L112(a,b,c);L113=L131=L311=L113(a,b,c);L122=L221=L212=L122(a,b,c)‎
‎L123=L132=L213=L231=L321=L312=L123(a,b,c);L133=L313=L331=L133(a,b,c)‎
‎L222=L222(a,b,c);L223=L232=L322=L223(a,b,c);L323=L332=L233=L233(a,b,c);L333=L333(a,b,c)‎
‎p1=((b1-1)/a)-b2;p2=((b3-1)/b)-b4;p3=((b5-1)/c)-b6‎
‎s11=rt[1,1];s22=rt[2,2];s33=rt[3,3];s12=s21=rt[1,2];s13=s31=rt[1,3];s23=s32=rt[2,3]‎
‎A=(s11*L111)+2*(s12*L121)+2*(s13*L131)+2*(s23*L231)+(s22*L221)+(s33*L331)‎
‎B=(s11*L112)+2*(s12*L122)+2*(s13*L132)+2*(s23*L232)+(s22*L222)+(s33*L332)‎
‎C=(s11*L113)+2*(s12*L123)+2*(s13*L133)+2*(s23*L233)+(s22*L223)+(s33*L333)‎
‎#################################################### square‎
‎u1=u2=u3=1‎
‎q1=a+(u1*((p1*s11+p2*s12+p3*s13)+(.5*(A*s11+B*s21+C*s31))))‎
‎q2=b+(u2*((p1*s21+p2*s22+p3*s23)+(.5*(A*s12+B*s22+C*s32))))‎
‎q3=c+(u3*((p1*s31+p2*s32+p3*s33)+(.5*(A*s13+B*s23+C*s33))))‎
‎#################################################### linex‎
‎u1=-h*exp(-h*a);u11=(h^2)*exp(-h*a)‎
‎m1=exp(-h*a)+u1*((p1*s11+p2*s12+p3*s13)+.5*(A*s11+B*s21+C*s31))+.5*u11*s11‎
‎q4=(-1/h)*log(m1);q4=c(q4)‎
‎##‎
‎k2=-h*exp(-h*b);k22=(h^2)*exp(-h*b)‎
‎m2=exp(-h*b)+k2*((p1*s21+p2*s22+p3*s23)+.5*(A*s12+B*s22+C*s32))+.5*k22*s22‎
‎q5=(-1/h)*log(m2);q5=c(q5)‎
‎##‎
‎j3=-h*exp(-h*c);j33=(h^2)*exp(-h*c)‎
‎m3=exp(-h*c)+j3*((p1*s31+p2*s32+p3*s33)+.5*(A*s13+B*s23+C*s33))+.5*j33*s33‎
‎q6=(-1/h)*log(m3);q6<-c(q3)‎
‎#q1=SB-alpha‎, ‎q2=SB-beta‎, ‎q3=LB-alpha‎, ‎q4=LB-beta‎
‎result<-c(q1,q2,q3,q4,q5,q6);result}‎
