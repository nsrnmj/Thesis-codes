‎library(ggplot2)‎


‎###############################‎ 
‎h1=function(x,a=1,b=8,c=1‎, ‎d=0.8)  ((a*b*(x^(b-1)))‎+ ‎(c*d*(x^(d-1))))*exp(-a*(x^b)‎ -‎c*(x^d))‎
‎h2=function(x,a=1,b=6,c=1‎, ‎d=4)    ((a*b*(x^(b-1)))‎+ ‎(c*d*(x^(d-1))))*exp(-a*(x^b)‎ -‎c*(x^d))‎
‎h3=function(x,a=1,b=0.8,c=1‎, ‎d=0.4)((a*b*(x^(b-1)))‎+ ‎(c*d*(x^(d-1))))*exp(-a*(x^b)‎ -‎c*(x^d))‎
‎h4=function(x,a=1,b=3,c=1‎, ‎d=2)    ((a*b*(x^(b-1)))‎+ ‎(c*d*(x^(d-1))))*exp(-a*(x^b)‎ -‎c*(x^d))‎
‎h5=function(x,a=1,b=2,c=1‎, ‎d=1.2)  ((a*b*(x^(b-1)))‎+ ‎(c*d*(x^(d-1))))*exp(-a*(x^b)‎ -‎c*(x^d))‎
‎h6=function(x,a=1,b=3,c=1‎, ‎d=0.2)  ((a*b*(x^(b-1)))‎+ ‎(c*d*(x^(d-1))))*exp(-a*(x^b)‎ -‎c*(x^d))‎


‎############################### gglot of probability distribution function‎
‎p = ggplot(data.frame(x=c(0,1.5)),aes(x))+stat_function(fun=h1,lty=1,aes(colour = 'blue4')‎,
 ‎linewidth=0.5)‎+ ‎xlab("x")‎ + ‎ylab("f(x)")‎+
‎stat_function(fun=h2,lty=1,aes(colour = 'red3')‎, ‎linewidth=1)‎+
‎stat_function(fun=h3,lty=1,aes(colour = 'blue')‎, ‎linewidth=1)‎+
‎stat_function(fun=h4,lty=1,aes(colour = 'cyan3')‎, ‎linewidth=1)‎+
‎stat_function(fun=h5,lty=1,aes(colour = 'yellow')‎, ‎linewidth=1)‎+
‎stat_function(fun=h6,lty=1,aes(colour = 'green')‎, ‎linewidth=1)‎+
 ‎xlab("x")‎ + ‎ylab("f(x)")‎+ ‎ylim(0‎, ‎2.5)‎+
 ‎scale_colour_manual(values = c('blue4'‎ ,‎'red3'‎ ‎,'blue','cyan3','yellow','green'),‎ 
          ‎labels = c("b=8,d=0.8"‎, ‎"b=6,d=4","b=0.8,d=0.4","b=3,d=2","b=2,d=1.2","b=3,d=0.2"))+‎
‎theme(‎
    ‎legend.position = c(1.02,0.7)‎,
    ‎legend.justification = c("right"‎, ‎"bottom"),‎
    ‎legend.box.just =‎ ‎"right",‎
    ‎legend.margin = margin(1,1,1,1)‎,
    ‎legend.title=element_blank()‎,
   ‎# Remove panel border‎
  ‎panel.border = element_blank()‎,  
  ‎# Remove panel grid lines‎
  ‎panel.grid.major = element_blank()‎,
  ‎panel.grid.minor = element_blank()‎,
  ‎# Remove panel background‎
  ‎panel.background = element_blank()‎,
  ‎# Add axis line‎
  ‎axis.line = element_line(colour =‎ "‎black")‎
  )
‎#+labs(caption="Hazard rate plot for Additive Weibull distribution")‎


‎# Adjust size and resolution‎
‎ggsave("pdf1.png"‎, ‎plot = p‎, ‎width = 6‎, ‎height = 6‎, ‎dpi = 900)‎

‎###############################‎ 
‎h1=function(x,a=1,b=8,c=1‎, ‎d=0.8)  (a*b*(x^(b-1)))‎+ ‎(c*d*(x^(d-1)))‎
‎h2=function(x,a=1,b=6,c=1‎, ‎d=4)  (a*b*(x^(b-1)))‎+ ‎(c*d*(x^(d-1)))‎
‎h3=function(x,a=1,b=0.8,c=1‎, ‎d=0.4)  (a*b*(x^(b-1)))‎+ ‎(c*d*(x^(d-1)))‎
‎h4=function(x,a=1,b=3,c=1‎, ‎d=2)  (a*b*(x^(b-1)))‎+ ‎(c*d*(x^(d-1)))‎
‎h5=function(x,a=1,b=2,c=1‎, ‎d=1.2)  (a*b*(x^(b-1)))‎+ ‎(c*d*(x^(d-1)))‎
‎h6=function(x,a=1,b=3,c=1‎, ‎d=0.2)  (a*b*(x^(b-1)))‎+ ‎(c*d*(x^(d-1)))‎

‎############################### gglot of hazard function‎
‎p = ggplot(data.frame(x=c(0.01,1)),aes(x))+stat_function(fun=h1,lty=1,aes(colour = 'blue4')‎,
 ‎linewidth=0.5)‎+ ‎xlab("x")‎ + ‎ylab("h(x)")‎+
‎stat_function(fun=h2,lty=1,aes(colour = 'red3')‎, ‎linewidth=1)‎+
‎stat_function(fun=h3,lty=1,aes(colour = 'blue')‎, ‎linewidth=1)‎+
‎stat_function(fun=h4,lty=1,aes(colour = 'cyan3')‎, ‎linewidth=1)‎+
‎stat_function(fun=h5,lty=1,aes(colour = 'yellow')‎, ‎linewidth=1)‎+
‎stat_function(fun=h6,lty=1,aes(colour = 'green')‎, ‎linewidth=1)‎+
 ‎xlab("x")‎ + ‎ylab("h(x)")‎+ ‎ylim(0‎, ‎4)‎+
 ‎scale_colour_manual(values = c('blue4'‎ ,‎'red3'‎ ‎,'blue','cyan3','yellow','green'),‎ 
          ‎labels = c("b=8,d=0.8"‎, ‎"b=6,d=4","b=0.8,d=0.4","b=3,d=2","b=2,d=1.2","b=3,d=0.2"))+‎
‎theme(‎
    ‎legend.position = c(1.02,0.7)‎,
    ‎legend.justification = c("right"‎, ‎"bottom"),‎
    ‎legend.box.just =‎ ‎"right",‎
    ‎legend.margin = margin(1,1,1,1)‎,
    ‎legend.title=element_blank()‎,
   ‎# Remove panel border‎
  ‎panel.border = element_blank()‎,  
  ‎# Remove panel grid lines‎
  ‎panel.grid.major = element_blank()‎,
  ‎panel.grid.minor = element_blank()‎,
  ‎# Remove panel background‎
  ‎panel.background = element_blank()‎,
  ‎# Add axis line‎
  ‎axis.line = element_line(colour =‎ "‎black")‎
  )
‎#+labs(caption="Hazard rate plot for Additive Weibull distribution")‎

‎ggsave("hzr.png"‎, ‎plot = p‎, ‎width = 6‎, ‎height = 6‎, ‎dpi = 900)‎
