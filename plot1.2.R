‎persp.lm <‎- 
‎function (x‎, ‎form‎, ‎at‎, ‎bounds‎, ‎zlim‎, ‎zlab‎, ‎xlabs‎, ‎col =‎ ‎"white",‎ ‎xlab=xlab‎,
    ‎contours = NULL‎, ‎hook‎, ‎atpos = 3‎, ‎theta =‎ -25, ‎phi = 20‎, 
    ‎r = 4‎, ‎border = NULL‎, ‎box = TRUE‎, ‎ticktype =‎ ‎"detailed",‎ ‎ylab‎,
    ... ) 
{
    ‎draw.cont.line = function(line)‎ {
        ‎if (cont.varycol)‎ {
            ‎cont.col = col‎
            ‎if (length(col) > 1)‎ 
                ‎cont.col = col[cut(c(line$level‎, ‎dat$zlim)‎, ‎length(col))][1]‎
        }
         ‎lines(trans3d(line$x‎, ‎line$y‎, ‎cont.z‎, ‎transf)‎,
              ‎col=cont.col‎, ‎lwd=2)‎
    }
    ‎plot.data = contour.lm(x‎, ‎form‎, ‎at‎, ‎bounds‎, ‎zlim‎, ‎xlabs‎, 
        ‎atpos = atpos‎, ‎plot.it = FALSE)‎
    ‎transf = list()‎
    ‎if (missing(zlab))‎ 
        ‎zlab =‎ ""
    ‎facet.col = col‎
    ‎cont = !is.null(contours)‎
    ‎if (mode(contours) ==‎ "‎logical")‎ 
        ‎cont = contours‎
    ‎cont.first = cont‎
    ‎cont.z = cz = plot.data[[1]]$zlim[1]‎
    ‎cont.col = 1‎
    ‎cont.varycol = FALSE‎
    ‎cont.lwd = 1‎
    ‎if (is.character(contours))‎ {
        ‎idx = charmatch(contours‎, ‎c("top"‎, ‎"bottom",‎ ‎"colors"),‎ 
            ‎0)‎
        ‎if (idx == 1)‎ {
            ‎cont.first = FALSE‎
            ‎cont.z = plot.data[[1]]$zlim[2]‎
        }
        ‎else if (idx == 2)‎ {
        }
        ‎else if (idx == 3)‎ {
            ‎cont.varycol = TRUE‎
            ‎if (length(col) < 2)‎ 
                ‎col =  rainbow(40)‎
        }
        ‎else cont.col = contours‎
    }
    ‎else if (is.list(contours))‎ {
        ‎if (!is.null(contours$z))‎ 
            ‎cz = contours$z‎
        ‎if (is.numeric(cz))‎ 
            ‎cont.z = cz‎
        ‎else if (cz ==‎ "‎top")‎ {
            ‎cont.first = FALSE‎
            ‎cont.z = plot.data[[1]]$zlim[2]‎
        }
        ‎if (!is.null(contours$col))‎ 
            ‎cont.col = contours$col‎
        ‎if (!is.null(contours$lwd))‎ 
            ‎cont.lwd = contours$lwd$‎
        ‎if (charmatch(cont.col‎, ‎"colors",‎ ‎0) == 1)‎ {
            ‎cont.varycol = TRUE‎
            ‎if (length(col) < 2)‎ 
                ‎col = rainbow(40)‎
        }
    }


    ‎for (i in 1:length(plot.data))‎ {
        ‎dat = plot.data[[i]]‎
        ‎cont.lines = NULL‎
        ‎if (!missing(hook))‎ 
            ‎if (!is.null(hook$pre.plot))‎ 
                ‎hook$pre.plot(dat$labs)‎
        ‎if (cont)‎ 
            ‎cont.lines = contourLines(dat$x‎, ‎dat$y‎, ‎dat$z)‎
        ‎if (cont && cont.first)‎ {
            ‎transf = persp(dat$x‎, ‎dat$y‎, ‎dat$z‎, ‎zlim = dat$zlim‎, ‎xlab=ylab‎,
                ‎theta = theta‎, ‎phi = phi‎, ‎r = r‎, ‎col = NA‎, ‎border = NA‎, 
                ‎box = FALSE)‎
            ‎lapply(cont.lines‎, ‎draw.cont.line)‎
            ‎par(new = TRUE)‎
        }
        ‎if (length(col) > 1)‎ {
            ‎nrz = nrow(dat$z)‎
            ‎ncz = ncol(dat$z)‎
            ‎zfacet = dat$z[-1‎, -‎1]‎ + ‎dat$z[-1‎, -‎ncz]‎ + ‎dat$z[-nrz‎, 
                -‎1]‎ + ‎dat$z[-nrz‎, -‎ncz]‎
            ‎zfacet = c(zfacet/4‎, ‎dat$zlim)‎
            ‎facet.col = cut(zfacet‎, ‎length(col))‎
            ‎facet.col = col[facet.col]‎
        }
        ‎transf = persp(dat$x‎, ‎dat$y‎, ‎dat$z‎, ‎xlab = xlab‎, 
             ‎zlab = zlab‎, ‎zlim = dat$zlim‎, ‎ylab=ylab‎,
            ‎col = facet.col‎, ‎border = border‎, ‎box = box‎, ‎theta = theta‎, 
            ‎phi = phi‎, ‎r = r‎, ‎ticktype = ticktype)‎
        ‎if (atpos == 3)‎ 
            ‎title(sub = dat$labs[5])‎
        ‎if (cont && !cont.first)‎ 
            ‎lapply(cont.lines‎, ‎draw.cont.line)‎
        ‎if (!missing(hook))‎ 
            ‎if (!is.null(hook$post.plot))‎ 
                ‎hook$post.plot(dat$labs)‎
        ‎plot.data[[i]]$transf = transf‎
    }
    ‎invisible(plot.data)‎
}

‎x <‎- ‎seq(0.01,.2,length=100)‎
‎y <‎- ‎seq(0.001,3,length=100)‎
‎d <‎- ‎expand.grid(x=x,y=y)‎
‎z <‎- ‎c(data=NA,10000)‎

‎n=100;a=1;b=.09;c=1.7‎

‎data=scan()‎
‎0.39 0.81 0.85 0.98 1.08 1.12 1.17 1.18 1.22 1.25 1.36 1.41 1.47 1.57‎
‎1.57 1.59 1.59 1.61 1.61 1.69 1.69 1.71 1.73 1.8 1.84 1.84 1.87 1.89‎
‎1.92 2 2.03 2.03 2.05 2.12 2.17 2.17 2.17 2.35 2.38 2.41 2.43 2.48‎
‎2.48 2.5 2.53 2.55 2.55 2.56 2.59 2.67 2.73 2.74 2.76 2.77 2.79 2.81‎
‎2.81 2.82 2.83 2.85 2.87 2.88 2.93 2.95 2.96 2.97 2.97 3.09 3.11 3.11‎
‎3.15 3.15 3.19 3.19 3.22 3.22 3.27 3.28 3.31 3.31 3.33 3.39 3.39 3.51‎
‎3.56 3.6 3.65 3.68 3.68 3.68 3.7 3.75 4.2 4.38 4.42 4.7 4.9 4.91‎
‎5.08 5.56‎


‎z=c()‎
‎k=1‎
‎for (i in 1:100)‎ {
‎for (j in 1:100)‎ {
‎z[k]=-((n*log(a))+(n*log(y[j]))+(n*log(x[i]))-(y[j]*sum(data^x[i]))‎+
           ‎((a-1)*sum(log(1-exp(-y[j]*(data^x[i]))))))‎
‎k=k+1‎
}}
‎library(rsm)‎
‎CR.rs2=rsm (z‎ ~ ‎SO(x,y)‎, ‎data=d)‎

‎png("persp.png",units="px‎" , ‎width=1600‎, ‎height=1600‎, ‎res=180‎, ‎type="cairo"‎, ‎pointsize = 16)‎
‎persp(CR.rs2,x~y‎, ‎theta=60‎, ‎xlab=expression(Lambda),phi=0‎, ‎r = 5‎, ‎d=2‎,
‎border =‎ ‎"black",‎ ‎ltheta =‎ -135, ‎lphi = 0‎, ‎shade = 0.75‎, ‎zlab="",ylab=expression(beta)‎, ‎col.axis=1‎,
‎font.lab=1,col.lab=1,contour=("colors"))‎
‎dev.off()‎

‎jpeg(filename =‎ ‎"persp.jpg",units="px"‎ , ‎width=1600‎, ‎height=1600‎, ‎res=180‎,
        ‎type="cairo"‎, ‎pointsize = 18,quality = 100)‎
‎persp(CR.rs2,x~y‎, ‎theta=60‎, ‎xlab=expression(Lambda),phi=0‎, ‎r = 5‎, ‎d=2‎,
‎border =‎ ‎"black",‎ ‎ltheta =‎ -135, ‎lphi = 0‎, ‎shade = 0.75‎, ‎zlab="",ylab=expression(beta)‎, ‎col.axis=1‎,
‎font.lab=1,col.lab=1,contour=("colors"))‎
‎dev.off()‎
