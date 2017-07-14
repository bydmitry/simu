library(plotly)

plyMargins <- list(pad = 1, l = 100, r = 80, t = 100, b = 100)

#-----------------------------------------------------------------------------------
# Modelling risk functions
#-----------------------------------------------------------------------------------

linear_h <- function(betas=c(1,2), x1.range=seq(-1,1,0.05), x2.range=seq(-1,1,0.05)){
  # Generate a dataframe with every possible combination of x1 and x2
  grid    <- expand.grid(x1 = x1.range, x2 = x2.range)
  grid$h  <- grid$x1*betas[1] + grid$x2*betas[2]
  return( grid )  
}

gaussian_h <- function( x1.range=seq(-1,1,0.05), x2.range=seq(-1,1,0.05),
                        rad=0.5, max.hr=2.0 ){
  grid    <- expand.grid(x1 = x1.range, x2 = x2.range)
  z       <- grid$x1**2 + grid$x2**2
  grid$h  <- max.hr * (exp( -(z) / (2 * rad ** 2) ))
  grid$h  <- grid$h - (max.hr/2.0)
  return( grid )
}

#-----------------------------------------------------------------------------------
# Plotting :: Risks
#-----------------------------------------------------------------------------------
plot_risk_surf <- function(data, mode = '2d', wsize=500, hsize=500){
  if(mode == '2d'){
    p <- plot_ly(data, x = ~x1, y = ~x2, z = ~h, type = "contour", 
                   width = wsize, height = hsize,
                   colorscale = 'RdYlBu',
                   zmin = floor( min(data$h) ),
                   zmax = ceiling( max(data$h) ),
                   autocontour = FALSE,
                   contours = list(coloring='heatmap'),
                   line = list(width=0)) %>%
                layout(title='True risk', margin = plyMargins) 
  }
  if(mode == '3d'){
    h  = matrix(data=data$h, nrow=sqrt(length(data$h)))
    x1 = sort(unique(data$x1))
    x2 = sort(unique(data$x2))
    
    p <- plot_ly(x = x1, y = x2, z = h, 
                   width = wsize, height = hsize,
                   colorscale = 'RdYlBu') %>%
                layout(title='True risk', margin = plyMargins) %>% add_surface()
  }
  return(p)
}

#-----------------------------------------------------------------------------------
# Plotting :: Survilal times
#-----------------------------------------------------------------------------------
plot_KMs <- function(survFitObj, h=500, w=750){
  fData <- data.frame(
    time  = survFitObj$time,
    surv  = survFitObj$surv,
    cens  = survFitObj$n.censor,
    nrisk = survFitObj$n.risk,
    low   = survFitObj$lower,
    high  = survFitObj$upper
  )
  cData <- data.frame(
      xs  = survFitObj$time[survFitObj$n.censor > 0],
      ys  = survFitObj$surv[survFitObj$n.censor > 0]
  )
  original_surv <- data.frame(
    xs  = fData$time,
    ys  = exp(-(1/mean(fData$time))*fData$time)
  )
  
  marker_style <- list(color = 'rgb(255, 182, 193)', symbol='hourglass',opacity=0.9, size=8)
  
  p <- plot_ly(data = fData, x = ~time, y = ~high, width = w, height = h,
          type = 'scatter', mode = 'lines',
          line = list(color = 'transparent'), showlegend = T, name = 'High CI95%' ) %>%
        add_trace(
          data = fData, x = ~time, y = ~low, type = 'scatter', mode = 'lines',
          fill = 'tonexty', fillcolor='rgba(22, 96, 167, 0.15)', line = list(color = 'transparent'),
          showlegend = T, name = 'Low CI95%' ) %>%
        add_trace(
          data = fData, x = ~time, y = ~surv, name = 'Estimated Survival', type = 'scatter', mode = 'lines',
          line = list(color = 'rgb(22, 96, 167)', width = 2, shape='vhv') ) %>%
        add_trace(
          data = cData, x = ~xs, y = ~ys, name = 'Censoring', visible = 'legendonly',
          mode = 'markers', line = list(width = 0), marker=marker_style ) %>%
        add_trace(
          data = original_surv, x = ~xs, y = ~ys, name = 'Exponential Survival', type = 'scatter', mode = 'lines',
          line = list(color = 'rgb(111, 22, 167)', width = 2) ) %>%
        layout(
          title   = "Observed Failure times",
          margin  = plyMargins,
          xaxis   = list(
            title = "Time",
            range = c(0,max(survFitObj$time) * 1.1)
          ),
          yaxis   = list(title = "Survival", range = c(0,1))
        )
  
  return( p )
}

plot_time_hist  <- function(data, nbinsx = 35, h=500, w=750){
  xlim <- max(data['t']) * 1.15
  
  lmbd <- 1/mean(data[,'t'])
  dens <- lmbd * exp(-lmbd*data[,'t'])
  
  p <- plot_ly(height = h, width = w, type='histogram',
          #histnorm = 'probability density',
          autobinx = FALSE, alpha = 0.7, 
          xbins = list(start=0, end=1000, size=xlim/nbinsx) ) %>%
    add_histogram(
      name  = 'Uncensored', yaxis = 'y',
      x     = data[data$e == 1,'t']
    ) %>% 
    add_histogram(
      name  = 'Censored', yaxis = 'y',
      x     = data[data$e == 0,'t']
    ) %>% 
    add_histogram(
      name  = 'Failure times', yaxis = 'y',
      x     = data$f
    ) %>%
    #add_lines(x = data$t, y = dens, fill = "tozeroy") %>%
    layout(barmode = "stacked", legend = list(orientation = 'h'))
    #layout(barmode = "stacked", legend = list(orientation = 'h'), yaxis2 = list(overlaying = "y", side = "right"))
  return( p )
}
#-----------------------------------------------------------------------------------
# Custom calculation of Baseline Hazard and Survival
#-----------------------------------------------------------------------------------
compute_baseline <- function(sdata, pred){
  # Sorted event times:
  y   <- sort( sdata[sdata$e == 1, 't'] ) 
  # Number of events at y_i:
  d   <- rep(1,length(y))
  
  # Baseline Hazard:
  h0 <- rep(NA, length(y))
  for(l in 1:length(y))
  {
    h0[l] <- d[l] / sum(exp(sdata[sdata$t >= y[l], pred]))
  }
  H0_hat <- data.frame(h=h0,t=y)
  
  # Baseline Survival:
  s0 <- rep(NA, length(y))
  for(l in 1:length(y))
  {
    s0[l] <- exp( -sum(H0_hat[H0_hat$t <= y[l], 'h']) )
  }
  S0_hat <- data.frame(s=s0,t=y)
  
  base_HS <- merge(H0_hat, S0_hat, by='t')
  return(base_HS)
}
  
#-----------------------------------------------------------------------------------
# EOF
#-----------------------------------------------------------------------------------
