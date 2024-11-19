library("tidyverse")
library(dplyr)
library(mirt)
library('ggpubr')
library(ggplot2)
library("catR")
library(reshape)
library(Metrics)
library(corrplot)
library(rstudioapi)
library(patchwork)
library("glue")
library("readr")

inv_logit<-function(theta, b) {
  return (1/(1+exp(-theta+b)))
}

func.create.response <- function(b, th, np, ni) {
  th.mat<-matrix(th,np,ni,byrow=FALSE) 
  b.mat <- matrix(rep(b, np), nrow = np, byrow = TRUE)
  pr<-inv_logit(th.mat, b.mat)
  resp <- pr
  for (i in 1:ncol(resp)) {
    resp[,i]<-rbinom(nrow(resp),1,resp[,i])
  }
  return (data.frame(resp))
}

func.catSim <- function(resp, item.bank, method, stop.size){
  ni = length(resp)-1
  np = length(resp[[1]])
  list.thetas <- NULL
  list.se <- NULL
  list.pid <- NULL
  list.item <- NULL
  
  pids <- resp$pid
  # delete pid column
  resp <- resp %>% select(-pid)
  
  test <- list(method = 'ML', itemSelect = method, infoType = "Fisher")
  stop <- list(rule = 'length', thr = stop.size)
  final <- list(method = 'ML')
  for (i in 1:np){
    pid <- pids[i]
    random_number <- 1
    start <- list(fixItems = random_number, nrItems = 1, theta = 0)
    res <- randomCAT(
      itemBank = item.bank, 
      responses = as.numeric(resp[i,]), 
      start = start, 
      test = test, 
      final = final, 
      stop = stop
    )
    list.thetas <- c(list.thetas, res$thetaProv)
    list.pid <- c(list.pid, rep(pid, times = stop.size))
    list.se <- c(list.se, res$seProv)
    list.item <- c(list.item, res$testItems)
  }
  list.trialNumBlock <- rep(1:stop.size, np)
  return(data.frame(
    pid = list.pid, 
    trialNumTotal = list.trialNumBlock, 
    item = list.item,
    thetaEstimate = list.thetas, 
    thetaSE = list.se
  ))
}

monte.carlo.cat.simulation <- function(th.sample, item.bank, iteration, stop.size, ni){
  resp.sample <- func.create.response(item.bank$b, th.sample, np, ni)
  
  # add simulated_pid to first column
  simulated_pid <- sprintf("sim_%03d", 1:np)
  df <- resp.sample %>%
    mutate(pid = simulated_pid) %>%
    relocate(pid)
  
  df.results <- NULL
  for (i in 1:iteration) {
    print(i)
    # shuffle rows of response matrix
    df.shuffle <- df %>% 
      sample_n(size = n(), replace = FALSE)
    
    # Calculate the midpoint
    n_rows <- nrow(df.shuffle)
    midpoint <- ceiling(n_rows / 2)
    
    # Split the response matrix into two halves
    half1 <- df.shuffle[1:midpoint, ]
    half2 <- df.shuffle[(midpoint + 1):n_rows, ]
    
    df.mfi.real <- func.catSim(half1, item.bank, "MFI", stop.size)
    df.random.real <- func.catSim(half2, item.bank, "random", stop.size)
    
    # add column: variant and iteration
    df.results <- df.results %>% 
      rbind(rbind(df.mfi.real %>% add_column(variant = "adaptive"), 
                  df.random.real %>% add_column(variant = "random")) %>% 
              add_column(iteration = i)) 
  }
  return (df.results)
}

func.visualize.differences.validate.all <- function(df.compare){
  df.plot_curve <- df.compare %>%
    group_by(variant, trialNumTotal) %>%
    dplyr::summarise(
      truetrueEstimate = trueEstimate,
      thetaEstimate = thetaEstimate,
      sem = mean(thetaSE), 
      reliability = empirical_rxx(as.matrix(tibble(F1 = thetaEstimate, SE_F1 = thetaSE))),
      mse = Metrics :: rmse(trueEstimate, thetaEstimate), 
      bias = Metrics :: bias(trueEstimate, thetaEstimate)
    )
  return (df.plot_curve %>% ungroup())
}

set.seed(42)
np <- 200
iter <- 5

args <- commandArgs(trailingOnly = TRUE)
arg <- args[1]

thata.path <- glue("../data/nonamor_calibration/{arg}/nonamor_theta.csv")
df.theta <- read_csv(thata.path, col_select = 1)
theta_mean <- mean(df.theta$theta)
theta_std <- sd(df.theta$theta)
theta <- rnorm(np, mean = theta_mean, sd = theta_std)

b.path <- glue("../data/nonamor_calibration/{arg}/nonamor_z.csv")
df.b <- read_csv(b.path, col_select = 1)
b <- df.b$z
b <- b * -1

# Fisher Large & Random
stop.size.full <- 50
ni.full <- length(b)
save.path.full <- glue("../data/cat/{arg}/cat_full.csv")
stop.size.full <- min(stop.size.full, ni.full)

item.bank.full <- data.frame(
  a = rep(1, ni.full),
  b = b,       
  c = rep(0, ni.full),
  d = rep(1, ni.full)
)

df.monte.carlo.results.v2.full <- monte.carlo.cat.simulation(theta, item.bank.full, iter, stop.size.full, ni.full)

df.true.theta.full <- data.frame(
  pid = sprintf("sim_%03d", 1:np), 
  trueEstimate = theta  
)

df.monte.carlo.simulation.compare.full <- df.true.theta.full %>%
  left_join(df.monte.carlo.results.v2.full, by = "pid")

df.compare.all.full <- df.monte.carlo.simulation.compare.full %>%
  select(pid, variant, trialNumTotal, thetaEstimate, iteration, thetaSE, trueEstimate) %>%
  mutate(variant = ifelse(variant == "adaptive", "CAT", "Random"))

df.aggregrate.learning.curve.all.full <- func.visualize.differences.validate.all(df.compare.all.full) %>% 
  mutate(bias = abs(bias))

write.csv(df.aggregrate.learning.curve.all.full, save.path.full, row.names = FALSE)

# Fisher small
b.first <- b[1]
b.rest <- sample(b[-1], 49)
b.sub <- c(b.first, b.rest)
ni.sub <- length(b.sub)
stop.size.sub <- length(b.sub)

save.path.sub <- glue("../data/cat/{arg}/cat_sub.csv")

item.bank.sub <- data.frame(
  a = rep(1, ni.sub),
  b = b.sub,       
  c = rep(0, ni.sub),
  d = rep(1, ni.sub)
)

df.monte.carlo.results.v2.sub <- monte.carlo.cat.simulation(theta, item.bank.sub, iter, stop.size.sub, ni.sub)

df.true.theta.sub <- data.frame(
  pid = sprintf("sim_%03d", 1:np), 
  trueEstimate = theta  
)

df.monte.carlo.simulation.compare.sub <- df.true.theta.sub %>%
  left_join(df.monte.carlo.results.v2.sub, by = "pid")

df.compare.all.sub <- df.monte.carlo.simulation.compare.sub %>%
  select(pid, variant, trialNumTotal, thetaEstimate, iteration, thetaSE, trueEstimate) %>%
  mutate(variant = ifelse(variant == "adaptive", "CAT", "Random"))

df.aggregrate.learning.curve.all.sub <- func.visualize.differences.validate.all(df.compare.all.sub) %>% 
  mutate(bias = abs(bias))

write.csv(df.aggregrate.learning.curve.all.sub, save.path.sub, row.names = FALSE)

# gif data
np.gif <- 2
iter.gif <- 1
theta.gif <- rep(0.5, np.gif)
stop.size.gif <- 100
ni.gif <- length(b)
save.path.gif <- glue("../data/cat/{arg}/cat_gif.csv")
stop.size.gif <- min(stop.size.gif, ni.gif)

item.bank.gif <- data.frame(
  a = rep(1, ni.gif),
  b = b,
  c = rep(0, ni.gif),
  d = rep(1, ni.gif)
)

df.monte.carlo.results.v2.gif <- monte.carlo.cat.simulation(theta.gif, item.bank.gif, iter.gif, stop.size.gif, ni.gif)

df.true.theta.gif <- data.frame(
  pid = sprintf("sim_%03d", 1:np.gif), 
  trueEstimate = theta.gif  
)

df.monte.carlo.simulation.compare.gif <- df.true.theta.gif %>%
  left_join(df.monte.carlo.results.v2.gif, by = "pid")

df.compare.all.gif <- df.monte.carlo.simulation.compare.gif %>%
  select(pid, variant, trialNumTotal, thetaEstimate, iteration, thetaSE, trueEstimate) %>%
  mutate(variant = ifelse(variant == "adaptive", "CAT", "Random"))

df.aggregrate.learning.curve.all.gif <- func.visualize.differences.validate.all(df.compare.all.gif) %>% 
  mutate(bias = abs(bias))

write.csv(df.aggregrate.learning.curve.all.gif, save.path.gif, row.names = FALSE)
