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

func.catSim <- function(resp, item.bank, method){
  ni = length(resp)-1
  np = length(resp[[1]])
  list.thetas <- NULL
  list.se <- NULL
  list.pid <- NULL
  list.item <- NULL
  
  pids <- resp$pid
  resp <- resp %>% select(-pid)
  
  test <- list(method = 'ML', itemSelect = method, infoType = "Fisher")
  stop <- list(rule = 'length',thr = ni)
  final <- list(method = 'ML')
  for (i in 1:np){
    pid <- pids[i]
    random_number <- 1
    start <- list(fixItems = random_number, nrItems = 1, theta = 0)
    res <- randomCAT(itemBank = item.bank, 
                     responses = as.numeric(resp[i,]), 
                     start = start,
                     test = test,
                     final = final,
                     stop = stop)
    list.thetas <- c(list.thetas, res$thetaProv)
    list.pid <- c(list.pid, rep(pid, times = ni))
    list.se <- c(list.se, res$seProv)
    list.item <- c(list.item, res$testItems)
  }
  list.trialNumBlock <- rep(1:ni, np)
  return(data.frame(pid = list.pid, 
                    trialNumTotal = list.trialNumBlock, 
                    item = list.item,
                    thetaEstimate = list.thetas, 
                    thetaSE = list.se))
}

monte.carlo.cat.simulation <- function(th.sample, item.bank, iteration){
  resp.sample <- func.create.response(item.bank$b, th.sample, np, ni)
  
  simulated_pid <- sprintf("sim_%03d", 1:np)
  df <- resp.sample %>%
    mutate(pid = simulated_pid) %>%
    relocate(pid)
  
  df.results <- NULL
  for (i in 1:iteration) {
    print(i)
    df.shuffle <- df %>% 
      sample_n(size = n(), replace = FALSE)
    
    # Calculate the midpoint
    n_rows <- nrow(df.shuffle)
    midpoint <- ceiling(n_rows / 2)
    
    # Split the data frame into two halves
    half1 <- df.shuffle[1:midpoint, ]
    half2 <- df.shuffle[(midpoint + 1):n_rows, ]
    
    df.mfi.real <- func.catSim(half1, item.bank, "MFI")
    df.random.real <- func.catSim(half2, item.bank, "random")
    
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
    dplyr::summarise(sem = mean(thetaSE), 
                     reliability = empirical_rxx(as.matrix(tibble(F1 = thetaEstimate, SE_F1 = thetaSE))),
                     mse = Metrics :: rmse(trueEstimate, thetaEstimate), 
                     bias = Metrics :: bias(trueEstimate, thetaEstimate))
  return (df.plot_curve %>% ungroup())
}

set.seed(42)
np <- 200
iter <- 5

thata.path <- glue("../data/nonamor_calibration/airbench/nonamor_theta.csv")
df.theta <- read_csv(thata.path, col_select = 1)
theta_mean <- mean(df.theta$theta)
theta_std <- sd(df.theta$theta)
theta <- rnorm(np, mean = theta_mean, sd = theta_std)

b.path <- glue("../data/nonamor_calibration/airbench/nonamor_z.csv")
df.b <- read_csv(b.path, col_select = 1)
b <- df.b$z
b <- b * -1
b_first <- b[1]
b_rest <- sample(b[-1], 49)
b <- c(b_first, b_rest)
ni <- length(b)

save.path <- glue("../data/catgt/cat_subset.csv")

item.bank <- data.frame(
  a = rep(1, ni),
  b = b,       
  c = rep(0, ni),
  d = rep(1, ni)
)

resp.sample <- func.create.response(b, theta, np, ni)

df.monte.carlo.results.v2 <- monte.carlo.cat.simulation(theta, item.bank, iter)

df.true.theta <- data.frame(
  pid = sprintf("sim_%03d", 1:np), 
  trueEstimate = theta  
)

df.monte.carlo.simulation.compare <- df.true.theta %>%
  left_join(df.monte.carlo.results.v2, by = "pid")

df.compare.all <- df.monte.carlo.simulation.compare %>%
  select(pid, variant, trialNumTotal, thetaEstimate, iteration, thetaSE, trueEstimate) %>%
  mutate(variant = ifelse(variant == "adaptive", "CAT", "Random"))

df.aggregrate.learning.curve.all <- func.visualize.differences.validate.all(df.compare.all) %>% 
  mutate(bias = abs(bias))

write.csv(df.aggregrate.learning.curve.all, save.path, row.names = FALSE)

