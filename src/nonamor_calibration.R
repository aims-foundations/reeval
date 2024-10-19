library(mirt)
library(ggplot2)
library(dplyr)

input_file <- "../data/pre_calibration/mmlu/matrix.csv"
z_files <- c("../data/nonamor_calibration_R/mmlu/R_1PL_z.csv", 
                  "../data/nonamor_calibration_R/mmlu/R_2PL_z.csv", 
                  "../data/nonamor_calibration_R/mmlu/R_3PL_z.csv")
theta_files <- c("../data/nonamor_calibration_R/mmlu/R_1PL_theta.csv", 
                  "../data/nonamor_calibration_R/mmlu/R_2PL_theta.csv", 
                  "../data/nonamor_calibration_R/mmlu/R_3PL_theta.csv")

models <- c('Rasch', '2PL', '3PL')
data <- read.csv(input_file, row.names=1)

for (j in 1:length(models)) {
  model <- mirt(data, 1, models[j])
  z <- coef(model)
  write.csv(z, z_files[j])
  theta <- fscores(model)
  write.csv(theta, theta_files[j])
}
