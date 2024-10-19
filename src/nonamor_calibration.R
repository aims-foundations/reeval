library(mirt)
library(ggplot2)
library(dplyr)

input_file <- "../data/pre_calibration/mmlu/matrix.csv"
output_files <- c("../nonamor_calibration_R/mmlu/R_1PL.csv", 
                  "../nonamor_calibration_R/mmlu/R_2PL.csv", 
                  "../nonamor_calibration_R/mmlu/R_3PL.csv")

models <- c('Rasch', '2PL', '3PL')

data <- read.csv(input_file, row.names=1)

for (j in 1:length(models)) {
  model <- mirt(data, 1, models[j])
  coefficient <- coef(model)
  write.csv(coefficient, output_files[j])
  fscores(model)
}
