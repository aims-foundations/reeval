library(mirt)
library(ggplot2)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)
data_type <- args[1]

if (data_type == "real_normal") {
  input_files <- c("../data/real/response_matrix/normal/base_matrix.csv", 
                   "../data/real/response_matrix/normal/perturb1_matrix.csv", 
                   "../data/real/response_matrix/normal/perturb2_matrix.csv", 
                   "../data/real/response_matrix/normal/all_matrix.csv")

  output_files_base_Z <- c("../data/real/irt_result/normal/Z/base_1PL_Z.csv", 
                           "../data/real/irt_result/normal/Z/base_2PL_Z.csv", 
                           "../data/real/irt_result/normal/Z/base_3PL_Z.csv")
  output_files_perturb1_Z <- c("../data/real/irt_result/normal/Z/perturb1_1PL_Z.csv", 
                               "../data/real/irt_result/normal/Z/perturb1_2PL_Z.csv", 
                               "../data/real/irt_result/normal/Z/perturb1_3PL_Z.csv")
  output_files_perturb2_Z <- c("../data/real/irt_result/normal/Z/perturb2_1PL_Z.csv", 
                               "../data/real/irt_result/normal/Z/perturb2_2PL_Z.csv", 
                               "../data/real/irt_result/normal/Z/perturb2_3PL_Z.csv")
  output_files_all_Z <- c("../data/real/irt_result/normal/Z/all_1PL_Z.csv", 
                          "../data/real/irt_result/normal/Z/all_2PL_Z.csv", 
                          "../data/real/irt_result/normal/Z/all_3PL_Z.csv")
  output_files_list_Z <- list(output_files_base_Z, output_files_perturb1_Z, output_files_perturb2_Z, output_files_all_Z)

  output_files_base_theta <- c("../data/real/irt_result/normal/theta/base_1PL_theta.csv", 
                               "../data/real/irt_result/normal/theta/base_2PL_theta.csv", 
                               "../data/real/irt_result/normal/theta/base_3PL_theta.csv")
  output_files_perturb1_theta <- c("../data/real/irt_result/normal/theta/perturb1_1PL_theta.csv", 
                                   "../data/real/irt_result/normal/theta/perturb1_2PL_theta.csv", 
                                   "../data/real/irt_result/normal/theta/perturb1_3PL_theta.csv")
  output_files_perturb2_theta <- c("../data/real/irt_result/normal/theta/perturb2_1PL_theta.csv", 
                                   "../data/real/irt_result/normal/theta/perturb2_2PL_theta.csv", 
                                   "../data/real/irt_result/normal/theta/perturb2_3PL_theta.csv")
  output_files_all_theta <- c("../data/real/irt_result/normal/theta/all_1PL_theta.csv", 
                              "../data/real/irt_result/normal/theta/all_2PL_theta.csv", 
                              "../data/real/irt_result/normal/theta/all_3PL_theta.csv")
  output_files_list_theta <- list(output_files_base_theta, output_files_perturb1_theta, output_files_perturb2_theta, output_files_all_theta)
} else if (data_type == "real_appendix1"){
  input_files <- c("../data/real/response_matrix/appendix1/base_matrix.csv", 
                   "../data/real/response_matrix/appendix1/perturb1_matrix.csv", 
                   "../data/real/response_matrix/appendix1/perturb2_matrix.csv", 
                   "../data/real/response_matrix/appendix1/all_matrix.csv")

  output_files_base_Z <- c("../data/real/irt_result/appendix1/Z/base_1PL_Z.csv", 
                           "../data/real/irt_result/appendix1/Z/base_2PL_Z.csv", 
                           "../data/real/irt_result/appendix1/Z/base_3PL_Z.csv")
  output_files_perturb1_Z <- c("../data/real/irt_result/appendix1/Z/perturb1_1PL_Z.csv", 
                               "../data/real/irt_result/appendix1/Z/perturb1_2PL_Z.csv", 
                               "../data/real/irt_result/appendix1/Z/perturb1_3PL_Z.csv")
  output_files_perturb2_Z <- c("../data/real/irt_result/appendix1/Z/perturb2_1PL_Z.csv", 
                               "../data/real/irt_result/appendix1/Z/perturb2_2PL_Z.csv", 
                               "../data/real/irt_result/appendix1/Z/perturb2_3PL_Z.csv")
  output_files_all_Z <- c("../data/real/irt_result/appendix1/Z/all_1PL_Z.csv", 
                          "../data/real/irt_result/appendix1/Z/all_2PL_Z.csv", 
                          "../data/real/irt_result/appendix1/Z/all_3PL_Z.csv")
  output_files_list_Z <- list(output_files_base_Z, output_files_perturb1_Z, output_files_perturb2_Z, output_files_all_Z)

  output_files_base_theta <- c("../data/real/irt_result/appendix1/theta/base_1PL_theta.csv", 
                               "../data/real/irt_result/appendix1/theta/base_2PL_theta.csv", 
                               "../data/real/irt_result/appendix1/theta/base_3PL_theta.csv")
  output_files_perturb1_theta <- c("../data/real/irt_result/appendix1/theta/perturb1_1PL_theta.csv", 
                                   "../data/real/irt_result/appendix1/theta/perturb1_2PL_theta.csv", 
                                   "../data/real/irt_result/appendix1/theta/perturb1_3PL_theta.csv")
  output_files_perturb2_theta <- c("../data/real/irt_result/appendix1/theta/perturb2_1PL_theta.csv", 
                                   "../data/real/irt_result/appendix1/theta/perturb2_2PL_theta.csv", 
                                   "../data/real/irt_result/appendix1/theta/perturb2_3PL_theta.csv")
  output_files_all_theta <- c("../data/real/irt_result/appendix1/theta/all_1PL_theta.csv", 
                              "../data/real/irt_result/appendix1/theta/all_2PL_theta.csv", 
                              "../data/real/irt_result/appendix1/theta/all_3PL_theta.csv")
  output_files_list_theta <- list(output_files_base_theta, output_files_perturb1_theta, output_files_perturb2_theta, output_files_all_theta)

} else if (data_type == "synthetic") {
  input_files <- c("../data/synthetic/response_matrix/synthetic_matrix.csv")

  output_files_synthetic_Z <- c("../data/synthetic/irt_result/Z/synthetic_1PL_Z.csv",
                                 "../data/synthetic/irt_result/Z/synthetic_2PL_Z.csv",
                                 "../data/synthetic/irt_result/Z/synthetic_3PL_Z.csv")
  output_files_list_Z <- list(output_files_synthetic_Z)

  output_files_synthetic_theta <- c("../data/synthetic/irt_result/theta/synthetic_1PL_theta.csv",
                                     "../data/synthetic/irt_result/theta/synthetic_2PL_theta.csv",
                                     "../data/synthetic/irt_result/theta/synthetic_3PL_theta.csv")
  output_files_list_theta <- list(output_files_synthetic_theta)
} else {
  stop("Invalid data type specified.")
}

models <- c('Rasch', '2PL', '3PL')

for (i in 1:length(input_files)) {
  data <- read.csv(input_files[i], row.names=1)
  # items_to_remove <- which(apply(data, 2, function(x) length(unique(x)) == 1))
  # if(length(items_to_remove) > 0) {
  #   data <- data[, -items_to_remove]
  # }

  for (j in 1:length(models)) {
    model <- mirt(data, 1, models[j])
    Z <- coef(model)
    write.csv(Z, output_files_list_Z[[i]][j])
    
    theta <- fscores(model)
    write.csv(theta, output_files_list_theta[[i]][j])
  }
}
