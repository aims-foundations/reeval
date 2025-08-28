# packages <- c("lavaan", "semPlot", "MASS")
# installed <- rownames(installed.packages())

# for (p in packages) {
#   if (!(p %in% installed)) {
#     install.packages(p, repos = "https://cloud.r-project.org")
#   }
#   library(p, character.only = TRUE)
# }

# ---------- packages ----------
library(lavaan)
library(MASS)
set.seed(42)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript script.R <path_to_Y.csv> <number_of_factors>")
}
y_path <- args[1]   # path to CSV file
K <- as.numeric(args[2])   # number of factors

# ---------- load data ----------
Y <- read.csv(y_path, header = FALSE)
item_names <- names(Y)
J <- length(item_names)  # total number of items

# ---------- generate factor names ----------
fac_names <- paste0("Fac", 1:K)

# ------------------cfa the Y ----------------
all_items_string <- paste(item_names, collapse = " + ")
model_parts <- paste(fac_names, "=~", all_items_string)
model <- paste(model_parts, collapse = "\n")

# model <- paste(
#   sprintf("%s =~ %s", fac_names,
#           sapply(fac_names, function(f)
#             paste(sprintf("%s%02d", f, 1:30), collapse = " + "))),
#   collapse = "\n"
# )


cat("CFA Model:\n")
cat(model, "\n\n")

fit <- cfa(
  model,
  data = Y,
  ordered = names(Y),
  estimator = "WLSMV",
  std.lv = TRUE,
  missing = "pairwise"
)
U <- lavPredict(fit, type = "lv")         # N x k latent scores (eta-hat)
# write.csv(U, "U.csv", row.names = TRUE)
write.table(U, "U.csv", row.names = FALSE, col.names = TRUE, sep = ",")
Lambda <- lavInspect(fit, "est")$lambda   # p x k loadings, V
write.table(Lambda, "V.csv", row.names = FALSE, col.names = TRUE, sep = ",")
write.table(Lambda, "lambda.csv", row.names = FALSE, col.names = TRUE, sep = ",")

