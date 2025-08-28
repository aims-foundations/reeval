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

K <- 2 # factors
Jper <- 30 # items per factor
J <- K * Jper
N <- 400 # people

# fac_names <- c("Prp","Prm","Cnc","Abs","Frm","Sys","Met")
fac_names <- c("Fac1","Fac2")

df <- read.csv("U.csv", header = TRUE) 
item_names <- unlist(lapply(seq_len(K), function(k)
  sprintf("%s%02d", fac_names[k], 1:Jper)))

# map each item to its factor
item2fac <- rep(fac_names, each = Jper)

# loadings (simple structure)
lambda <- runif(J, 0.6, 0.9)     # moderately strong loadings

R <- matrix(0.5, K, K); diag(R) <- 1
Sigma_eta <- R                   # SDs = 1

# residual variance so that Var(y*) = 1 for ordered-probit link
theta <- 1 - lambda^2            # ensures identification for simulation

# ---------- simulate latent and continuous responses y* ----------
eta <- mvrnorm(N, mu = rep(0, K), Sigma = Sigma_eta)

Ystar <- matrix(NA, N, J)

colnames(Ystar) <- item_names

for (j in seq_len(J)) {
  k <- match(item2fac[j], fac_names)
  e <- rnorm(N, 0, sqrt(theta[j]))
  # for each job, we find Ystar
  # lambda * factor
  Ystar[, j] <- lambda[j] * eta[, k] + e
}


# thresholds (single cut at 0 -> dichotomous)
Y <- 1 * (Ystar > 0)
Y <- as.data.frame(Y)
write.table(Y, "Y.csv", row.names = FALSE, col.names = TRUE, sep = ",")
for (nm in names(Y)) Y[[nm]] <- ordered(Y[[nm]], levels = c(0,1))

# ------------------cfa the Y ----------------
model <- paste(
  sprintf("%s =~ %s", fac_names,
          sapply(fac_names, function(f)
            paste(sprintf("%s%02d", f, 1:Jper), collapse = " + "))),
  collapse = "\n"
)
fit <- cfa(
  model,
  data = Y,
  ordered = names(Y),
  estimator = "WLSMV",
  std.lv = TRUE
)
U <- lavPredict(fit, type = "lv")         # N x k latent scores (eta-hat)
# write.csv(U, "U.csv", row.names = TRUE)
write.table(U, "U.csv", row.names = FALSE, col.names = TRUE, sep = ",")
Lambda <- lavInspect(fit, "est")$lambda   # p x k loadings, V
write.table(Lambda, "V.csv", row.names = FALSE, col.names = TRUE, sep = ",")

