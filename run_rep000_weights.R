#!/usr/bin/env Rscript

# Reads ./rep000.csv from the current working directory,
# computes independence weights, and writes ./rep000_weights.csv.

suppressPackageStartupMessages(library(osqp))
solve_osqp <- osqp::solve_osqp

source("Huling/independence_weights_estimation.r")

csv_path <- "rep000.csv"
if (!file.exists(csv_path)) {
  stop(sprintf("File not found: %s (current dir: %s)", csv_path, getwd()))
}

dat <- read.csv(csv_path, header = TRUE)
if (!all(c("T", "Y") %in% names(dat))) {
  stop("CSV must include columns named T and Y.")
}

A <- dat[["T"]]
X <- as.matrix(dat[, setdiff(names(dat), c("T", "Y")), drop = FALSE])

fit <- independence_weights(A, X)
write.csv(data.frame(weight = fit$weights), "rep000_weights.csv", row.names = FALSE)

cat("[DONE] saved rep000_weights.csv\n")
