#!/usr/bin/env Rscript

# Train Huling independence weights using CSV inputs (no npz dependency).
# CSV format per replication: columns = X1..Xd, T, Y (header required).

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  key <- paste0("--", flag, "=")
  hit <- args[startsWith(args, key)]
  if (length(hit) == 0) return(default)
  sub(key, "", hit[1])
}

out_dir  <- get_arg("out_dir", "./models/huling")
max_reps <- as.integer(get_arg("max_reps", "100"))

csv_dir <- getwd()

script_path <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA)
script_dir <- if (!is.na(script_path)) dirname(script_path) else getwd()
if (!requireNamespace("osqp", quietly = TRUE)) {
  stop("Package 'osqp' is required. Install with install.packages('osqp').")
}
# ensure solve_osqp is available (independence_weights uses it unqualified)
solve_osqp <- osqp::solve_osqp
source(file.path(script_dir, "independence_weights_estimation.r"))
source(file.path(script_dir, "print_indep_weights_object.R"))

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("[INFO] Using CSV directory: %s\n", csv_dir))

rep_files <- sort(list.files(path = csv_dir, pattern = "^rep\\d+\\.csv$", full.names = TRUE))
n_rpt <- length(rep_files)
if (n_rpt == 0) stop("No rep*.csv files found in csv_dir.")
n_run <- min(max_reps, n_rpt)

stem <- basename(csv_dir)
results <- list()

cat(sprintf("[INFO] Replications: %d (running %d)\n", n_rpt, n_run))

for (r in seq_len(n_run)) {
  rep_path <- file.path(csv_dir, sprintf("rep%03d.csv", r - 1))
  if (!file.exists(rep_path)) {
    stop(sprintf("Missing CSV for rep %03d: %s", r - 1, rep_path))
  }

  dat <- read.csv(rep_path, header = TRUE)
  if (ncol(dat) < 3) {
    stop(sprintf("CSV has too few columns: %s", rep_path))
  }
  if (!all(c("T", "Y") %in% names(dat))) {
    stop(sprintf("CSV must include columns named T and Y: %s", rep_path))
  }

  X <- as.matrix(dat[, setdiff(names(dat), c("T", "Y")), drop = FALSE])
  A <- as.numeric(dat[["T"]])

  t0 <- Sys.time()
  fit <- independence_weights(A, X)
  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  out_path <- file.path(out_dir, sprintf("%s_rep%03d_huling_weights.csv", stem, r - 1))
  write.csv(data.frame(weight = fit$weights), out_path, row.names = FALSE)

  results[[length(results) + 1]] <- data.frame(
    rep_idx = r - 1,
    ess = fit$ess,
    max_w = max(fit$weights),
    D_w = fit$D_w,
    time_sec = dt,
    stringsAsFactors = FALSE
  )

  cat(sprintf("[DONE] rep=%03d ess=%.1f max_w=%.2f time=%.1fs\n",
              r - 1, fit$ess, max(fit$weights), dt))
}

if (length(results) > 0) {
  res_df <- do.call(rbind, results)
  write.csv(res_df, file.path(out_dir, sprintf("%s_huling_results.csv", stem)), row.names = FALSE)
}
