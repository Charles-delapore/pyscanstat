# Cross-validation driver: run R smerc::scan.test on a fixture CSV and
# write the most likely cluster to an output CSV.
#
# Usage: Rscript smerc_compare.R <in_csv> <out_csv> <nsim> <seed>
# in_csv columns: x, y, cases, pop (header row).
# out_csv columns: loglikrat, pvalue, rr, cases, expected, locids
# (locids = 1-based location ids, semicolon-separated).
# Exits non-zero with SMERC_NOT_AVAILABLE on stderr when the smerc
# package cannot be loaded.

args <- commandArgs(trailingOnly = TRUE)
in_csv <- args[1]
out_csv <- args[2]
nsim <- as.integer(args[3])
seed <- as.integer(args[4])

if (!requireNamespace("smerc", quietly = TRUE)) {
  cat("SMERC_NOT_AVAILABLE\n", file = stderr())
  quit(status = 2)
}
suppressMessages(library(smerc))

d <- read.csv(in_csv)
set.seed(seed)
s <- scan.test(coords = as.matrix(d[, c("x", "y")]),
               cases = d$cases, pop = d$pop,
               nsim = nsim, longlat = FALSE, min.cases = 2)
mlc <- s$clusters[[1]]
write.csv(
  data.frame(
    loglikrat = mlc$loglikrat,
    pvalue = mlc$pvalue,
    rr = mlc$rr,
    cases = mlc$cases,
    expected = mlc$expected,
    locids = paste(sort(mlc$locids), collapse = ";")
  ),
  out_csv, row.names = FALSE
)
