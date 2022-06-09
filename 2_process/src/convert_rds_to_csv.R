#' @description Convert an rds file that contains a single dataframe to csv
#' @param rds_file Path to rds file to convert
#' @param csv_file Path to csv to save
rds_to_csv <- function(rds_file, csv_file) {
  df <- readRDS(rds_file)
  write.csv(df, csv_file, row.names=FALSE)
}

rds_to_csv(snakemake@input[['in_file']], snakemake@output[['csv_file']])
