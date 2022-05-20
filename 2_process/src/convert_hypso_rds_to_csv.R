# Return a vector of elevation, area, and depth from a hypsography dataframe of H/A (elevation/area) pairs
get_hypso_attributes <- function(df) {
  # Find index of min (lake bottom) and max (lake surface) elevation
  i_surface <- which.max(df[['H']])
  i_bottom <- which.min(df[['H']])
  # Lake elevation is max elevation
  elevation <- df[i_surface, 'H']
  # Lake area is area at lake surface
  area <- df[i_surface, 'A']
  # Lake depth is max elevation - min elevation
  bottom_elevation <- df[i_bottom, 'H']
  depth_hypso <- elevation - bottom_elevation
  # Return vector of derived attributes
  c(elevation, area, depth_hypso)
}

# Convert hypsography rds file to a csv with columns site_id, elevation, area, depth
hypso_rds_to_csv <- function(rds_file, csv_file) {
  all_H_A <- readRDS(rds_file)
  # Obtain a list of vectors of c(elevation, area, depth)
  all_attributes_list <- lapply(all_H_A, get_hypso_attributes)
  # Convert to data.frame
  all_attributes <- as.data.frame(do.call(rbind, all_attributes_list))
  # Change row names (which are NHDHR site_ids) to a column
  site_ids <- rownames(all_attributes)
  all_attributes <- cbind(site_ids, data.frame(all_attributes, row.names=NULL))
  # Add column names and write to csv file
  # Name depth as depth_hypso to keep separate from other sources of lake depth
  colnames(all_attributes) <- c("site_id", "elevation", "area", "depth_hypso")
  write.csv(all_attributes, csv_file, row.names=FALSE)
}

hypso_rds_to_csv(snakemake@input[['in_file']], snakemake@output[['csv_file']])
