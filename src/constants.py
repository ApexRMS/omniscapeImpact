# Value used throughout omniscape and omniscapeImpact to flag "no data"
NODATA_VALUE = -9999

# How closely two rasters must be aligned before they can be compared, as a
# fraction of one pixel. Loose enough to tolerate floating-point differences
# between a raster merged from spatial tiles and one written in a single pass,
# tight enough that a genuine offset of even one pixel is rejected.
GRID_TOLERANCE_FRACTION = 0.01