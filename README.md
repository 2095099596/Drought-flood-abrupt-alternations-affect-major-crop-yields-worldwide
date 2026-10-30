# Drought-flood-abrupt-alternations-affect-major-crop-yields-worldwide

## Figure1a

### Figure1a_1

#### 1. Correlation Between Drought–Flood Abrupt Alternation Events and Crop Yield Anomalies

**Function**

Analyzes the relationship between drought–flood abrupt alternation events (LDFAI) of varying intensity and detrended crop yield residuals across crop phenological stages, and compares SPI-LDFAI with SPEI-LDFAI.

**Input**
- Crop yield data (`merged_yield_data.nc`)
- Crop phenology data (`Soybeans.crop.calendar.fill.nc`)
- Monthly SPI-LDFAI and SPEI-LDFAI data (`.nc`)

**Output** (generated separately for |LDFAI| > 4–9)
- `*_correlation_summary.csv`: Correlations, significance, and sample sizes
- `*_spi_spei_compare.csv`: SPI vs. SPEI comparison
- `*_pairs.csv`: Paired LDFAI and yield residual samples
- `*_count_check.csv`: SPI/SPEI sample count consistency check
