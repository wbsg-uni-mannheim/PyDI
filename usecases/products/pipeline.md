# Normalization Pipeline Summary

## Attribute Filtering
- Applied a **15% coverage threshold** to 66 initial columns.

### Exceptions
- Retained `write_speed` (for symmetry).
- Retained `weight`, `dimensions`, and `color` (Professor Bizer’s requirement).

## Brand & Category
- Standardized casing.
- Unified `product_type` into four classes:
  - GPU
  - SSD
  - HDD
  - USB_STICK

## Weight Normalization
- Converted mixed units (`kg`, `oz`) to **grams (g)**.
- Logic used product-type context.  
  - Example: `1.5 kg GPU → 1500 g`

## Storage Normalization
- Standardized all capacity values to **gigabytes (GB)**.

### Logic
- Multiplied values `< 64` by **1000** for HDDs/SSDs to resolve TB/GB confusion.

## Price Cleaning
- Resolved scientific notation  
  - Example: `1.5E3 → 1500.0`
- Fixed ZAR formatting.
- Stripped currency prefixes (`SAR`, `MX$`).
- Fixed European decimal commas.

## Type & Schema Finalization

### Casting
- Technical specs (`vram`, `storage`, `speeds`) cast to **Int64** to remove `.0` decimals.

### Renaming
- Shifted units to headers:
  - `vram_gb`
  - `storage_gb`
  - `read_speed_mb_s`
  - `weight_g`

---

# Next Steps: Entity Matching

- Use the **WDCproducts repo**.
- Filter it to build the **ground truth dataset for evaluation**.
- Manually create **100 example fusion sets**.