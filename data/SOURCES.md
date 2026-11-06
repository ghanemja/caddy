# Data sources

- `bspt_nipple.csv`: NERO Pipe Fittings, "Reducing Hex Nipple BSPT & NPT 3000LB 316/L" datasheet
  (https://www.nero.co.uk/files/d203df08-f0df-44b2-a27b-8da38276f1dd/REDUCING%20HEX%20NIPPLE%20BSPT%20NPT.pdf).
  Columns: A = hex across flats, W = overall length, C = hex length, E = large-end thread length,
  F = small-end thread length, b = bore. Rows with a BSPT product code only, sizes up to 1".
  Values are minimums ("MIN") on the sheet; read from the drawing image; F + C + E = W holds on every row.
- `iso7_1.csv`: ISO 7-1:1994 Table 1 (external thread R): major diameter at the gauge plane, pitch, nominal gauge
  length (small end to gauge plane), minimum useful thread for nominal gauge length. Taper 1:16 on diameter.
  Read from the public preview at standards.iteh.ai; check against a clean copy before relying on it.

## Stability library (`data/library/stability_index.json`)

2,166 NVIDIA SimReady objects with their measured friction, slide angle and archetype, taken from
ghanemja/behaviors (`out/site/data.json`). SimReady assets: NVIDIA, CC BY 4.0. The index holds numbers only (no
meshes, no textures). Regenerate with `python tools/build_stability_library.py <data.json> data/library/stability_index.json`.

