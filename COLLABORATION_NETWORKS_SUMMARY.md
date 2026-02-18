# Collaboration Network Analysis - Summary

## Overview

Updated the LaTeX statistics generation and created new collaboration network visualizations for the Comparative Animal Linguistics survey.

## Changes Made

### 1. Updated `extract_latex_stats.py`

#### New Features Added

- **Country/University Normalization**: Added comprehensive normalization functions to handle naming variations
  - `normalize_country()`: Merges "USA", "United States", "US" → "United States", etc.
  - `normalize_university()`: Merges variations like "MIT", "Massachusetts Institute of Technology" → "MIT"

- **Top 15 Universities Graph**: New function `generate_university_stats()`
  - Generates LaTeX coordinates for horizontal bar chart
  - Uses normalized university names
  - Format: same as country stats (yticklabels + coordinates)

- **Collaboration Network Analysis**: New function `build_collaboration_network()`
  - Analyzes which universities/countries collaborate on papers together
  - Builds network with nodes (entities) and edges (collaborations)
  - Uses force-directed layout algorithm for better visualization
  - Generates coordinates and edge weights for LaTeX/TikZ graphs

#### Updated Functions

- **`generate_country_stats()`**: Now uses `normalize_country()` for consistent naming
- **`run()`**: Added new output sections:
  - Section 3b: Top 15 Universities
  - Section 6: University Collaboration Network
  - Section 7: Country Collaboration Network

### 2. Created `generate_collaboration_networks.py`

A new visualization script that creates publication-quality network graphs:

#### Features

- **Beautiful Network Layouts**: Force-directed spring layout algorithm
  - Repulsive forces keep nodes apart
  - Attractive forces pull collaborating nodes together
  - Iterative optimization for clean layout

- **Visual Encoding**:
  - **Node size**: Proportional to number of papers
  - **Node color**: Gradient based on paper count (viridis colormap)
  - **Edge width**: Proportional to number of collaborations
  - **Edge opacity**: Stronger for more collaborations

- **Outputs**:
  - `data/graphs/collaboration_networks/university_network.png` (+ PDF)
  - `data/graphs/collaboration_networks/country_network.png` (+ PDF)

## Key Insights from the Data

### University Collaborations (Top Collaborations)

1. **University of St Andrews ↔ Woods Hole Oceanographic**: 8 joint papers
2. **MIT ↔ Woods Hole Oceanographic**: 7 joint papers
3. **Aarhus University ↔ University of St Andrews**: 5 joint papers
4. **Max Planck Institute ↔ University of St Andrews**: 4 joint papers
5. **University of Cambridge ↔ University of Zurich**: 4 joint papers

**Total**: 15 top universities, 61 collaboration edges

### Country Collaborations (Top Collaborations)

1. **United Kingdom ↔ United States**: 77 joint papers
2. **Germany ↔ United States**: 51 joint papers
3. **Canada ↔ United States**: 41 joint papers
4. **Switzerland ↔ United Kingdom**: 40 joint papers
5. **France ↔ United Kingdom**: 35 joint papers

**Total**: 15 top countries, 96 collaboration edges

## Top 15 Universities by Papers

1. University of St Andrews - 90 papers
2. University of Zurich - 51 papers
3. Cornell University - 49 papers
4. Max Planck Institute - 38 papers
5. Woods Hole Oceanographic Institution - 26 papers
6. Harvard University - 24 papers
7. UC San Diego - 24 papers
8. MIT - 22 papers
9. University of Vienna - 22 papers
10. Aarhus University - 21 papers
11. University of Cambridge - 21 papers
12. Rockefeller U. - 20 papers
13. UCLA - 20 papers
14. Smithsonian Tropical Research Institute - 20 papers
15. University of Oxford - 19 papers

## Top 15 Countries by Papers

1. United States - 638 papers
2. United Kingdom - 254 papers
3. Germany - 181 papers
4. France - 140 papers
5. Australia - 93 papers
6. Switzerland - 88 papers
7. Canada - 87 papers
8. Japan - 74 papers
9. China - 59 papers
10. Brazil - 58 papers
11. Netherlands - 55 papers
12. Italy - 53 papers
13. Denmark - 45 papers
14. Austria - 35 papers
15. Spain - 33 papers

## Usage

### Generate LaTeX Stats

```bash
python3 extract_latex_stats.py > latex_output.txt
```

This outputs LaTeX-ready coordinates for:

- Linguistic features
- Papers by period
- Top 15 countries
- **NEW: Top 15 universities**
- Top 15 species
- Animal categories timeline
- **NEW: University collaboration network (nodes + edges)**
- **NEW: Country collaboration network (nodes + edges)**

### Generate Network Visualizations

```bash
python3 generate_collaboration_networks.py
```

This creates:

- PNG files (300 DPI) for presentations/papers
- PDF files for LaTeX inclusion
- Both university and country collaboration networks

## Network Visualization Design

The visualizations are designed to be:

- **Readable**: Clear labels, good spacing, large fonts
- **Informative**: Multiple visual channels (size, color, width)
- **Publication-ready**: High DPI, clean styling, proper legends
- **Print-friendly**: Works well in grayscale

Unlike the reference broken code, these networks:

- Use proper force-directed layout (not random)
- Scale appropriately (nodes don't overlap)
- Have readable labels (not truncated or overlapping)
- Show edge weights visually (not just in raw data)
- Use professional color schemes

## Technical Notes

### Force-Directed Layout Algorithm

- **Initial placement**: Circular layout for stability
- **Repulsive forces**: Keep all nodes separated (inverse square law)
- **Attractive forces**: Pull collaborating nodes together (log-weighted by collaboration count)
- **Damping**: Prevents oscillation, ensures convergence
- **Iterations**: 50-100 iterations for good layouts

### Normalization Benefits

- Reduced 1,814 unique affiliations → 1,603 (211 duplicates merged)
- Reduced 90 unique countries → 79 (11 variants merged)
- Ensures consistent counting and visualization
- Makes collaboration detection more accurate

## Files Modified/Created

### Modified

1. `/Users/ndesai-air/Documents/GitHub/clsurveytool/extract_latex_stats.py`
   - Added normalization functions
   - Added university stats generation
   - Added collaboration network building
   - Updated run() to output new data

### Created

1. `/Users/ndesai-air/Documents/GitHub/clsurveytool/generate_collaboration_networks.py`
   - Standalone visualization script
   - Creates publication-quality network graphs
   - Forces-directed layout implementation

2. `/Users/ndesai-air/Documents/GitHub/clsurveytool/data/graphs/collaboration_networks/`
   - university_network.png
   - university_network.pdf
   - country_network.png
   - country_network.pdf

## Next Steps

To use these networks in LaTeX:

1. Include the PDF files directly: `\includegraphics{university_network.pdf}`
2. Or use the coordinate data from `extract_latex_stats.py` output to create TikZ graphs
3. The TikZ approach gives more control but requires more LaTeX coding

The PNG/PDF visualizations are ready to use immediately!
