#!/usr/bin/env python3
"""
Analyze computational stages in surveyed papers.
Creates visualizations showing distribution over time and by stage.
Uses the same dataset approach as extract_latex_stats.py
"""

import json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter
import pandas as pd

# Configuration
DATA_DIR = os.path.join(os.getcwd(), 'data')
HUMAN_DATASET_FILE = os.path.join(DATA_DIR, 'human_dataset.jsonl')
SUBSET_DATASET_FILE = os.path.join(DATA_DIR, 'subset_dataset.jsonl')
FULLSET_DATASET_FILE = os.path.join(DATA_DIR, 'fullset_dataset.jsonl')
OUTPUT_DIR = os.path.join(DATA_DIR, 'graphs', 'computational_stages')

# Canonical computational stages and their colors
STAGE_COLOR_MAP = {
    'Data Collection': '#1F4E79',           # Darker blue
    'Data Pre-Processing': '#ED7D31',       # Orange
    'Sequence Representation': '#70AD47',   # Green
    'Analysis & Classification': '#FFC000', # Yellow/Gold
    'Generation': '#5B9BD5'                 # Lighter blue
}

# Normalize stage names from both human/subset format and fullset format
STAGE_NORMALIZATION = {
    # human/subset names
    'Pre-processing': 'Data Pre-Processing',
    'Data Pre-Processing': 'Data Pre-Processing',
    'Data Collection': 'Data Collection',
    'Sequence Representation': 'Sequence Representation',
    'Meaning Identification': 'Analysis & Classification',
    'Generation': 'Generation',
    # fullset names
    'Preprocessing': 'Data Pre-Processing',
    'Analysis & Classification': 'Analysis & Classification',
    'Foundation Model': 'Generation',
}


def load_dataset_file(filepath):
    """Load entries from a JSONL dataset file."""
    entries = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return entries


def load_data(sources):
    """Load data from specified sources."""
    source_map = {
        'human': HUMAN_DATASET_FILE,
        'subset': SUBSET_DATASET_FILE,
        'fullset': FULLSET_DATASET_FILE
    }
    
    # Deduplicate by ID
    seen_ids = set()
    unique_entries = []
    
    for source in sources:
        if source in source_map:
            file_entries = load_dataset_file(source_map[source])
            for e in file_entries:
                eid = e.get('id')
                if eid and eid not in seen_ids:
                    seen_ids.add(eid)
                    unique_entries.append(e)
                elif not eid:
                    unique_entries.append(e)
                    
    return unique_entries


def get_year(entry):
    """Extract year from entry.

    Handles both human/subset format (``year``) and fullset format (``publication_year``).
    """
    year_str = str(entry.get('year', '') or entry.get('publication_year', '') or '').strip()
    try:
        y = int(year_str.split('-')[0]) if year_str else None
        if y and 1900 <= y <= 2100:
            return y
        return None
    except ValueError:
        return None


def normalize_stage_name(stage):
    """Normalize stage names to standard format.

    Handles noisy fullset names like 'Analysis & Classification: ML classifiers, ...'
    by extracting the base name before any colon.
    """
    # Strip the stage name; fullset entries sometimes have descriptions after a colon
    base = stage.split(':')[0].strip()
    return STAGE_NORMALIZATION.get(base, base)


# Valid canonical stage names (anything else is excluded)
VALID_STAGES = set(STAGE_COLOR_MAP.keys())


def extract_stage_data(entries):
    """Extract computational stage data from entries.

    Handles two data formats:
    - human/subset: ``"computational_stages": ["Data Collection", ...]`` (list of strings)
    - fullset: ``"stages": [{"stage": "Data Collection", "evidence": "..."}, ...]`` (list of dicts)

    Returns:
        stage_data: dict mapping stage name -> list of paper info dicts
        annotated_count: number of papers that had stage annotations
        total_count: total number of papers in the dataset
    """
    stage_data = defaultdict(list)

    annotated_count = 0

    for entry in entries:
        year = get_year(entry)

        # Try both field names / formats
        raw_stages = entry.get('computational_stages', [])
        if not raw_stages:
            raw_stages = entry.get('stages', [])

        if not raw_stages:
            continue

        # Extract stage name strings from either format
        stage_names = []
        for item in raw_stages:
            if isinstance(item, dict):
                s = item.get('stage', '')
            else:
                s = item
            if s:
                stage_names.append(s)

        if not stage_names:
            continue

        # Normalize and deduplicate stages per paper
        normalized_stages = set()
        for stage in stage_names:
            normalized = normalize_stage_name(stage)
            # Only keep recognized canonical stages
            if normalized in VALID_STAGES:
                normalized_stages.add(normalized)

        if normalized_stages:
            annotated_count += 1

        # Add to stage data
        for stage in normalized_stages:
            stage_data[stage].append({
                'year': year,
                'title': entry.get('title', 'Unknown'),
                'id': entry.get('id', 'Unknown')
            })

    return stage_data, annotated_count, len(entries)


def create_distribution_over_years_plot(stage_data, output_path, dataset_name, annotated_count=None, total_count=None):
    """Create a stacked area plot showing distribution over years."""
    # Prepare data for plotting
    year_stage_counts = defaultdict(lambda: defaultdict(int))
    
    for stage, papers in stage_data.items():
        for paper in papers:
            if paper['year'] is not None and paper['year'] >= 2000:
                year_stage_counts[paper['year']][stage] += 1
    
    # Convert to DataFrame
    years = sorted(year_stage_counts.keys())
    stages = sorted(stage_data.keys())
    
    if not years:
        print(f"Warning: No data with years >= 2000 for {dataset_name}")
        return None
    
    data = []
    for year in years:
        row = {'Year': year}
        for stage in stages:
            row[stage] = year_stage_counts[year][stage]
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create figure - square aspect ratio, TikZ-like styling
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get colors in the order of stages
    colors = [STAGE_COLOR_MAP.get(stage, '#808080') for stage in stages]
    
    # Create stacked area plot with TikZ-like styling
    ax.stackplot(df['Year'], 
                 *[df[stage] for stage in stages],
                 labels=stages,
                 colors=colors,
                 alpha=0.7,
                 edgecolor='gray',
                 linewidth=0.8)
    
    # TikZ-like axis styling
    base_fontsize = 20
    ax.set_xlabel('Year', fontsize=base_fontsize, fontweight='normal')
    ax.set_ylabel('Number of Papers', fontsize=base_fontsize, fontweight='normal')
    subtitle = dataset_name
    if annotated_count is not None and total_count is not None:
        subtitle = f'{dataset_name} — {annotated_count}/{total_count} papers with stage annotations'
    ax.set_title(f'Distribution of Computational Stages Over Years\n({subtitle})',
                 fontsize=base_fontsize, fontweight='normal', pad=15)
    
    # Clean, TikZ-like legend
    ax.legend(loc='upper left', frameon=True, fontsize=base_fontsize, 
              framealpha=0.95, edgecolor='gray', facecolor='white')
    
    # TikZ-like grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # Clean axis appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Set x-axis to start at 2000
    ax.set_xlim(left=2000, right=max(years) + 1)
    
    # Clean tick styling
    ax.tick_params(colors='black', labelsize=base_fontsize)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved distribution over years plot to: {output_path}")
    plt.close()
    
    return df


def create_papers_by_stage_plot(stage_data, output_path, dataset_name, annotated_count=None, total_count=None):
    """Create a bar plot showing number of papers by computational stage."""
    # Count papers per stage
    stage_counts = {stage: len(papers) for stage, papers in stage_data.items()}
    
    if not stage_counts:
        print(f"Warning: No stage data for {dataset_name}")
        return None
    
    # Sort by count (descending)
    sorted_stages = sorted(stage_counts.items(), key=lambda x: x[1], reverse=True)
    stages_original = [s[0] for s in sorted_stages]
    counts = [s[1] for s in sorted_stages]
    
    # Format stage names: split two-word labels onto 2 lines for larger font
    def format_stage_label(stage):
        """Split stage name into two lines if it has multiple words."""
        words = stage.split()
        if len(words) > 1:
            mid = len(words) // 2
            return '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
        return stage
    
    stages = [format_stage_label(s) for s in stages_original]
    
    # Create figure - square aspect ratio, TikZ-like styling
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get colors in the order of stages
    colors = [STAGE_COLOR_MAP.get(stage, '#808080') for stage in stages_original]
    
    # Create bar plot with TikZ-like styling
    bars = ax.barh(stages, counts, color=colors, alpha=0.7, 
                   edgecolor='gray', linewidth=0.8)
    
    # TikZ-like axis styling
    base_fontsize = 18
    ax.set_xlabel('Number of Papers', fontsize=base_fontsize, fontweight='normal')
    ax.set_ylabel('Computational Stage', fontsize=base_fontsize, fontweight='normal')
    subtitle = dataset_name
    if annotated_count is not None and total_count is not None:
        subtitle = f'{dataset_name} — {annotated_count}/{total_count} papers with stage annotations'
    ax.set_title(f'Number of Papers by Computational Stage\n({subtitle})',
                 fontsize=base_fontsize, fontweight='normal', pad=15)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + 0.5, i, str(count), 
               va='center', ha='left', fontsize=base_fontsize, fontweight='normal')
    
    # TikZ-like grid
    ax.grid(True, axis='x', alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # Clean axis appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Clean tick styling
    ax.tick_params(colors='black', labelsize=base_fontsize)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved papers by stage plot to: {output_path}")
    plt.close()
    
    return dict(sorted_stages)


def generate_summary_statistics(stage_data, dataset_name, annotated_count, total_count):
    """Generate and print summary statistics."""
    print("\n" + "="*70)
    print(f"SUMMARY STATISTICS - {dataset_name}")
    print("="*70)

    total_stage_mentions = sum(len(papers) for papers in stage_data.values())
    print(f"\nTotal papers in dataset: {total_count}")
    print(f"Papers with stage annotations: {annotated_count} ({100*annotated_count/total_count:.1f}%)")
    print(f"Total stage mentions (papers can have multiple): {total_stage_mentions}")
    
    print("\n" + "-"*70)
    print("Papers per stage:")
    print("-"*70)
    for stage, papers in sorted(stage_data.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {stage}: {len(papers)}")
    
    # Year range statistics
    print("\n" + "-"*70)
    print("Year range by stage:")
    print("-"*70)
    for stage, papers in stage_data.items():
        years = [p['year'] for p in papers if p['year'] is not None]
        if years:
            print(f"  {stage}: {min(years)} - {max(years)}")
        else:
            print(f"  {stage}: No years available")
    
    print("\n" + "="*70)


def process_dataset(sources, dataset_name, suffix):
    """Process a dataset and generate all visualizations."""
    print("\n" + "="*70)
    print(f"PROCESSING: {dataset_name}")
    print("="*70)
    
    # Load data
    entries = load_data(sources)
    print(f"Total papers loaded: {len(entries)}")
    
    # Extract stage data
    stage_data, annotated_count, total_count = extract_stage_data(entries)

    # Generate summary statistics
    generate_summary_statistics(stage_data, dataset_name, annotated_count, total_count)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate visualizations
    print("\n" + "-"*70)
    print("GENERATING VISUALIZATIONS")
    print("-"*70)

    # Distribution over years (stacked plot)
    dist_png_path = os.path.join(OUTPUT_DIR, f'computational_stages_timeline_{suffix}.png')
    df_dist = create_distribution_over_years_plot(stage_data, dist_png_path, dataset_name, annotated_count, total_count)

    # Papers by stage (bar chart)
    stage_png_path = os.path.join(OUTPUT_DIR, f'papers_by_stage_{suffix}.png')
    stage_counts = create_papers_by_stage_plot(stage_data, stage_png_path, dataset_name, annotated_count, total_count)
    
    print("\n" + "="*70)
    print(f"COMPLETE - {dataset_name}")
    print("="*70)


def main():
    """Main function to process both dataset combinations."""
    print("\n" + "="*70)
    print("COMPUTATIONAL STAGES ANALYSIS")
    print("="*70)
    
    # Process human + subset
    process_dataset(['human', 'subset'], 'Human + Subset', 'human_subset')
    
    print("\n\n")
    
    # Process human + subset + fullset
    process_dataset(['human', 'subset', 'fullset'], 'Human + Subset + Fullset', 'human_subset_fullset')
    
    print("\n" + "="*70)
    print("ALL PROCESSING COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - computational_stages_timeline_human_subset.png")
    print("  - papers_by_stage_human_subset.png")
    print("  - computational_stages_timeline_human_subset_fullset.png")
    print("  - papers_by_stage_human_subset_fullset.png")


if __name__ == '__main__':
    main()
