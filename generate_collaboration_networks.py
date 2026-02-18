#!/usr/bin/env python3
"""
Generate publication-quality network visualizations for university and country collaborations.
This creates beautiful, readable network graphs with proper layouts and styling.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import Counter
from itertools import combinations
from pathlib import Path
import math

# Import normalization functions from extract_latex_stats
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from extract_latex_stats import normalize_country, normalize_university, load_data

# Configuration
OUTPUT_DIR = Path('data/graphs/collaboration_networks')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_network_data(entries, entity_type='university', top_n=15):
    """Build collaboration network data."""
    edge_weights = Counter()
    node_papers = Counter()
    
    for e in entries:
        affiliations = e.get('affiliations', [])
        if not affiliations:
            continue
        
        entities_in_paper = set()
        for aff in affiliations:
            if entity_type == 'university':
                entity = aff.get('university', '').strip()
                if entity:
                    normalized = normalize_university(entity)
                    if normalized:
                        entities_in_paper.add(normalized)
            else:  # country
                entity = aff.get('country', '').strip()
                if entity:
                    normalized = normalize_country(entity)
                    if normalized:
                        entities_in_paper.add(normalized)
        
        for entity in entities_in_paper:
            node_papers[entity] += 1
        
        if len(entities_in_paper) > 1:
            for e1, e2 in combinations(sorted(entities_in_paper), 2):
                edge_weights[(e1, e2)] += 1
    
    top_entities = [entity for entity, count in node_papers.most_common(top_n)]
    
    filtered_edges = {}
    for (e1, e2), weight in edge_weights.items():
        if e1 in top_entities and e2 in top_entities:
            filtered_edges[(e1, e2)] = weight
    
    # NetworkX layout - robust and standard
    import networkx as nx
    
    G = nx.Graph()
    G.add_nodes_from(top_entities)
    
    # Add weighted edges
    for (e1, e2), weight in filtered_edges.items():
        G.add_edge(e1, e2, weight=weight)
        
    # Use spring layout but IGNORE weights for positioning
    # This ensures clumping is determined by topology, not strength
    # weight=None treats all edges equally layout-wise
    # k=1.2 spreads nodes out even more to prevent overlapping and clipping
    # scale=2.0 restores original spread
    pos = nx.spring_layout(G, k=1.2, iterations=100, weight=None, seed=42, scale=2.0)
    
    # Convert to expected format
    positions = {node: {'x': coords[0], 'y': coords[1]} for node, coords in pos.items()}
    
    return positions, filtered_edges, node_papers


def create_network_visualization(entries, entity_type='university', title='Collaboration Network'):
    """Create a beautiful network visualization."""
    positions, edges, node_papers = build_network_data(entries, entity_type, top_n=15)
    
    # Text helper for publication quality labels
    import textwrap
    
    # Increased figure size to help fit everything
    fig, ax = plt.subplots(figsize=(26, 24))
    
    # Draw edges first (so they appear behind nodes)
    max_weight = max(edges.values()) if edges else 1
    num_buckets = 16
    
    for (e1, e2), weight in edges.items():
        x1, y1 = positions[e1]['x'], positions[e1]['y']
        x2, y2 = positions[e2]['x'], positions[e2]['y']
        
        # Split into 16 discrete buckets
        if max_weight > 1:
            ratio = (weight - 1) / (max_weight - 1)
            bucket = int(ratio * (num_buckets - 1))
            bucket = max(0, min(num_buckets - 1, bucket))
        else:
            bucket = num_buckets - 1
            
        # Thicker lines based on bucket
        # Width range: 2.0 (thinner start) to 18.0 (more apparent strength difference)
        line_width = 2.0 + 16.0 * (bucket / (num_buckets - 1))
        alpha = 0.6 + 0.3 * (bucket / (num_buckets - 1))
        
        ax.plot([x1, x2], [y1, y2], 'gray', linewidth=line_width, alpha=alpha, zorder=1)
    
    # Draw nodes - HUGE for readability
    node_list = list(positions.keys())
    max_papers = max(node_papers.values()) if node_papers else 1
    
    for entity in node_list:
        x, y = positions[entity]['x'], positions[entity]['y']
        papers = node_papers[entity]
        
        # Significantly larger node size to contain text comfortably
        # Base size 16000 + up to 14000 more (increased base size)
        node_size = 16000 + 14000 * (papers / max_papers)
        
        # Color gradient based on papers - using Wistia (Yellow-Orange) for better text contrast
        color_intensity = papers / max_papers
        color = plt.cm.Wistia(color_intensity)
        
        ax.scatter(x, y, s=node_size, c=[color], alpha=0.9, edgecolors='black', 
                  linewidths=4, zorder=2)
        
        # Smart label wrapping
        # Wrap to ~10 chars per line to force "United Kingdom" to split and generally fit better
        label = textwrap.fill(entity, width=10, break_long_words=False)
        
        # Font size adjustment based on label length but kept large enough to read
        font_size = 14 if len(label) > 40 else 16
        
        ax.text(x, y, label, fontsize=font_size, ha='center', va='center', 
               fontweight='bold', zorder=3, color='black')
    
    # Remove title as requested
    ax.axis('equal')
    ax.axis('off')
    
    # Add EXTRA margins to ensure full circles are visible and nothing is cut off
    ax.margins(0.3)
    
    # NO LEGEND as requested
    
    plt.tight_layout()
    return fig


def main():
    print("Loading data...")
    entries = load_data(['human', 'subset'])
    print(f"Loaded {len(entries)} papers")
    
    # Generate university network
    print("\nGenerating university collaboration network...")
    fig = create_network_visualization(entries, entity_type='university', 
                                      title='University Collaboration Network in Comparative Animal Linguistics')
    output_path = OUTPUT_DIR / 'university_network.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved to: {output_path}")
    
    pdf_path = OUTPUT_DIR / 'university_network.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved to: {pdf_path}")
    plt.close()
    
    # Generate country network
    print("\nGenerating country collaboration network...")
    fig = create_network_visualization(entries, entity_type='country', 
                                      title='Country Collaboration Network in Comparative Animal Linguistics')
    output_path = OUTPUT_DIR / 'country_network.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved to: {output_path}")
    
    pdf_path = OUTPUT_DIR / 'country_network.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved to: {pdf_path}")
    plt.close()
    
    print("\n✅ All visualizations generated successfully!")


if __name__ == '__main__':
    main()
