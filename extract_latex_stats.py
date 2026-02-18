
import json
import os
import re
from collections import defaultdict, Counter
from itertools import combinations
import unicodedata

# Configuration
# Run from repo root
DATA_DIR = os.path.join(os.getcwd(), 'data')
HUMAN_DATASET_FILE = os.path.join(DATA_DIR, 'human_dataset.jsonl')
SUBSET_DATASET_FILE = os.path.join(DATA_DIR, 'subset_dataset.jsonl')
FULLSET_DATASET_FILE = os.path.join(DATA_DIR, 'fullset_dataset.jsonl')

# Constants
LINGUISTIC_FEATURES = [
    "Vocal Auditory Channel and Turn-taking",
    "Broadcast and Direct Reception",
    "Reference and Displacement",
    "Specialization",
    "Arbitrariness and Duality of Patterns",
    "Discreteness and Syntax",
    "Recursion",
    "Semanticity",
    "Prevarication",
    "Openness",
    "Tradition and Cultural Transmission",
    "Learnability"
]

# Order for the chart (as in user's prompt)
FEATURE_ORDER = [
    "Prevarication", 
    "Learnability", 
    "Recursion",
    "Reference and Displacement", 
    "Specialization",
    "Arbitrariness and Duality of Patterns", 
    "Openness",
    "Tradition and Cultural Transmission", 
    "Broadcast and Direct Reception",
    "Vocal Auditory Channel and Turn-taking", 
    "Semanticity",
    "Discreteness and Syntax"
]

FEATURE_MAP_LATEX = {
    "Vocal Auditory Channel and Turn-taking": "Vac. AC & TT",
    "Broadcast and Direct Reception": "BT & DT",
    "Reference and Displacement": "Disp. & Ref.",
    "Specialization": "Spec.",
    "Arbitrariness and Duality of Patterns": "Arb. & DOP",
    "Discreteness and Syntax": "Disc. & Syn.",
    "Recursion": "Recur.",
    "Semanticity": "Semant.",
    "Prevarication": "Prev.",
    "Openness": "Open.",
    "Tradition and Cultural Transmission": "Trad. & CT",
    "Learnability": "Learn."
}

# --- Normalization Functions ---

def normalize_country(country):
    """Normalize country names to handle variations."""
    if not country:
        return ""
    
    c = country.strip()
    c_lower = c.lower()
    
    # USA variations
    if c_lower in ['usa', 'united states', 'united states of america', 'the united states', 
                   'the united states of america', 'u.s.a.', 'u.s.', 'us']:
        return "United States"
    # UK variations
    elif c_lower in ['uk', 'united kingdom', 'england', 'scotland', 'wales', 
                     'northern ireland', 'great britain', 'the united kingdom', 'u.k.']:
        return "United Kingdom"
    # China variations
    elif c_lower in ['china', 'prc', "people's republic of china"]:
        return "China"
    # Russia variations
    elif c_lower in ['russia', 'russian federation']:
        return "Russia"
    # Korea variations
    elif c_lower in ['south korea', 'republic of korea', 'korea']:
        return "South Korea"
    # Netherlands variations
    elif c_lower in ['netherlands', 'the netherlands']:
        return "Netherlands"
    # Czech variations
    elif c_lower in ['czech republic', 'czechia']:
        return "Czech Republic"
    # Brunei variations
    elif c_lower in ['brunei', 'brunei darussalam']:
        return "Brunei"
    
    # Return original if no normalization needed
    return c

def normalize_university(university):
    """Normalize university names to handle variations (abbreviated version for top institutions)."""
    if not university:
        return ""
    
    u = university.strip()
    u_lower = u.lower()
    u_clean = u_lower.replace(',', '').replace('.', '').replace('-', ' ').replace('  ', ' ').strip()
    
    # St Andrews
    if 'st andrews' in u_clean or 'st. andrews' in u_clean or 'saint andrews' in u_clean:
        if 'scottish oceans' not in u_clean and 'sea mammal' not in u_clean:
            return "University of St Andrews"
    # Zurich
    if ('zurich' in u_clean or 'zürich' in u_clean) and 'eth' not in u_clean and 'university' in u_clean:
        return "University of Zurich"
    # Cambridge
    if 'cambridge' in u_clean and 'university' in u_clean:
        return "University of Cambridge"
    # Oxford
    if 'oxford' in u_clean and 'university' in u_clean:
        return "University of Oxford"
    # MIT
    if u_clean in ['mit', 'mit csail'] or 'massachusetts institute of technology' in u_clean:
        return "MIT"
    # UC San Diego
    if 'university of california' in u_clean or 'uc ' in u_clean:
        if 'san diego' in u_clean or 'ucsd' in u_clean:
            return "UC San Diego"
        elif 'los angeles' in u_clean or 'ucla' in u_clean:
            return "UCLA"
        elif 'berkeley' in u_clean:
            return "UC Berkeley"
        elif 'davis' in u_clean:
            return "UC Davis"
    # Rockefeller
    if 'rockefeller' in u_clean:
        return "Rockefeller U."
    # Max Planck
    if 'max planck' in u_clean:
        if 'animal behav' in u_clean:
            return "MPI Animal Behavior"
        elif 'evolutionary anthropology' in u_clean:
            return "MPI Evol. Anthro."
        elif 'psycholinguistics' in u_clean:
            return "MPI Psycholinguistics"
        return "Max Planck Institute"
    # Cornell
    if 'cornell' in u_clean and 'ornithology' not in u_clean:
        return "Cornell University"
    # German Primate Center
    if 'german primate center' in u_clean or 'deutsches primatenzentrum' in u_clean:
        return "German Primate Ctr."
    # CNRS
    if u_clean == 'cnrs' or 'centre national de la recherche' in u_clean:
        return "CNRS"
    
    # Return original if no specific normalization
    return u

# --- Helpers ---

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
    all_entries = []
    source_map = {
        'human': HUMAN_DATASET_FILE,
        'subset': SUBSET_DATASET_FILE,
        'fullset': FULLSET_DATASET_FILE
    }
    # Deduplicate by ID just in case
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
    year_str = str(entry.get('year', '')).strip()
    try:
        # Handle "2023" or "2023-01-01"
        y = int(year_str.split('-')[0]) if year_str else None
        if y and 1900 <= y <= 2100:
            return y
        return None
    except ValueError:
        return None

# --- Report Generators ---

def generate_linguistic_feature_stats(entries):
    counts = Counter()
    for e in entries:
        for feat in e.get('linguistic_features', []):
            counts[feat] += 1
            
    # Generate coordinates for LaTeX: (index, count)
    coords = []
    for i, feat in enumerate(FEATURE_ORDER, 1):
        count = counts[feat] # Default 0
        coords.append(f"({i},{count})")
    
    return " ".join(coords)

def generate_papers_by_period_stats(entries):
    # Bin logic: 5-year buckets starting 1971
    # 71-75 (1), 76-80 (2), ... 21-25 (11)
    
    # Bucket 1: 1971-1975
    # Bucket 11: 2021-2025
    
    buckets = defaultdict(int)
    
    for e in entries:
        year = get_year(e)
        if year:
            if year < 1971: continue 
            if year > 2025: continue
            
            # 1971 -> bucket 0 (internal) -> 1 (latex)
            # (year - 1971) // 5
            b_idx = (year - 1971) // 5
            buckets[b_idx + 1] += 1
            
    # Generate coords for 1..11
    coords = []
    for i in range(1, 12):
        count = buckets[i]
        coords.append(f"({i},{count})")
        
    return " ".join(coords)

def generate_country_stats(entries):
    """Generate top 15 countries by number of papers (with normalization)."""
    cnt = Counter()
    for e in entries:
        # Count each country once per paper
        countries_in_paper = set()
        for aff in e.get('affiliations', []):
            c = aff.get('country', '').strip()
            if c:
                # Use normalization function
                normalized = normalize_country(c)
                if normalized:
                    countries_in_paper.add(normalized)
        
        for c in countries_in_paper:
            cnt[c] += 1
            
    # Top 15
    top15 = cnt.most_common(15)
    # Sort for horizontal bar chart (ascending count)
    top15.sort(key=lambda x: x[1])
    
    # Coords: (count, index) for xbar
    coords = []
    labels = []
    for i, (country, count) in enumerate(top15, 1):
        coords.append(f"({count},{i})")
        labels.append(country)
        
    return labels, coords

def generate_species_stats(entries):
    # Top 15 "specialized_species"
    # Again, per paper unique?
    # "A paper discusses Zebra Finch and Babbler" -> +1 Zebra Finch, +1 Babbler.
    
    sp_cnt = Counter()
    sp_cats = {} # Map species -> most common category
    
    for e in entries:
        # Unique species per paper
        paper_species = set()
        cats = e.get('species_categories', [])
        
        if not cats: continue # Can't categorize if no category
        
        # Taking raw strings
        for sp in e.get('specialized_species', []):
            sp_clean = sp.strip()
            if sp_clean:
                # Force Title Case for better merging (e.g. "whale" vs "Whale")
                # Handle edge cases like "CB's" ideally, but standard title() is a good heuristic
                # "CB's Monkey" -> "Cb'S Monkey" is annoying but better than duplicates.
                # Let's use a helper that preserves some things if needed, but title() is simplest loop fix.
                # Actually, let's just use title() for now.
                sp_clean = sp_clean.title()
                
                paper_species.add(sp_clean)
        
        for sp in paper_species:
            sp_cnt[sp] += 1
            # Track category association
            # If paper has multiple categories, it's ambiguous. 
            # We'll just collect all seen categories and pick most frequent later.
            if sp not in sp_cats: sp_cats[sp] = Counter()
            for c in cats:
                sp_cats[sp][c] += 1

    top15 = sp_cnt.most_common(15)
    top15.sort(key=lambda x: x[1])
    
    # Resolve category for each top species
    final_cats = {}
    for sp, _ in top15:
        if sp in sp_cats and sp_cats[sp]:
            final_cats[sp] = sp_cats[sp].most_common(1)[0][0]
        else:
            final_cats[sp] = "Other"
            
    # Map to User Latex Categories
    # "Primate", "Bird", "Marine Mammal", "Terr. Mammal", "Amphibian", "Reptile", "Insect", "Fish"
    # Categories in our dataset: "Terrestrial Mammal" -> "Terr. Mammal"
    
    cat_map = {
        "Terrestrial Mammal": "Terr. Mammal",
        "Marine Mammal": "Marine Mammal",
        "Primate": "Primate",
        "Bird": "Bird", 
        "Amphibian": "Amphibian",
        "Reptile": "Reptile", 
        "Insect": "Insect",
        "Fish": "Fish"
    }
    
    # Prepare coords per category plot
    # Logic: We have 15 bars. Each bar belongs to a specific category series. 
    # We need to output coordinates for each category series.
    # The y-coordinate is the index (1..15).
    # "symbolic y coords" will be the list of species names.
    
    cat_coords = defaultdict(list)
    ylabels = []
    
    for i, (sp, count) in enumerate(top15, 1):
        c_raw = final_cats.get(sp, "Other")
        c_latex = cat_map.get(c_raw, "Other")
        
        # Add to the correct series
        # Note: In the user's latex, they use (count, SpeciesName) with symbolic coords.
        # "addplot ... coordinates {(4,Orangutan) ...}"
        # So we use the name as Y, not index.
        
        cat_coords[c_latex].append(f"({count},{sp})")
        ylabels.append(sp)
        
    return ylabels, cat_coords

def generate_category_timeline(entries):
    # Stacked bar: 5-year periods
    # Periods: 1 (81-85) ... 9 (21-25)
    # Range 1981-2025
    
    # Start year: 1981
    # Period 1: 81-85
    # Period 9: 21-25
    
    # Data: period_idx -> category -> count
    data = defaultdict(lambda: defaultdict(int))
    
    for e in entries:
        year = get_year(e)
        if year:
            if year < 1981: continue
            if year > 2025: continue
            
            # Period index 1-based
            # (year - 1981) // 5 + 1
            p_idx = (year - 1981) // 5 + 1
            
            # "Relative contribution of each category"
            # Count each category mentioned in paper?
            seen_cats = set()
            for c in e.get('species_categories', []):
                seen_cats.add(c)
            
            for c in seen_cats:
                data[p_idx][c] += 1
                
    # Generate full plots for each category
    # Categories: Bird, Primate, Terrestrial Mammal, Marine Mammal, Amphibian, Insect, Reptile
    # Colors/Order from user latex
    
    all_cats = ["Bird", "Primate", "Terrestrial Mammal", "Marine Mammal", "Amphibian", "Insect", "Reptile"]
    
    plots = {}
    for cat in all_cats:
        coords = []
        for p in range(1, 10): # 1 to 9
            val = data[p][cat]
            coords.append(f"({p},{val})")
        plots[cat] = "\n    ".join(coords)
        
    return plots


def generate_university_stats(entries):
    """Generate top 15 universities by number of papers (with normalization)."""
    cnt = Counter()
    for e in entries:
        # Count each university once per paper
        universities_in_paper = set()
        for aff in e.get('affiliations', []):
            u = aff.get('university', '').strip()
            if u:
                # Use normalization function
                normalized = normalize_university(u)
                if normalized:
                    universities_in_paper.add(normalized)
        
        for u in universities_in_paper:
            cnt[u] += 1
            
    # Top 15
    top15 = cnt.most_common(15)
    # Sort for horizontal bar chart (ascending count)
    top15.sort(key=lambda x: x[1])
    
    # Coords: (count, index) for xbar
    coords = []
    labels = []
    for i, (university, count) in enumerate(top15, 1):
        coords.append(f"({count},{i})")
        labels.append(university)
        
    return labels, coords


def build_collaboration_network(entries, entity_type='university'):
    """
    Build collaboration network for universities or countries.
    Returns edge data for network visualization.
    entity_type: 'university' or 'country'
    """
    # Track collaborations (edges between entities on same paper)
    edge_weights = Counter()
    node_papers = Counter()
    
    for e in entries:
        affiliations = e.get('affiliations', [])
        if not affiliations:
            continue
        
        # Get unique entities in this paper
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
        
        # Count papers per entity
        for entity in entities_in_paper:
            node_papers[entity] += 1
        
        # Create edges for all pairs (collaborations)
        if len(entities_in_paper) > 1:
            for e1, e2 in combinations(sorted(entities_in_paper), 2):
                edge_weights[(e1, e2)] += 1
    
    # Get top N entities by paper count
    top_entities = [entity for entity, count in node_papers.most_common(15)]
    
    # Filter edges to only include top entities
    filtered_edges = {}
    for (e1, e2), weight in edge_weights.items():
        if e1 in top_entities and e2 in top_entities:
            filtered_edges[(e1, e2)] = weight
    
    # Build node positions using simple force-directed layout simulation
    # Place nodes in a circle initially, then adjust based on connections
    import math
    n = len(top_entities)
    positions = {}
    
    # Initial circular layout
    for i, entity in enumerate(top_entities):
        angle = 2 * math.pi * i / n
        positions[entity] = {
            'x': 5 * math.cos(angle),
            'y': 5 * math.sin(angle)
        }
    
    # Simple force-directed adjustment (a few iterations)
    for iteration in range(50):
        forces = {entity: {'x': 0, 'y': 0} for entity in top_entities}
        
        # Repulsive forces between all nodes
        for i, e1 in enumerate(top_entities):
            for e2 in top_entities[i+1:]:
                dx = positions[e2]['x'] - positions[e1]['x']
                dy = positions[e2]['y'] - positions[e1]['y']
                dist = math.sqrt(dx*dx + dy*dy) + 0.01
                
                # Repulsion
                force = 0.5 / (dist * dist)
                forces[e1]['x'] -= force * dx / dist
                forces[e1]['y'] -= force * dy / dist
                forces[e2]['x'] += force * dx / dist
                forces[e2]['y'] += force * dy / dist
        
        # Attractive forces for connected nodes
        for (e1, e2), weight in filtered_edges.items():
            dx = positions[e2]['x'] - positions[e1]['x']
            dy = positions[e2]['y'] - positions[e1]['y']
            dist = math.sqrt(dx*dx + dy*dy) + 0.01
            
            # Attraction proportional to weight
            force = 0.01 * dist * math.log(1 + weight)
            forces[e1]['x'] += force * dx / dist
            forces[e1]['y'] += force * dy / dist
            forces[e2]['x'] -= force * dx / dist
            forces[e2]['y'] -= force * dy / dist
        
        # Apply forces with damping
        damping = 0.5
        for entity in top_entities:
            positions[entity]['x'] += damping * forces[entity]['x']
            positions[entity]['y'] += damping * forces[entity]['y']
    
    # Normalize positions to fit in (-5, 5) range
    all_x = [pos['x'] for pos in positions.values()]
    all_y = [pos['y'] for pos in positions.values()]
    max_coord = max(max(abs(x) for x in all_x), max(abs(y) for y in all_y))
    if max_coord > 0:
        scale = 5.0 / max_coord
        for entity in positions:
            positions[entity]['x'] *= scale
            positions[entity]['y'] *= scale
    
    return {
        'nodes': [(entity, positions[entity], node_papers[entity]) 
                  for entity in top_entities],
        'edges': [((e1, e2), weight) for (e1, e2), weight in filtered_edges.items()],
        'positions': positions
    }


def print_section_header(title):
    print(f"\n{'='*20} {title} {'='*20}")

def run(name, sources):
    print_section_header(f"DATASET: {name}")
    entries = load_data(sources)
    print(f"Total Papers: {len(entries)}")
    
    # 1. Linguistic
    print("\n--- (a) Linguistic Features ---")
    print(generate_linguistic_feature_stats(entries))
    
    # 2. Papers Timeline
    print("\n--- (b) Papers by Period (1971-2025) ---")
    print(generate_papers_by_period_stats(entries))
    
    # 3. Countries
    print("\n--- Papers by Country (Top 15) ---")
    clabs, ccoords = generate_country_stats(entries)
    print(f"yticklabels={{{', '.join(clabs)}}}")
    # Format coords slightly nicer
    print("coordinates {")
    print("    " + " ".join(ccoords))
    print("};")
    
    # 3b. Universities
    print("\n--- Papers by University (Top 15) ---")
    ulabs, ucoords = generate_university_stats(entries)
    print(f"yticklabels={{{', '.join(ulabs)}}}")
    print("coordinates {")
    print("    " + " ".join(ucoords))
    print("};")

    # 4. Species
    print("\n--- Top 15 Species ---")
    slabs, splots = generate_species_stats(entries)
    print(f"symbolic y coords={{{', '.join(slabs)}}}")
    for cat, coords in splots.items():
        print(f"% {cat}")
        print("\\addplot ... coordinates {")
        print("    " + " ".join(coords))
        print("};")
        
    # 5. Categories Timeline
    print("\n--- Animal Categories over Time (1981-2025) ---")
    cplots = generate_category_timeline(entries)
    for cat, coords in cplots.items():
        print(f"% {cat}")
        print("\\addplot ... coordinates {")
        print("    " + coords)
        print("};")
    
    # 6. University Collaboration Network
    print("\n--- University Collaboration Network (Top 15) ---")
    uni_network = build_collaboration_network(entries, entity_type='university')
    print(f"% Nodes: {len(uni_network['nodes'])}, Edges: {len(uni_network['edges'])}")
    print("\n% Node positions (x,y) and sizes (papers):")
    for entity, pos, papers in uni_network['nodes']:
        print(f"% {entity}: ({pos['x']:.2f}, {pos['y']:.2f}), {papers} papers")
    
    print("\n% Collaboration edges (weight = number of joint papers):")
    for (e1, e2), weight in sorted(uni_network['edges'], key=lambda x: x[1], reverse=True):
        print(f"\\draw[line width={weight}pt] ({e1}) -- ({e2}); % {weight} collaborations")
    
    # 7. Country Collaboration Network
    print("\n--- Country Collaboration Network (Top 15) ---")
    country_network = build_collaboration_network(entries, entity_type='country')
    print(f"% Nodes: {len(country_network['nodes'])}, Edges: {len(country_network['edges'])}")
    print("\n% Node positions (x,y) and sizes (papers):")
    for entity, pos, papers in country_network['nodes']:
        print(f"% {entity}: ({pos['x']:.2f}, {pos['y']:.2f}), {papers} papers")
    
    print("\n% Collaboration edges (weight = number of joint papers):")
    for (e1, e2), weight in sorted(country_network['edges'], key=lambda x: x[1], reverse=True):
        print(f"\\draw[line width={weight*0.2}pt] ({e1}) -- ({e2}); % {weight} collaborations")

if __name__ == "__main__":
    run("HUMAN + SUBSET", ['human', 'subset'])
    print("\n\n\n")
    run("HUMAN + SUBSET + FULLSET", ['human', 'subset', 'fullset'])
