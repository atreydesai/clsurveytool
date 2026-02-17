
import json
import os
import re
from collections import defaultdict, Counter
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
    # Extract countries from affiliations
    # Need normalization? User prompt implies specific countries: 
    # Hungary, Italy, Côte d'Ivoire...
    # We will trust the data but maybe map some variations if we see them.
    
    cnt = Counter()
    for e in entries:
        # Avoid double counting country per paper? 
        # "Papers by Country" -> usually means distinct papers involving that country.
        # If a paper has 2 authors from UK, counts as 1 for UK.
        # If a paper has 1 UK and 1 US, counts as 1 UK and 1 US.
        
        
        countries_in_paper = set()
        for aff in e.get('affiliations', []):
            c = aff.get('country', '').strip()
            if c:
                # Normalize country names
                c_lower = c.lower()
                
                # USA variations
                if c_lower in ['usa', 'united states', 'united states of america', 'the united states', 'the united states of america', 'u.s.a.', 'u.s.']:
                    c = "US"
                # UK variations
                elif c_lower in ['uk', 'united kingdom', 'england', 'scotland', 'wales', 'northern ireland', 'great britain', 'the united kingdom']:
                    c = "UK"
                # Common other normalizations if needed
                elif c_lower in ['china', 'prc', 'people\'s republic of china']:
                    c = "China"
                elif c_lower in ['russia', 'russian federation']:
                    c = "Russia"
                
                
                # Ensure US/UK are uppercase
                if c_lower == 'us': c = "US"
                if c_lower == 'uk': c = "UK"
                
                countries_in_paper.add(c)
        
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
        # Escape special latex chars? e.g. Côte d'Ivoire handles utf8 fine in modern latex usually
        # But let's check output.
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

if __name__ == "__main__":
    run("HUMAN + SUBSET", ['human', 'subset'])
    print("\n\n\n")
    run("HUMAN + SUBSET + FULLSET", ['human', 'subset', 'fullset'])
