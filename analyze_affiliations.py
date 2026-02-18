#!/usr/bin/env python3
"""
Analyze countries and affiliations from the dataset.
Generates lists organized by number of papers.
"""

import json
from collections import Counter
from pathlib import Path

# Paths
DATA_DIR = Path('data')
HUMAN_DATASET = DATA_DIR / 'human_dataset.jsonl'
SUBSET_DATASET = DATA_DIR / 'subset_dataset.jsonl'

def load_jsonl(filepath):
    """Load data from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def normalize_university(university):
    """Normalize university names to handle variations."""
    if not university:
        return ""
    
    u = university.strip()
    u_lower = u.lower()
    
    # Remove common prefixes/suffixes for comparison
    u_clean = u_lower.replace(',', '').replace('.', '').replace('-', ' ').replace('  ', ' ').strip()
    
    # St Andrews variations
    if 'st andrews' in u_clean or 'st. andrews' in u_clean or 'saint andrews' in u_clean:
        if 'scottish oceans' not in u_clean and 'sea mammal' not in u_clean:
            return "University of St Andrews"
    
    # Zurich variations
    if ('zurich' in u_clean or 'zürich' in u_clean) and 'eth' not in u_clean:
        if 'university' in u_clean:
            return "University of Zurich"
    
    # Cambridge
    if 'cambridge' in u_clean and 'university' in u_clean:
        return "University of Cambridge"
    
    # Oxford
    if 'oxford' in u_clean and 'university' in u_clean:
        return "University of Oxford"
    
    # Neuchâtel variations
    if 'neuchatel' in u_clean or 'neuchâtel' in u_clean:
        return "University of Neuchâtel"
    
    # UC system
    if 'university of california' in u_clean or 'uc ' in u_clean:
        if 'los angeles' in u_clean or 'ucla' in u_clean:
            return "University of California, Los Angeles"
        elif 'san diego' in u_clean or 'ucsd' in u_clean:
            return "University of California, San Diego"
        elif 'berkeley' in u_clean:
            return "University of California, Berkeley"
        elif 'davis' in u_clean:
            return "University of California, Davis"
        elif 'santa cruz' in u_clean:
            return "University of California, Santa Cruz"
        elif 'merced' in u_clean:
            return "University of California, Merced"
    
    # Lyon variations
    if 'lyon' in u_clean and 'university' in u_clean:
        return "University of Lyon"
    
    # Tokyo variations
    if 'tokyo' in u_clean and 'university' in u_clean and 'metropolitan' not in u_clean:
        return "University of Tokyo"
    
    # Konstanz
    if 'konstanz' in u_clean:
        return "University of Konstanz"
    
    # São Paulo
    if 'sao paulo' in u_clean or 'são paulo' in u_clean:
        return "University of São Paulo"
    
    # Rockefeller
    if 'rockefeller' in u_clean:
        return "Rockefeller University"
    
    # Turin/Torino
    if ('turin' in u_clean or 'torino' in u_clean) and 'university' in u_clean:
        return "University of Turin"
    
    # MIT
    if u_clean in ['mit', 'mit csail'] or 'massachusetts institute of technology' in u_clean:
        return "Massachusetts Institute of Technology"
    
    # Liège
    if 'liege' in u_clean or 'liège' in u_clean:
        return "University of Liège"
    
    # Ohio State
    if 'ohio state' in u_clean:
        return "Ohio State University"
    
    # Chicago
    if 'chicago' in u_clean and 'university' in u_clean and 'illinois' not in u_clean:
        return "University of Chicago"
    
    # Australian National University
    if 'australian national university' in u_clean or 'anu' in u_clean:
        return "Australian National University"
    
    # Freie Universität Berlin
    if ('freie' in u_clean or 'free university' in u_clean) and 'berlin' in u_clean:
        return "Freie Universität Berlin"
    
    # Göttingen variations
    if 'gottingen' in u_clean or 'göttingen' in u_clean or 'goettingen' in u_clean:
        if 'georg' in u_clean or 'august' in u_clean:
            return "Georg-August-Universität Göttingen"
        return "University of Göttingen"
    
    # Max Planck Institute of Animal Behavior
    if 'max planck' in u_clean and 'animal behav' in u_clean:
        return "Max Planck Institute of Animal Behavior"
    
    # Max Planck Institute for Evolutionary Anthropology
    if 'max planck' in u_clean and 'evolutionary anthropology' in u_clean:
        return "Max Planck Institute for Evolutionary Anthropology"
    
    # German Primate Center
    if 'german primate center' in u_clean or 'deutsches primatenzentrum' in u_clean:
        return "German Primate Center"
    
    # University of Texas variations
    if 'university of texas' in u_clean or 'ut austin' in u_clean:
        if 'austin' in u_clean:
            return "University of Texas at Austin"
        elif 'arlington' in u_clean:
            return "University of Texas at Arlington"
        elif 'san antonio' in u_clean:
            return "University of Texas at San Antonio"
        elif 'dallas' in u_clean:
            return "University of Texas at Dallas"
    
    # Humboldt Berlin
    if 'humboldt' in u_clean and 'berlin' in u_clean:
        return "Humboldt-Universität zu Berlin"
    
    # Hebrew University
    if 'hebrew university' in u_clean:
        return "Hebrew University of Jerusalem"
    
    # Paris-Sud
    if 'paris sud' in u_clean or 'paris-sud' in u_clean:
        return "Paris-Sud University"
    
    # Rennes
    if 'rennes' in u_clean and 'university' in u_clean:
        return "University of Rennes"
    
    # Würzburg
    if 'wurzburg' in u_clean or 'würzburg' in u_clean or 'wuerzburg' in u_clean:
        return "University of Würzburg"
    
    # Tübingen
    if 'tubingen' in u_clean or 'tübingen' in u_clean or 'tuebingen' in u_clean:
        return "University of Tübingen"
    
    # Czech University of Life Sciences
    if 'czech university of life sciences' in u_clean:
        return "Czech University of Life Sciences Prague"
    
    # Pennsylvania State
    if 'pennsylvania state' in u_clean or 'penn state' in u_clean:
        return "Pennsylvania State University"
    
    # Wisconsin-Madison
    if 'wisconsin' in u_clean and 'university' in u_clean:
        if 'madison' in u_clean or 'wisconsin–madison' in u_lower:
            return "University of Wisconsin-Madison"
        elif 'milwaukee' in u_clean:
            return "University of Wisconsin-Milwaukee"
        return "University of Wisconsin"
    
    # Aix-Marseille
    if 'aix' in u_clean and 'marseille' in u_clean:
        return "Aix-Marseille University"
    
    # Friedrich-Alexander Erlangen-Nürnberg
    if 'erlangen' in u_clean and ('friedrich' in u_clean or 'alexander' in u_clean or 'nuremberg' in u_clean or 'nurnberg' in u_clean or 'nürnberg' in u_clean):
        return "Friedrich-Alexander-Universität Erlangen-Nürnberg"
    
    # Utrecht
    if 'utrecht' in u_clean and 'university' in u_clean:
        return "Utrecht University"
    
    # Ludwig-Maximilians München
    if 'ludwig' in u_clean and 'maximilian' in u_clean and ('munich' in u_clean or 'munchen' in u_clean or 'münchen' in u_clean):
        return "Ludwig-Maximilians-Universität München"
    
    # Technical University of Munich
    if ('technische' in u_clean or 'technical' in u_clean) and ('munich' in u_clean or 'munchen' in u_clean or 'münchen' in u_clean):
        return "Technical University of Munich"
    
    # Sorbonne
    if 'sorbonne' in u_clean:
        return "Sorbonne Université"
    
    # Toulouse
    if 'toulouse' in u_clean and 'university' in u_clean:
        return "University of Toulouse"
    
    # Polytechnic/Technical University of Catalonia
    if ('polytechnic' in u_clean or 'technical' in u_clean or 'upc' == u_clean or 'politecnica' in u_clean) and 'catalonia' in u_clean:
        return "Polytechnic University of Catalonia"
    
    # CNRS
    if u_clean == 'cnrs' or 'centre national de la recherche' in u_clean or 'national centre for scientific research' in u_clean:
        return "CNRS"
    
    # Queensland
    if 'queensland' in u_clean and 'university' in u_clean and 'technology' not in u_clean:
        return "University of Queensland"
    
    # Goethe Frankfurt
    if 'goethe' in u_clean:
        return "Goethe University Frankfurt"
    
    # San Diego State
    if 'san diego state' in u_clean:
        return "San Diego State University"
    
    # Amsterdam
    if 'amsterdam' in u_clean and 'university' in u_clean and 'vrije' not in u_clean:
        return "University of Amsterdam"
    
    # Toulon
    if 'toulon' in u_clean:
        return "University of Toulon"
    
    # Auckland
    if 'auckland' in u_clean and 'university' in u_clean and 'technology' not in u_clean:
        return "University of Auckland"
    
    # Adam Mickiewicz
    if 'adam mickiewicz' in u_clean:
        return "Adam Mickiewicz University"
    
    # Illinois Urbana-Champaign
    if 'illinois' in u_clean and 'university' in u_clean:
        if 'urbana' in u_clean or 'champaign' in u_clean:
            return "University of Illinois Urbana-Champaign"
        elif 'chicago' in u_clean:
            return "University of Illinois Chicago"
        return "University of Illinois"
    
    # Lisbon
    if 'lisbon' in u_clean or 'lisboa' in u_clean:
        if 'nova' in u_clean:
            return "NOVA University of Lisbon"
        return "University of Lisbon"
    
    # Moscow State
    if 'moscow' in u_clean and ('lomonosov' in u_clean or 'state' in u_clean):
        return "Lomonosov Moscow State University"
    
    # Saarland
    if 'saarland' in u_clean or 'saarlandes' in u_clean:
        return "Saarland University"
    
    # Bar-Ilan
    if 'bar ilan' in u_clean or 'bar-ilan' in u_clean:
        return "Bar-Ilan University"
    
    # Buenos Aires
    if 'buenos aires' in u_clean:
        return "University of Buenos Aires"
    
    # Montpellier
    if 'montpellier' in u_clean:
        return "University of Montpellier"
    
    # Ghent
    if 'ghent' in u_clean:
        return "Ghent University"
    
    # Gdańsk
    if 'gdansk' in u_clean or 'gdańsk' in u_clean:
        return "University of Gdańsk"
    
    # Clermont Auvergne
    if 'clermont' in u_clean and 'auvergne' in u_clean:
        return "Université Clermont Auvergne"
    
    # Paris Cité
    if 'paris cite' in u_clean or 'paris cité' in u_clean:
        return "Université Paris Cité"
    
    # Zoological Research Museum Alexander Koenig
    if 'alexander koenig' in u_clean or 'koenig' in u_clean and 'museum' in u_clean:
        return "Zoological Research Museum Alexander Koenig"
    
    # Swiss Center for Scientific Research
    if 'swiss center' in u_clean or 'centre suisse' in u_clean:
        return "Swiss Center for Scientific Research"
    
    # Saint-Étienne
    if 'saint etienne' in u_clean or 'st etienne' in u_clean or 'saint-etienne' in u_clean:
        return "University of Saint-Étienne"
    
    # Melbourne
    if 'melbourne' in u_clean and 'university' in u_clean:
        return "University of Melbourne"
    
    # Osnabrück
    if 'osnabruck' in u_clean or 'osnabrück' in u_clean:
        return "University of Osnabrück"
    
    # Western Australia
    if 'western australia' in u_clean:
        return "University of Western Australia"
    
    # Sheffield
    if 'sheffield' in u_clean and 'university' in u_clean:
        return "University of Sheffield"
    
    # York (UK)
    if 'york' in u_clean and 'university' in u_clean and 'new york' not in u_clean:
        return "University of York"
    
    # Arizona
    if 'arizona' in u_clean and 'university' in u_clean and 'state' not in u_clean and 'northern' not in u_clean:
        return "University of Arizona"
    
    # North Carolina
    if 'north carolina' in u_clean and 'university' in u_clean and 'state' not in u_clean:
        return "University of North Carolina"
    
    # Brest
    if 'brest' in u_clean and 'university' in u_clean:
        return "University of Brest"
    
    # Northeast Fisheries Science Center
    if 'northeast fisheries' in u_clean:
        return "Northeast Fisheries Science Center"
    
    # Severtsov Institute
    if 'severtsov' in u_clean:
        return "A.N. Severtsov Institute of Ecology and Evolution"
    
    # Queen's Belfast
    if "queen" in u_clean and 'belfast' in u_clean:
        return "Queen's University Belfast"
    
    # Wageningen
    if 'wageningen' in u_clean:
        return "Wageningen University and Research"
    
    # JASCO
    if 'jasco' in u_clean:
        return "JASCO Applied Sciences"
    
    # Leibniz IZW
    if 'leibniz' in u_clean and ('zoo' in u_clean or 'wildlife' in u_clean or 'izw' in u_clean):
        return "Leibniz Institute for Zoo and Wildlife Research"
    
    # EPFL
    if 'epfl' in u_clean or ('polytechnique' in u_clean and 'lausanne' in u_clean):
        return "École Polytechnique Fédérale de Lausanne"
    
    # Google
    if 'google' in u_clean:
        return "Google"
    
    # Microsoft
    if 'microsoft' in u_clean:
        return "Microsoft"
    
    # Universitas Nasional
    if 'universitas nasional' in u_clean:
        return "Universitas Nasional"
    
    # Federal University of Rio Grande do Norte
    if 'rio grande do norte' in u_clean:
        return "Federal University of Rio Grande do Norte"
    
    # St. Mary's University
    if "st mary" in u_clean or "st. mary" in u_clean:
        return "St. Mary's University"
    
    # Leipzig
    if 'leipzig' in u_clean and 'university' in u_clean and 'applied' not in u_clean:
        return "Leipzig University"
    
    # Bonn
    if 'bonn' in u_clean and 'university' in u_clean:
        return "University of Bonn"
    
    # Tel Aviv
    if 'tel aviv' in u_clean or 'tel-aviv' in u_clean:
        return "Tel Aviv University"
    
    # Sapienza Rome
    if 'sapienza' in u_clean or ('la sapienza' in u_clean):
        return "Sapienza University of Rome"
    
    # Paris-Saclay
    if 'paris saclay' in u_clean or 'paris-saclay' in u_clean:
        return "Université Paris-Saclay"
    
    # Veterinary Medicine Hannover
    if 'veterinary' in u_clean and 'hannover' in u_clean:
        return "University of Veterinary Medicine Hannover"
    
    # RIKEN Brain Science
    if 'riken' in u_clean and ('brain' in u_clean or 'neuroscience' in u_clean):
        return "RIKEN Center for Brain Science"
    
    # Cornell Lab of Ornithology
    if 'cornell' in u_clean and 'ornithology' in u_clean:
        return "Cornell Lab of Ornithology"
    
    # École Normale Supérieure (keep Lyon separate from Paris)
    if 'ecole normale' in u_clean or 'école normale' in u_clean:
        if 'lyon' in u_clean:
            return "École Normale Supérieure de Lyon"
        return "École Normale Supérieure"
    
    # Return original if no normalization
    return u

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

def analyze_affiliations(papers):
    """Analyze countries and universities from papers."""
    country_counter = Counter()
    university_counter = Counter()
    
    for paper in papers:
        # Get affiliations
        affiliations = paper.get('affiliations', [])
        
        if not affiliations:
            continue
            
        # Track countries and universities for this paper
        paper_countries = set()
        paper_universities = set()
        
        for affiliation in affiliations:
            country = affiliation.get('country', '').strip()
            university = affiliation.get('university', '').strip()
            
            if country:
                # Normalize country name
                normalized_country = normalize_country(country)
                if normalized_country:
                    paper_countries.add(normalized_country)
            if university:
                # Normalize university name
                normalized_university = normalize_university(university)
                if normalized_university:
                    paper_universities.add(normalized_university)
        
        # Count each country/university once per paper
        for country in paper_countries:
            country_counter[country] += 1
        for university in paper_universities:
            university_counter[university] += 1
    
    return country_counter, university_counter

def main():
    print("="*80)
    print("AFFILIATION ANALYSIS")
    print("="*80)
    
    # Load datasets
    print("\nLoading datasets...")
    human_data = load_jsonl(HUMAN_DATASET)
    subset_data = load_jsonl(SUBSET_DATASET)
    
    # Combine datasets
    all_papers = human_data + subset_data
    print(f"Total papers: {len(all_papers)}")
    
    # Analyze
    print("\nAnalyzing affiliations...")
    country_counter, university_counter = analyze_affiliations(all_papers)
    
    # Print results
    print("\n" + "="*80)
    print("COUNTRIES BY NUMBER OF PAPERS")
    print("="*80)
    print(f"\nTotal unique countries: {len(country_counter)}\n")
    
    for i, (country, count) in enumerate(country_counter.most_common(), 1):
        print(f"{i:3d}. {country:40s} {count:4d} papers")
    
    print("\n" + "="*80)
    print("UNIVERSITIES/AFFILIATIONS BY NUMBER OF PAPERS")
    print("="*80)
    print(f"\nTotal unique affiliations: {len(university_counter)}\n")
    
    for i, (university, count) in enumerate(university_counter.most_common(), 1):
        print(f"{i:3d}. {university:60s} {count:4d} papers")
    
    # Save to file
    output_file = DATA_DIR / 'affiliation_analysis.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("COUNTRIES BY NUMBER OF PAPERS\n")
        f.write("="*80 + "\n")
        f.write(f"\nTotal unique countries: {len(country_counter)}\n\n")
        
        for i, (country, count) in enumerate(country_counter.most_common(), 1):
            f.write(f"{i:3d}. {country:40s} {count:4d} papers\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("UNIVERSITIES/AFFILIATIONS BY NUMBER OF PAPERS\n")
        f.write("="*80 + "\n")
        f.write(f"\nTotal unique affiliations: {len(university_counter)}\n\n")
        
        for i, (university, count) in enumerate(university_counter.most_common(), 1):
            f.write(f"{i:3d}. {university:60s} {count:4d} papers\n")
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_file}")
    print("="*80)

if __name__ == '__main__':
    main()
