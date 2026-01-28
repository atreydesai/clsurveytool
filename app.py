"""
CL Survey Tool - Flask Backend
A simple web app for annotating computational linguistics research papers.
"""

import os
import json
import uuid
import re
import io
import base64
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from itertools import combinations
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
import openai
import bibtexparser
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for server
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import networkx as nx
import pdfplumber

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PENDING_FILE = os.path.join(DATA_DIR, 'pending.json')
DATASET_FILE = os.path.join(DATA_DIR, 'dataset.jsonl')

# Multi-dataset file paths
HUMAN_DATASET_FILE = os.path.join(DATA_DIR, 'human_dataset.jsonl')
SUBSET_DATASET_FILE = os.path.join(DATA_DIR, 'subset_dataset.jsonl')
FULLSET_DATASET_FILE = os.path.join(DATA_DIR, 'fullset_dataset.jsonl')

# Source data directories
SOURCE_METADATA_DIR = os.path.join(DATA_DIR, 'source_metadata')
SOURCE_PDFS_DIR = os.path.join(DATA_DIR, 'source_pdfs')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Constants
SPECIES_CATEGORIES = [
    "Amphibian", "Terrestrial Mammal", "Marine Mammal", "Bird", 
    "Primate", "Reptile", "Fish", "Insect", "Other"
]

COMPUTATIONAL_STAGES = [
    "Data Collection", "Pre-processing", "Sequence Representation",
    "Meaning Identification", "Generation"
]

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

DISCIPLINES = ["Linguistics", "Computer Science", "Biology", "Other"]

COUNTRIES = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Argentina", "Armenia", "Australia",
    "Austria", "Azerbaijan", "Bahrain", "Bangladesh", "Belarus", "Belgium", "Bhutan", "Bolivia",
    "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Cambodia", "Cameroon",
    "Canada", "Chile", "China", "Colombia", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic",
    "Denmark", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Estonia", "Ethiopia",
    "Finland", "France", "Georgia", "Germany", "Ghana", "Greece", "Guatemala", "Honduras", "Hong Kong",
    "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica",
    "Japan", "Jordan", "Kazakhstan", "Kenya", "Kuwait", "Latvia", "Lebanon", "Lithuania", "Luxembourg",
    "Malaysia", "Maldives", "Malta", "Mexico", "Moldova", "Monaco", "Mongolia", "Morocco", "Myanmar",
    "Nepal", "Netherlands", "New Zealand", "Nigeria", "North Korea", "Norway", "Oman", "Pakistan",
    "Panama", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia",
    "Saudi Arabia", "Senegal", "Serbia", "Singapore", "Slovakia", "Slovenia", "South Africa",
    "South Korea", "Spain", "Sri Lanka", "Sudan", "Sweden", "Switzerland", "Syria", "Taiwan",
    "Tanzania", "Thailand", "Tunisia", "Turkey", "UAE", "Uganda", "UK", "Ukraine", "Uruguay", "United States",
    "Uzbekistan", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
]

AI_SYSTEM_PROMPT = """Analyze the following notes about a research paper based on what is explicitly stated in the text. This task supports a survey paper on computational animal linguistics. Try not to infer, extrapolate, or generalize beyond the information given. 

Extract only those elements that are supported by the notes: Specific animal species (use common names only). Include multiple species only if each is a primary focus of the study, not incidental or comparative. General species categories corresponding to the included species. Computational stages only if the paper describes methods or analyses that match the provided stage definitions. Linguistic features if the paper provides direct evidence, analysis, or claims addressing that feature. If only given the abstract, you can loosen this criteria slightly to prevent information loss.

Definition of the computational stages:
1. Data Collection: This stage involves the acquisition of high-quality, contextualized animal vocalizations and videos from sources such as wild recordings, laboratory experiments, and crowdsourced citizen science platforms. It serves as the foundation for the pipeline by building large, diverse datasets that capture the rich behavioral contexts necessary for analysis.
2. Pre-processing: Preprocessing applies computational techniques like denoising and sound source separation to isolate pure animal vocals from environmental background noise or overlapping signals. Additionally, it utilizes sound event detection to accurately identify and segment specific acoustic events within long-form recordings for further study.
3. Sequence Representation: In this stage, vocal segments are divided into smaller, fine-grained units and tokenized into discrete symbols or "phones" to create a structural repertoire. These representations allow researchers to model the syntactical patterns of animal communication and discover underlying phonetic alphabets.
4. Meaning Identification: This process employs machine learning classifiers and statistical correlations to associate specific vocal tokens with semantic meanings, such as emotional states, individual identities, or environmental contexts. By analyzing these relationships, researchers attempt to uncover the functional intent and communicative structure behind animal signals.
5. Generation: Generation uses statistical sequence models or deep generative neural networks to synthesize realistic, targeted animal vocalizations for use in controlled playback experiments. This final stage completes the computational loop, enabling researchers to test biological hypotheses and potentially move toward human-to-animal translation.

The 12 linguistic features are:
1. Vocal Auditory Channel and Turn-taking: The exchange of language, where communication will occur by the producer emitting sounds i.e. speech, and the receiving of these sounds by another animal completes the exchange. Turn-taking refers to the taking of turns when communicating.
2. Broadcast and Direct Reception: Vocalizations can be sent out in all directions but will be localized in space by the receiver.
3. Reference and Displacement: Reference is the relationship between an object and its associated vocalization/word. Displacement is an animal's ability to refer to objects that are remote in time and space.
4. Specialization: The idea that the meaning of a word is not dependent on how softly or loudly it is said.
5. Arbitrariness and Duality of Patterns: Words sound different from their inherent meaning, i.e., a long word doesn't need to represent a complex idea.
6. Discreteness and Syntax: Vocabulary is made of distinct units, and syntax is the way these units are strung together to form words and sentences.
7. Recursion: The structuring of language. It occurs when units of words are repeated within the words.
8. Semanticity: Words have meaning and are used to explain features of the world.
9. Prevarication: With access to language, animals also have access to lie and deceive one another.
10. Openness: The ability to generate new words/create new messages.
11. Tradition and Cultural Transmission: Tradition is the ability of animals to learn/teach their language, and cultural transmission is the passing down of language.
12. Learnability: Ability of an animal to learn another species' language or dialects outside of what an animal has been taught.

Species categories: Amphibian, Terrestrial Mammal, Marine Mammal, Bird, Primate, Reptile, Fish, Insect, Other

Return JSON only, with no explanatory text before or after.
All values must be chosen exactly and verbatim from the predefined lists in this prompt. Do not invent labels, placeholders (for example, "feature1"), abbreviations, or paraphrases. Do not lowercase, split, merge, or reword any item names. 

Formatting rules:
Use exact string matches for all list items.
Preserve capitalization and punctuation exactly as given.
Multi-word features must appear as one complete string and must not be split into multiple entries.
Though not every paper has computational stages, there should be linguistic features present.

Allowed values:

For "computational_stages", choose only from:
"Data Collection"
"Pre-processing"
"Sequence Representation"
"Meaning Identification"
"Generation"

For "linguistic_features", choose only from the following exact names:
"Vocal Auditory Channel and Turn-taking"
"Broadcast and Direct Reception"
"Reference and Displacement"
"Specialization"
"Arbitrariness and Duality of Patterns"
"Discreteness and Syntax"
"Recursion"
"Semanticity"
"Prevarication"
"Openness"
"Tradition and Cultural Transmission"
"Learnability"

For "species_categories", choose only from:
"Amphibian"
"Terrestrial Mammal"
"Marine Mammal"
"Bird"
"Primate"
"Reptile"
"Fish"
"Insect"
"Other"

For "specialized_species", use common names only, exactly as written in the notes.

Required JSON schema:

{
"specialized_species": [],
"species_categories": [],
"computational_stages": [],
"linguistic_features": []
}

"""


# ============================================================================
# DATA PERSISTENCE
# ============================================================================

def load_pending():
    """Load pending entries."""
    if os.path.exists(PENDING_FILE):
        try:
            with open(PENDING_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


def save_pending(entries):
    """Save pending entries."""
    with open(PENDING_FILE, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def load_dataset():
    """Load saved dataset entries."""
    entries = []
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return entries


def save_to_dataset(entry):
    """Append entry to dataset."""
    with open(DATASET_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def update_dataset_entry(entry_id, updated_entry):
    """Update a specific entry in the dataset."""
    entries = load_dataset()
    for i, e in enumerate(entries):
        if e.get('id') == entry_id:
            entries[i] = updated_entry
            break
    # Rewrite file
    with open(DATASET_FILE, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')


def delete_from_dataset(entry_id):
    """Delete entry from dataset."""
    entries = load_dataset()
    entries = [e for e in entries if e.get('id') != entry_id]
    with open(DATASET_FILE, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')


# ============================================================================
# MULTI-DATASET SUPPORT
# ============================================================================

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


def save_dataset_file(filepath, entries):
    """Save entries to a JSONL dataset file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def load_all_datasets(sources=None):
    """Load entries from specified dataset sources.
    
    Args:
        sources: List of source names ('human', 'subset', 'fullset').
                 If None, loads from all sources.
    
    Returns:
        List of all entries from specified sources.
    """
    if sources is None:
        sources = ['human', 'subset', 'fullset']
    
    all_entries = []
    
    source_files = {
        'human': HUMAN_DATASET_FILE,
        'subset': SUBSET_DATASET_FILE,
        'fullset': FULLSET_DATASET_FILE
    }
    
    for source in sources:
        if source in source_files:
            entries = load_dataset_file(source_files[source])
            all_entries.extend(entries)
    
    # Fallback: if no entries found in new files, try legacy dataset.jsonl
    if not all_entries and os.path.exists(DATASET_FILE):
        all_entries = load_dataset()
    
    return all_entries


def get_all_dois(filepath):
    """Get set of DOIs from a dataset file."""
    dois = set()
    entries = load_dataset_file(filepath)
    for entry in entries:
        doi = entry.get('doi', '').strip()
        if doi:
            # Normalize DOI format
            doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '')
            dois.add(doi.lower())
    return dois


# ============================================================================
# BIBTEX PARSING
# ============================================================================

def parse_bibtex(bibtex_str):
    """Parse BibTeX string into list of entries."""
    entries = []
    try:
        bib_database = bibtexparser.loads(bibtex_str)
        for entry in bib_database.entries:
            # Parse authors
            authors_raw = entry.get('author', '')
            if ' and ' in authors_raw:
                authors = [a.strip() for a in authors_raw.split(' and ')]
            else:
                authors = [authors_raw.strip()] if authors_raw else []
            
            parsed = {
                'id': str(uuid.uuid4()),
                'title': entry.get('title', '').strip('{}'),
                'authors': authors,
                'year': entry.get('year', ''),
                'journal': entry.get('journal', entry.get('booktitle', '')),
                'abstract': entry.get('abstract', ''),
                'doi': entry.get('doi', ''),
                'analysis_notes': entry.get('abstract', ''),
                'affiliations': [],
                'species_categories': [],
                'specialized_species': [],
                'computational_stages': [],
                'linguistic_features': [],
                'status': 'pending',
                'created_at': datetime.now().isoformat()
            }
            entries.append(parsed)
    except Exception as e:
        print(f"BibTeX parse error: {e}")
    return entries


# ============================================================================
# AI ANALYSIS
# ============================================================================

def analyze_with_ai(notes):
    """Run AI analysis on notes."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return {'error': 'No API key configured'}
    
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": notes}
            ],
            response_format={"type": "json_object"}
        )
        raw_response = response.choices[0].message.content
        result = json.loads(raw_response)
        
        # Log the raw AI response (keep last 10)
        log_file = Path('ai_logs.txt')
        log_entry = f"\n{'='*80}\nTimestamp: {datetime.now().isoformat()}\nNotes: {notes[:500]}...\nRaw Response:\n{raw_response}\n"
        
        try:
            existing = log_file.read_text() if log_file.exists() else ""
            # Split by separator and keep last 9, then add new entry
            entries = existing.split('='*80)
            entries = [e.strip() for e in entries if e.strip()]
            entries = entries[-9:]  # Keep last 9
            entries.append(log_entry.strip())
            log_file.write_text(('='*80 + '\n').join([''] + entries))
        except Exception as log_error:
            print(f"Log error: {log_error}")
        
        return result
    except Exception as e:
        return {'error': str(e)}


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/entries', methods=['GET'])
def get_entries():
    """Get all entries (pending + saved) filtered by sources."""
    # Parse sources parameter
    sources_param = request.args.get('sources', '')
    if sources_param == 'none':
        sources = []
    elif sources_param:
        sources = [s.strip() for s in sources_param.split(',') if s.strip()]
    else:
        sources = ['human', 'subset', 'fullset']  # Default to all
    
    pending = []
    saved = []
    
    # Load pending entries (from main pending.json, filtered by data_source)
    all_pending = load_pending()
    for entry in all_pending:
        source = entry.get('data_source', 'human')  # Default to human for legacy
        if source in sources:
            pending.append(entry)
    
    # Load saved entries from each source's dataset file
    source_files = {
        'human': HUMAN_DATASET_FILE,
        'subset': SUBSET_DATASET_FILE,
        'fullset': FULLSET_DATASET_FILE
    }
    for source in sources:
        if source in source_files:
            entries = load_dataset_file(source_files[source])
            saved.extend(entries)
    
    # Fallback: if no saved entries found but legacy dataset exists, load from it
    if not saved and os.path.exists(DATASET_FILE) and 'human' in sources:
        saved = load_dataset()
    
    # Collect known universities, countries, and disciplines from all entries
    known_universities = set()
    known_countries = set(COUNTRIES)  # Start with base countries
    known_disciplines = set(DISCIPLINES)  # Start with base disciplines
    university_country_map = {}  # Map university -> country for auto-fill
    discipline_counts = {}  # Track (university, country) -> {discipline: count}
    for entry in pending + saved:
        for aff in entry.get('affiliations', []):
            if aff.get('university'):
                known_universities.add(aff['university'])
                # Store university -> country mapping (last one wins if duplicates)
                if aff.get('country'):
                    university_country_map[aff['university']] = aff['country']
                    # Track discipline counts for each (university, country) pair
                    if aff.get('discipline'):
                        key = (aff['university'], aff['country'])
                        if key not in discipline_counts:
                            discipline_counts[key] = {}
                        disc = aff['discipline']
                        discipline_counts[key][disc] = discipline_counts[key].get(disc, 0) + 1
            if aff.get('country'):
                known_countries.add(aff['country'])
            if aff.get('discipline'):
                known_disciplines.add(aff['discipline'])
    
    # Build university_discipline_map with most common discipline for each (university, country) pair
    # Key format: "university|country" -> most common discipline
    university_discipline_map = {}
    for (uni, country), disc_dict in discipline_counts.items():
        if disc_dict:
            most_common = max(disc_dict, key=disc_dict.get)
            university_discipline_map[f"{uni}|{country}"] = most_common
    
    return jsonify({
        'pending': pending,
        'saved': saved,
        'constants': {
            'species_categories': SPECIES_CATEGORIES,
            'computational_stages': COMPUTATIONAL_STAGES,
            'linguistic_features': LINGUISTIC_FEATURES,
            'disciplines': sorted(list(known_disciplines)),
            'countries': sorted(list(known_countries)),
            'known_universities': sorted(list(known_universities)),
            'university_country_map': university_country_map,
            'university_discipline_map': university_discipline_map
        }
    })


@app.route('/api/import', methods=['POST'])
def import_bibtex():
    """Import BibTeX entries."""
    data = request.get_json()
    bibtex_str = data.get('bibtex', '')
    
    if not bibtex_str:
        return jsonify({'error': 'No BibTeX provided'}), 400
    
    entries = parse_bibtex(bibtex_str)
    if not entries:
        return jsonify({'error': 'No valid entries found'}), 400
    
    # Add to pending
    pending = load_pending()
    pending.extend(entries)
    save_pending(pending)
    
    return jsonify({'message': f'Imported {len(entries)} entries', 'entries': entries})


@app.route('/api/entries/<entry_id>', methods=['PUT'])
def update_entry(entry_id):
    """Update an entry (auto-save)."""
    data = request.get_json()
    
    # Check pending first
    pending = load_pending()
    for i, e in enumerate(pending):
        if e.get('id') == entry_id:
            # Merge updates
            pending[i].update(data)
            pending[i]['updated_at'] = datetime.now().isoformat()
            save_pending(pending)
            return jsonify({'message': 'Saved', 'entry': pending[i]})
    
    # Check saved entries
    saved = load_dataset()
    for e in saved:
        if e.get('id') == entry_id:
            e.update(data)
            e['updated_at'] = datetime.now().isoformat()
            update_dataset_entry(entry_id, e)
            return jsonify({'message': 'Saved', 'entry': e})
    
    return jsonify({'error': 'Entry not found'}), 404


@app.route('/api/entries/<entry_id>/commit', methods=['POST'])
def commit_entry(entry_id):
    """Move entry from pending to saved."""
    pending = load_pending()
    
    for i, e in enumerate(pending):
        if e.get('id') == entry_id:
            entry = pending.pop(i)
            entry['status'] = 'saved'
            entry['committed_at'] = datetime.now().isoformat()
            save_pending(pending)
            
            # Save to correct dataset file based on data_source
            data_source = entry.get('data_source', 'human')
            source_files = {
                'human': HUMAN_DATASET_FILE,
                'subset': SUBSET_DATASET_FILE,
                'fullset': FULLSET_DATASET_FILE
            }
            target_file = source_files.get(data_source, HUMAN_DATASET_FILE)
            
            # Append to the appropriate dataset file
            with open(target_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            return jsonify({'message': 'Committed', 'entry': entry})
    
    return jsonify({'error': 'Entry not found in pending'}), 404


@app.route('/api/entries/<entry_id>', methods=['DELETE'])
def delete_entry(entry_id):
    """Delete an entry."""
    # Check pending
    pending = load_pending()
    for i, e in enumerate(pending):
        if e.get('id') == entry_id:
            pending.pop(i)
            save_pending(pending)
            return jsonify({'message': 'Deleted'})
    
    # Check saved
    delete_from_dataset(entry_id)
    return jsonify({'message': 'Deleted'})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Run AI analysis."""
    data = request.get_json()
    notes = data.get('notes', '')
    
    if not notes:
        return jsonify({'error': 'No notes provided'}), 400
    
    result = analyze_with_ai(notes)
    return jsonify(result)


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get analytics data."""
    saved = load_dataset()
    
    if not saved:
        return jsonify({'empty': True})
    
    # Aggregate stats
    years = {}
    species = {}
    stages = {}
    features = {}
    
    for e in saved:
        # Year distribution
        year = e.get('year', 'Unknown')
        years[year] = years.get(year, 0) + 1
        
        # Species categories
        for cat in e.get('species_categories', []):
            species[cat] = species.get(cat, 0) + 1
        
        # Computational stages
        for stage in e.get('computational_stages', []):
            stages[stage] = stages.get(stage, 0) + 1
        
        # Linguistic features
        for feat in e.get('linguistic_features', []):
            features[feat] = features.get(feat, 0) + 1
    
    return jsonify({
        'total': len(saved),
        'years': years,
        'species': species,
        'stages': stages,
        'features': features
    })


@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get comprehensive analytics data for all visualizations."""
    # Parse sources parameter (comma-separated: human,subset,fullset)
    sources_param = request.args.get('sources', '')
    if sources_param == 'none':
        # Explicitly no sources selected - return empty
        return jsonify({'empty': True})
    elif sources_param:
        sources = [s.strip() for s in sources_param.split(',') if s.strip()]
    else:
        sources = None  # Load all sources
    
    saved = load_all_datasets(sources)
    
    if not saved:
        return jsonify({'empty': True})
    
    # ---------- Group 1: Longitudinal Analysis ----------
    # Papers per year
    papers_by_year = defaultdict(int)
    features_by_year = defaultdict(lambda: defaultdict(int))
    stages_by_year = defaultdict(lambda: defaultdict(int))
    all_text_by_year = defaultdict(list)
    
    for e in saved:
        year_str = str(e.get('year', '')).strip()
        try:
            year = int(year_str.split('-')[0]) if year_str else None
        except ValueError:
            year = None
        
        if year and 1900 <= year <= 2100:
            papers_by_year[year] += 1
            
            # Features by year
            for feat in e.get('linguistic_features', []):
                features_by_year[year][feat] += 1
            
            # Stages by year
            for stage in e.get('computational_stages', []):
                stages_by_year[year][stage] += 1
            
            # Collect text for keyword extraction
            notes = e.get('analysis_notes', '') or ''
            title = e.get('title', '') or ''
            all_text_by_year[year].append(notes + ' ' + title)
    
    # Extract top keywords by year
    stopwords = set(STOPWORDS)
    stopwords.update(['paper', 'study', 'research', 'using', 'used', 'use', 'based', 
                      'results', 'show', 'found', 'also', 'one', 'two', 'may', 'can',
                      'however', 'et', 'al', 'well', 'via', 'new', 'first'])
    
    def extract_keywords(texts):
        all_words = ' '.join(texts).lower()
        words = re.findall(r'\b[a-z]{4,}\b', all_words)
        return Counter(w for w in words if w not in stopwords)
    
    # Get overall top keywords
    all_texts = []
    for texts in all_text_by_year.values():
        all_texts.extend(texts)
    overall_keywords = extract_keywords(all_texts).most_common(10)
    top_keywords = [kw for kw, _ in overall_keywords[:5]]
    
    # Track top 5 keywords over years
    keywords_by_year = {}
    for year, texts in all_text_by_year.items():
        word_counts = extract_keywords(texts)
        keywords_by_year[year] = {kw: word_counts.get(kw, 0) for kw in top_keywords}
    
    # ---------- Group 2: Distribution Stats ----------
    # Feature counts
    feature_counts = defaultdict(int)
    stage_counts = defaultdict(int)
    species_cat_counts = defaultdict(int)
    specialized_species_counts = defaultdict(int)
    country_counts = defaultdict(int)
    discipline_counts = defaultdict(int)
    affiliation_counts = defaultdict(int)
    
    for e in saved:
        for feat in e.get('linguistic_features', []):
            feature_counts[feat] += 1
        for stage in e.get('computational_stages', []):
            stage_counts[stage] += 1
        for cat in e.get('species_categories', []):
            species_cat_counts[cat] += 1
        for sp in e.get('specialized_species', []):
            specialized_species_counts[sp.lower().strip()] += 1
        for aff in e.get('affiliations', []):
            if aff.get('country'):
                country_counts[aff['country']] += 1
            if aff.get('discipline'):
                discipline_counts[aff['discipline']] += 1
            if aff.get('university'):
                affiliation_counts[aff['university']] += 1
    
    # Top 10 specialized species
    top_species = sorted(specialized_species_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Top affiliations
    top_affiliations = sorted(affiliation_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    
    return jsonify({
        'total': len(saved),
        # Group 1: Longitudinal
        'papers_by_year': dict(papers_by_year),
        'features_by_year': {y: dict(f) for y, f in features_by_year.items()},
        'stages_by_year': {y: dict(s) for y, s in stages_by_year.items()},
        'keywords_by_year': keywords_by_year,
        'top_keywords': top_keywords,
        # Group 2: Distributions
        'feature_counts': dict(feature_counts),
        'stage_counts': dict(stage_counts),
        'species_category_counts': dict(species_cat_counts),
        'top_specialized_species': top_species,
        'country_counts': dict(country_counts),
        'discipline_counts': dict(discipline_counts),
        'top_affiliations': top_affiliations,
        # Constants for chart ordering
        'all_features': LINGUISTIC_FEATURES,
        'all_stages': COMPUTATIONAL_STAGES,
        'all_species_categories': SPECIES_CATEGORIES
    })


@app.route('/api/analytics/wordcloud/<era>', methods=['GET'])
def get_wordcloud(era):
    """Generate word cloud for pre-LLM (<=2020) or post-LLM (>2020) era."""
    # Parse sources parameter
    sources_param = request.args.get('sources', '')
    if sources_param == 'none':
        return jsonify({'error': 'No sources selected'}), 400
    sources = [s.strip() for s in sources_param.split(',') if s.strip()] if sources_param else None
    
    saved = load_all_datasets(sources)
    
    if not saved:
        return jsonify({'error': 'No data'}), 400
    
    # Collect text based on era
    texts = []
    for e in saved:
        year_str = str(e.get('year', '')).strip()
        try:
            year = int(year_str.split('-')[0]) if year_str else None
        except ValueError:
            year = None
        
        if year:
            include = (era == 'pre' and year <= 2020) or (era == 'post' and year > 2020)
            if include:
                notes = e.get('analysis_notes', '') or ''
                title = e.get('title', '') or ''
                texts.append(notes + ' ' + title)
    
    if not texts:
        return jsonify({'error': f'No papers found for {era}-LLM era'}), 400
    
    # Generate word cloud
    all_text = ' '.join(texts)
    
    # Clean text
    all_text = re.sub(r'[^a-zA-Z\s]', ' ', all_text.lower())
    
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update([
        'paper', 'study', 'research', 'analysis', 'using', 'used', 'use',
        'based', 'results', 'show', 'found', 'also', 'however', 'may',
        'one', 'two', 'three', 'et', 'al', 'can', 'well', 'via',
        'however', 'moreover', 'furthermore', 'therefore', 'thus',
        'abstract', 'introduction', 'conclusion', 'discussion',
        'method', 'methods', 'approach', 'approaches', 'data',
        'present', 'presented', 'propose', 'proposed', 'although',
        'either', 's', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
        'l', 'm', 'n', 'o', 'p', 'q', 'r', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        'set', 'know', 'first', 'new', 'different', 'many', 'will', 'able','doi','org','https','http','www'
    ])
    
    try:
        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=custom_stopwords,
            max_words=100,
            colormap='viridis'
        ).generate(all_text)
        
        # Convert to base64 image
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        plt.close(fig)
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return jsonify({
            'image': f'data:image/png;base64,{img_base64}',
            'paper_count': len(texts),
            'era': era
        })
    except Exception as ex:
        return jsonify({'error': str(ex)}), 500


@app.route('/api/analytics/network/<network_type>', methods=['GET'])
def get_network(network_type):
    """Generate network graph data for affiliations, countries, or disciplines."""
    # Parse sources parameter
    sources_param = request.args.get('sources', '')
    if sources_param == 'none':
        return jsonify({'error': 'No sources selected'}), 400
    sources = [s.strip() for s in sources_param.split(',') if s.strip()] if sources_param else None
    
    saved = load_all_datasets(sources)
    
    if not saved:
        return jsonify({'error': 'No data'}), 400
    
    # Discipline subfield mapping - merge subfields into parent disciplines
    DISCIPLINE_MAP = {
        'Ecology': 'Biology',
        'Zoology': 'Biology',
        'Neuroscience': 'Biology',
        'Bioacoustics': 'Biology',
        'Ethology': 'Biology',
        'Marine Biology': 'Biology',
        'Cognitive Science': 'Computer Science',
        'Machine Learning': 'Computer Science',
        'NLP': 'Computer Science',
        'Psychology': 'Other',
        'Anthropology': 'Other',
        'Philosophy': 'Other',
        'Music': 'Other'
    }
    
    def map_discipline(disc):
        """Map subfield disciplines to parent disciplines."""
        if not disc:
            return None
        return DISCIPLINE_MAP.get(disc, disc)
    
    # For affiliation network, first find top 25 universities by paper count
    top_universities = None
    if network_type == 'affiliation':
        uni_counts = {}
        for e in saved:
            for a in e.get('affiliations', []):
                uni = a.get('university')
                if uni:
                    uni_counts[uni] = uni_counts.get(uni, 0) + 1
        # Get top 25 universities
        top_universities = set(
            uni for uni, _ in sorted(uni_counts.items(), key=lambda x: x[1], reverse=True)[:25]
        )
    
    G = nx.Graph()
    
    for e in saved:
        affiliations = e.get('affiliations', [])
        if len(affiliations) < 2:
            continue
        
        # Determine what nodes to use based on network type
        if network_type == 'affiliation':
            nodes = [a.get('university') for a in affiliations 
                     if a.get('university') and a.get('university') in top_universities]
        elif network_type == 'country':
            nodes = list(set(a.get('country') for a in affiliations if a.get('country')))
        elif network_type == 'discipline':
            # Map disciplines to parent disciplines
            raw_disciplines = [a.get('discipline') for a in affiliations if a.get('discipline')]
            nodes = list(set(map_discipline(d) for d in raw_disciplines if map_discipline(d)))
        else:
            return jsonify({'error': 'Invalid network type'}), 400
        
        # Add nodes
        for node in nodes:
            if not G.has_node(node):
                G.add_node(node, weight=1)
            else:
                G.nodes[node]['weight'] = G.nodes[node].get('weight', 1) + 1
        
        # Add edges for all pairs (co-occurrence)
        if len(nodes) >= 2:
            for n1, n2 in combinations(set(nodes), 2):
                if G.has_edge(n1, n2):
                    G[n1][n2]['weight'] += 1
                else:
                    G.add_edge(n1, n2, weight=1)
    
    # Convert to JSON format for frontend visualization
    nodes_list = []
    for node in G.nodes():
        nodes_list.append({
            'id': node,
            'label': node,
            'size': G.nodes[node].get('weight', 1)
        })
    
    edges_list = []
    for u, v, data in G.edges(data=True):
        edges_list.append({
            'source': u,
            'target': v,
            'weight': data.get('weight', 1)
        })
    
    # Get centrality metrics for top nodes
    if G.number_of_nodes() > 0:
        degree_cent = nx.degree_centrality(G)
        top_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    else:
        top_nodes = []
    
    return jsonify({
        'nodes': nodes_list,
        'edges': edges_list,
        'node_count': G.number_of_nodes(),
        'edge_count': G.number_of_edges(),
        'top_nodes': top_nodes,
        'network_type': network_type
    })


# ============================================================================
# MULTI-DATASET IMPORT ENDPOINTS
# ============================================================================

@app.route('/api/sources/stats', methods=['GET'])
def get_source_stats():
    """Get paper counts per data source."""
    human_count = len(load_dataset_file(HUMAN_DATASET_FILE))
    subset_count = len(load_dataset_file(SUBSET_DATASET_FILE))
    fullset_count = len(load_dataset_file(FULLSET_DATASET_FILE))
    legacy_count = len(load_dataset()) if os.path.exists(DATASET_FILE) else 0
    
    return jsonify({
        'human': human_count,
        'subset': subset_count,
        'fullset': fullset_count,
        'legacy': legacy_count,
        'total': human_count + subset_count + fullset_count
    })


@app.route('/api/import/fullset', methods=['POST'])
def import_fullset():
    """Import fullset metadata from source file, deduplicating against subset."""
    source_file = os.path.join(SOURCE_METADATA_DIR, 'final_relevant_papers.jsonl')
    
    if not os.path.exists(source_file):
        return jsonify({'error': 'Source file not found', 'path': source_file}), 404
    
    # Get DOIs from subset to exclude duplicates
    subset_dois = get_all_dois(SUBSET_DATASET_FILE)
    
    # Also get DOIs from existing human dataset
    human_dois = get_all_dois(HUMAN_DATASET_FILE)
    exclude_dois = subset_dois | human_dois
    
    # Load source data (it's a JSON array, not JSONL)
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            source_entries = json.loads(content)
        else:
            # Handle JSONL format
            source_entries = [json.loads(line) for line in content.split('\n') if line.strip()]
    
    # Filter and add data_source field
    imported = []
    skipped = 0
    
    for entry in source_entries:
        doi = entry.get('doi', '').strip()
        if doi:
            normalized_doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '').lower()
            if normalized_doi in exclude_dois:
                skipped += 1
                continue
        
        # Add data_source field
        entry['data_source'] = 'fullset'
        imported.append(entry)
    
    # Save to fullset dataset file
    save_dataset_file(FULLSET_DATASET_FILE, imported)
    
    return jsonify({
        'message': f'Imported {len(imported)} entries, skipped {skipped} duplicates',
        'imported': len(imported),
        'skipped': skipped
    })


@app.route('/api/import/migrate-legacy', methods=['POST'])
def migrate_legacy():
    """Migrate legacy dataset.jsonl to human_dataset.jsonl."""
    if not os.path.exists(DATASET_FILE):
        return jsonify({'error': 'No legacy dataset found'}), 404
    
    entries = load_dataset()
    
    # Add data_source field to each entry
    for entry in entries:
        entry['data_source'] = 'human'
    
    # Save to human dataset file
    save_dataset_file(HUMAN_DATASET_FILE, entries)
    
    return jsonify({
        'message': f'Migrated {len(entries)} entries to human dataset',
        'count': len(entries)
    })


@app.route('/api/import/subset', methods=['POST'])
def import_subset():
    """Import subset PDF metadata as pending entries for processing."""
    download_metadata_file = os.path.join(SOURCE_METADATA_DIR, 'download_metadata.jsonl')
    fullset_file = os.path.join(SOURCE_METADATA_DIR, 'final_relevant_papers.jsonl')
    
    if not os.path.exists(download_metadata_file):
        return jsonify({'error': 'Download metadata file not found', 'path': download_metadata_file}), 404
    
    # Load fullset metadata to get paper details (title, authors, etc.)
    fullset_by_doi = {}
    if os.path.exists(fullset_file):
        with open(fullset_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                fullset_entries = json.loads(content)
            else:
                fullset_entries = [json.loads(line) for line in content.split('\n') if line.strip()]
        
        # Build DOI lookup map (normalize DOIs for matching)
        for entry in fullset_entries:
            doi = entry.get('doi', '').strip()
            if doi:
                # Normalize: remove URL prefix, lowercase
                normalized_doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '').lower()
                fullset_by_doi[normalized_doi] = entry
    
    # Load download metadata (JSONL format)
    download_entries = []
    with open(download_metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    download_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    # Get existing pending entries and filter out already imported subset entries
    pending = load_pending()
    existing_dois = set()
    for e in pending:
        if e.get('doi'):
            normalized = e['doi'].replace('https://doi.org/', '').replace('http://doi.org/', '').lower().strip()
            existing_dois.add(normalized)
    
    # Create pending entries for each PDF
    imported = []
    skipped = 0
    not_found = 0
    
    for entry in download_entries:
        doi = entry.get('doi', '').strip()
        # Normalize DOI for lookup and deduplication
        normalized_doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '').lower()
        
        if normalized_doi in existing_dois:
            skipped += 1
            continue
        
        # Look up paper details from fullset
        paper_info = fullset_by_doi.get(normalized_doi, {})
        if not paper_info:
            not_found += 1
        
        # Extract text from PDF
        pdf_filename = entry.get('filename', '')
        pdf_text = ''
        if pdf_filename:
            pdf_path = os.path.join(SOURCE_PDFS_DIR, pdf_filename)
            if os.path.exists(pdf_path):
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        pages_text = []
                        for page in pdf.pages[:20]:  # Limit to first 20 pages
                            text = page.extract_text()
                            if text:
                                pages_text.append(text)
                        pdf_text = '\n\n'.join(pages_text)
                        # Truncate if too long (keep first 50k chars)
                        if len(pdf_text) > 50000:
                            pdf_text = pdf_text[:50000] + '\n\n[TRUNCATED...]'
                except Exception as e:
                    pdf_text = f'[Error extracting PDF: {str(e)}]'
        
        # Create pending entry with data from fullset metadata
        # analysis_notes contains extracted PDF text for AI analysis
        pending_entry = {
            'id': str(uuid.uuid4()),
            'doi': doi,
            'title': paper_info.get('title', ''),
            'authors': paper_info.get('authors', []),
            'year': paper_info.get('year', ''),
            'journal': paper_info.get('journal', ''),
            'abstract': paper_info.get('abstract', ''),
            'pdf_filename': pdf_filename,
            'data_source': 'subset',
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'analysis_notes': pdf_text,  # PDF text for AI analysis
            'affiliations': [],
            'species_categories': [],
            'specialized_species': [],
            'computational_stages': [],
            'linguistic_features': []
        }
        imported.append(pending_entry)
        existing_dois.add(normalized_doi)
    
    # Add to pending
    pending.extend(imported)
    save_pending(pending)
    
    return jsonify({
        'message': f'Imported {len(imported)} subset entries as pending, skipped {skipped} duplicates, {not_found} without metadata',
        'imported': len(imported),
        'skipped': skipped,
        'not_found': not_found
    })


if __name__ == '__main__':
    app.run(debug=True, port=5001)

