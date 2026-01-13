"""
CL Survey Tool - Flask Backend
A simple web app for annotating computational linguistics research papers.
"""

import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
import openai
import bibtexparser

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PENDING_FILE = os.path.join(DATA_DIR, 'pending.json')
DATASET_FILE = os.path.join(DATA_DIR, 'dataset.jsonl')

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

DISCIPLINES = [
    "Biology", "Ecology", "Animal Behavior", "Neuroscience", "Bioacoustics",
    "Linguistics", "Computer Science", "AI/ML", "Psychology", "Cognitive Science",
    "Zoology", "Marine Biology", "Veterinary Science", "Anthropology", "Other"
]

AI_SYSTEM_PROMPT = """Analyze the following notes about a research paper. Extract:
1. Specific animal species (main ones, include all if multiple focuses)
2. General species category from: Amphibian, Terrestrial Mammal, Marine Mammal, Bird, Primate, Reptile, Fish, Insect, Other
3. Computational stages from: Data Collection, Pre-processing, Sequence Representation, Meaning Identification, Generation
4. Which of these 12 linguistic features are present:
   - Vocal Auditory Channel and Turn-taking
   - Broadcast and Direct Reception
   - Reference and Displacement
   - Specialization
   - Arbitrariness and Duality of Patterns
   - Discreteness and Syntax
   - Recursion
   - Semanticity
   - Prevarication
   - Openness
   - Tradition and Cultural Transmission
   - Learnability

Return JSON only:
{
    "specialized_species": ["species1", "species2"],
    "species_categories": ["category1"],
    "computational_stages": ["stage1", "stage2"],
    "linguistic_features": ["feature1", "feature2"]
}"""


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
        result = json.loads(response.choices[0].message.content)
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
    """Get all entries (pending + saved)."""
    pending = load_pending()
    saved = load_dataset()
    return jsonify({
        'pending': pending,
        'saved': saved,
        'constants': {
            'species_categories': SPECIES_CATEGORIES,
            'computational_stages': COMPUTATIONAL_STAGES,
            'linguistic_features': LINGUISTIC_FEATURES,
            'disciplines': DISCIPLINES
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
            save_to_dataset(entry)
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


if __name__ == '__main__':
    app.run(debug=True, port=5000)
