"""
CL Survey Tool - Flask Backend
A simple web app for annotating computational linguistics research papers.
"""

import os
import json
import uuid
from pathlib import Path
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
    """Get all entries (pending + saved)."""
    pending = load_pending()
    saved = load_dataset()
    
    # Collect known universities and disciplines from all entries
    known_universities = set()
    known_disciplines = set(DISCIPLINES)  # Start with base disciplines
    for entry in pending + saved:
        for aff in entry.get('affiliations', []):
            if aff.get('university'):
                known_universities.add(aff['university'])
            if aff.get('discipline'):
                known_disciplines.add(aff['discipline'])
    
    return jsonify({
        'pending': pending,
        'saved': saved,
        'constants': {
            'species_categories': SPECIES_CATEGORIES,
            'computational_stages': COMPUTATIONAL_STAGES,
            'linguistic_features': LINGUISTIC_FEATURES,
            'disciplines': sorted(list(known_disciplines)),
            'countries': COUNTRIES,
            'known_universities': sorted(list(known_universities))
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
