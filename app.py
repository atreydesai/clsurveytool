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


@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get comprehensive analytics data for all visualizations."""
    saved = load_dataset()
    
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
    saved = load_dataset()
    
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
        'set', 'know', 'first', 'new', 'different', 'many', 'will', 'able'
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
    saved = load_dataset()
    
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


if __name__ == '__main__':
    app.run(debug=True, port=5001)
