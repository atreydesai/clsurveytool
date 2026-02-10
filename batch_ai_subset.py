#!/usr/bin/env python3
"""
Batch process subset entries: Run AI analysis and commit to dataset.
Processes in batches of 8.

Usage:
  python batch_ai_subset.py           # Process all subset entries
  python batch_ai_subset.py --limit 5 # Process only first 5 entries (for testing)
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Constants
PENDING_FILE = 'data/pending.json'
SUBSET_DATASET_FILE = 'data/subset_dataset.jsonl'
BATCH_SIZE = 8

AI_SYSTEM_PROMPT = """Analyze the following notes about a research paper based on what is explicitly stated in the text. This task supports a survey paper on computational animal linguistics. Try not to infer, extrapolate, or generalize beyond the information given. 

Extract only those elements that are supported by the notes: Specific animal species (use common names only). Include multiple species only if each is a primary focus of the study, not incidental or comparative. General species categories corresponding to the included species. Computational stages only if the paper describes methods or analyses that match the provided stage definitions. Linguistic features if the paper provides direct evidence, analysis, or claims addressing that feature. If only given the abstract, you can loosen this criteria slightly to prevent information loss.

AFFILIATIONS EXTRACTION:
Extract author affiliations from the paper. For each UNIQUE combination of university and discipline, create one affiliation entry.
- "university": The English name of the university or research institution. If the original is in another language, translate to English. Use standard ASCII characters only (no special characters like ʻ or ā - convert to regular letters). Do not include footnote numbers or superscripts.
- "country": The country where that university is located. Use standard English country names.
- "discipline": ONLY include if the department/field is EXPLICITLY stated in the paper. Map to one of: "Linguistics", "Computer Science", "Biology", "Other". If a specific discipline like "Marine Biology", "Ecology", "Zoology", "Neuroscience" is mentioned, map to "Biology". If "Machine Learning", "AI", "Data Science" is mentioned, map to "Computer Science". If no discipline is explicitly stated, leave discipline as empty string "".

Rules for affiliations:
- Only include UNIQUE combinations of (university, discipline). If multiple authors are from the same university and department, include only one entry.
- If the same university has authors in different departments, include separate entries for each unique discipline.
- If an author has multiple affiliations, include all applicable as separate entries.
- Do NOT invent or guess affiliations - only extract what is explicitly written.
- Convert non-English institution names to their English equivalents (e.g., "Université de Lyon" -> "University of Lyon").

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

For "discipline" in affiliations, choose only from:
"Linguistics"
"Computer Science"
"Biology"
"Other"
""  (empty string if not explicitly stated)

Required JSON schema:

{
"specialized_species": [],
"species_categories": [],
"computational_stages": [],
"linguistic_features": [],
"affiliations": [
  {"university": "Example University", "country": "Country Name", "discipline": "Biology"}
]
}

"""

import unicodedata
import re

def normalize_university(name):
    """Normalize university name: convert to ASCII, remove footnotes."""
    if not name:
        return name
    
    # Normalize unicode characters to ASCII equivalents
    # e.g., ʻ -> ', ā -> a, é -> e
    normalized = unicodedata.normalize('NFKD', name)
    ascii_name = normalized.encode('ascii', 'ignore').decode('ascii')
    
    # Remove trailing footnote numbers (e.g., "Roma1" -> "Roma", "University2" -> "University")
    ascii_name = re.sub(r'(\d+)\s*$', '', ascii_name)
    
    # Remove superscript-like patterns
    ascii_name = re.sub(r'\s*\d+\s*$', '', ascii_name)
    
    # Clean up extra whitespace
    ascii_name = ' '.join(ascii_name.split())
    
    return ascii_name.strip()

def analyze_with_ai(notes, max_retries=3):
    """Run AI analysis on notes with rate limit retry and gpt-5-nano fallback."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return {'error': 'No API key configured'}
    
    client = openai.OpenAI(api_key=api_key)
    
    def parse_wait_time(error_str):
        """Parse wait time from rate limit error message."""
        match = re.search(r'Please try again in (\d+\.?\d*)s', error_str)
        if match:
            return float(match.group(1)) + 0.5
        return 5.0  # Default fallback
    
    def call_gpt5_nano():
        """Call gpt-5-nano with reasoning effort none."""
        response = client.responses.create(
            model="gpt-5-nano",
            input=f"{AI_SYSTEM_PROMPT}\n\n---\n\nAnalyze this paper:\n{notes}",
            reasoning={"effort": "none"}
        )
        return json.loads(response.output_text)
    
    def call_primary_model():
        """Call the primary model."""
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": notes}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    
    for attempt in range(max_retries):
        try:
            return call_primary_model()
        except Exception as e:
            error_str = str(e)
            # Check for rate limit error
            if '429' in error_str or 'rate_limit' in error_str.lower():
                wait_time = parse_wait_time(error_str)
                print(f"      Rate limit hit, trying gpt-5-nano while waiting {wait_time:.1f}s...")
                
                # Try gpt-5-nano as fallback
                try:
                    result = call_gpt5_nano()
                    return result
                except Exception as nano_error:
                    print(f"      gpt-5-nano also failed, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
            return {'error': error_str}
    
    return {'error': 'Max retries exceeded due to rate limits'}

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def append_jsonl(path, entry):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Strict Validation Lists
SPECIES_CATEGORIES_ALLOWED = {
    "Amphibian", "Terrestrial Mammal", "Marine Mammal", "Bird", 
    "Primate", "Reptile", "Fish", "Insect", "Other"
}

COMPUTATIONAL_STAGES_ALLOWED = {
    "Data Collection", "Pre-processing", "Sequence Representation",
    "Meaning Identification", "Generation"
}

LINGUISTIC_FEATURES_ALLOWED = {
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
}

DISCIPLINES_ALLOWED = {"Linguistics", "Computer Science", "Biology", "Other", ""}

def validate_ai_result(result):
    """Filter AI results to only include allowed values."""
    validated = {}
    
    # specialized_species: pass through (no strict validation list, just strings)
    validated['specialized_species'] = result.get('specialized_species', [])
    
    # species_categories
    raw_species = result.get('species_categories', [])
    validated['species_categories'] = [s for s in raw_species if s in SPECIES_CATEGORIES_ALLOWED]
    
    # computational_stages
    raw_stages = result.get('computational_stages', [])
    validated['computational_stages'] = [s for s in raw_stages if s in COMPUTATIONAL_STAGES_ALLOWED]
    
    # linguistic_features
    raw_features = result.get('linguistic_features', [])
    validated['linguistic_features'] = [f for f in raw_features if f in LINGUISTIC_FEATURES_ALLOWED]
    
    # affiliations: validate and deduplicate
    raw_affiliations = result.get('affiliations', [])
    seen_combos = set()
    validated_affiliations = []
    
    for aff in raw_affiliations:
        if not isinstance(aff, dict):
            continue
        
        university = aff.get('university', '').strip()
        country = aff.get('country', '').strip()
        discipline = aff.get('discipline', '').strip()
        
        # Normalize university name (remove special chars, footnotes)
        university = normalize_university(university)
        
        # Skip if no university
        if not university:
            continue
        
        # Validate discipline
        if discipline not in DISCIPLINES_ALLOWED:
            discipline = ""  # Clear invalid disciplines
        
        # Check for unique combination
        combo = (university.lower(), discipline.lower())
        if combo in seen_combos:
            continue
        seen_combos.add(combo)
        
        validated_affiliations.append({
            'university': university,
            'country': country,
            'discipline': discipline
        })
    
    validated['affiliations'] = validated_affiliations
    
    return validated

def main():
    # Check for --limit argument
    limit = None
    if '--limit' in sys.argv:
        try:
            idx = sys.argv.index('--limit')
            limit = int(sys.argv[idx + 1])
            print(f"Running in TEST MODE: processing only {limit} entries")
        except (IndexError, ValueError):
            print("Error: --limit requires a number")
            return
    
    # Check for --workers argument (default 20)
    workers = 20
    if '--workers' in sys.argv:
        try:
            idx = sys.argv.index('--workers')
            workers = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            pass
    
    print(f"Using {workers} concurrent workers")
    print("Loading pending entries...")
    pending = load_json(PENDING_FILE)
    
    # Filter for subset entries
    subset_entries = [e for e in pending if e.get('data_source') == 'subset']
    
    # Apply limit if specified
    if limit:
        subset_entries = subset_entries[:limit]
    
    print(f"Found {len(subset_entries)} subset entries to process.")
    
    if not subset_entries:
        print("No subset entries found.")
        return

    # Import here to avoid issues if not needed
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    # Thread-safe counter and lock for file operations
    file_lock = threading.Lock()
    
    def process_single_entry(entry):
        """Process a single entry - called in parallel."""
        notes = entry.get('analysis_notes', '')
        if not notes:
            return None, entry, "no_notes"
        
        # Run AI Analysis
        result = analyze_with_ai(notes)
        
        if 'error' in result:
            return None, entry, result['error']
        
        # Validate and Filter Result
        validated_result = validate_ai_result(result)
        
        # Update entry with validated AI results
        entry['specialized_species'] = validated_result['specialized_species']
        entry['species_categories'] = validated_result['species_categories']
        entry['computational_stages'] = validated_result['computational_stages']
        entry['linguistic_features'] = validated_result['linguistic_features']
        entry['affiliations'] = validated_result['affiliations']
        
        # Set status to saved
        entry['status'] = 'saved'
        entry['committed_at'] = datetime.now().isoformat()
        
        return entry, None, None
    
    # Process in batches with parallel API calls
    total = len(subset_entries)
    processed_count = 0
    errors = 0
    batch_num = 0
    
    while subset_entries:
        batch = subset_entries[:BATCH_SIZE]
        subset_entries = subset_entries[BATCH_SIZE:]
        batch_num += 1
        
        print(f"\nBatch {batch_num}: Processing {len(batch)} entries in parallel...")
        
        batch_processed = []
        batch_ids = set()
        
        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_entry = {executor.submit(process_single_entry, entry): entry for entry in batch}
            
            for future in as_completed(future_to_entry):
                original_entry = future_to_entry[future]
                try:
                    processed_entry, failed_entry, error = future.result()
                    
                    if processed_entry:
                        title = processed_entry.get('title', 'Untitled')[:40]
                        aff_count = len(processed_entry.get('affiliations', []))
                        print(f"  ✓ {title}... ({aff_count} affiliations)")
                        
                        # Thread-safe file append
                        with file_lock:
                            append_jsonl(SUBSET_DATASET_FILE, processed_entry)
                        
                        batch_processed.append(processed_entry)
                        batch_ids.add(processed_entry['id'])
                    else:
                        title = original_entry.get('title', 'Untitled')[:40]
                        print(f"  ✗ {title}... Error: {error}")
                        errors += 1
                        
                except Exception as e:
                    title = original_entry.get('title', 'Untitled')[:40]
                    print(f"  ✗ {title}... Exception: {e}")
                    errors += 1
        
        # Remove processed entries from pending (thread-safe)
        with file_lock:
            pending = [e for e in pending if e['id'] not in batch_ids]
            save_json(PENDING_FILE, pending)
        
        processed_count += len(batch_processed)
        print(f"  Saved {len(batch_processed)} entries. Progress: {processed_count}/{total}")
        
        # Small delay between batches to not overwhelm
        time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"Complete! Processed {processed_count} entries. Errors: {errors}")

if __name__ == '__main__':
    main()
