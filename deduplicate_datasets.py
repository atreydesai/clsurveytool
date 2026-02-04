#!/usr/bin/env python3
"""
Dataset Deduplication Script

This script:
1. Enriches human_dataset.jsonl with DOIs from human_dataset_doi.txt (BibTeX format)
2. For entries without DOIs, queries CrossRef API to find them by title
3. Reports entries that couldn't be auto-matched for manual review
4. Deduplicates: subset vs fullset (remove from fullset if in subset)
5. Deduplicates: human vs subset (remove from subset if in human)

Priority (most restrictive to least):
1. human_dataset.jsonl (never modified except to add DOIs)
2. subset_dataset.jsonl (remove duplicates found in human)
3. fullset_dataset.jsonl (remove duplicates found in human or subset)
"""

import json
import re
import os
import time
import requests
from pathlib import Path
from difflib import SequenceMatcher

DATA_DIR = Path(__file__).parent / 'data'

HUMAN_DATASET = DATA_DIR / 'human_dataset.jsonl'
SUBSET_DATASET = DATA_DIR / 'subset_dataset.jsonl'
FULLSET_DATASET = DATA_DIR / 'fullset_dataset.jsonl'
HUMAN_DOI_FILE = DATA_DIR / 'human_dataset_doi.txt'

# Backup directory
BACKUP_DIR = DATA_DIR / 'backups'


def load_jsonl(filepath):
    """Load entries from a JSONL file."""
    entries = []
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return entries


def save_jsonl(filepath, entries):
    """Save entries to a JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def normalize_doi(doi):
    """Normalize DOI for comparison."""
    if not doi:
        return None
    doi = doi.strip().lower()
    # Remove common prefixes
    doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi)
    doi = re.sub(r'^doi:\s*', '', doi)
    return doi


def normalize_title(title):
    """Normalize title for comparison."""
    if not title:
        return ""
    # Remove special characters, convert to lowercase
    title = re.sub(r'[^\w\s]', '', title.lower())
    # Remove extra whitespace
    title = ' '.join(title.split())
    return title


def title_similarity(title1, title2):
    """Calculate similarity between two titles."""
    t1 = normalize_title(title1)
    t2 = normalize_title(title2)
    return SequenceMatcher(None, t1, t2).ratio()


def parse_bibtex_dois(bibtex_file):
    """Parse DOIs from BibTeX file, mapping title -> DOI."""
    title_to_doi = {}
    
    with open(bibtex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into entries
    entries = re.split(r'@\w+\{', content)[1:]  # Skip first empty split
    
    for entry in entries:
        # Extract DOI
        doi_match = re.search(r'DOI\s*=\s*\{([^}]+)\}', entry, re.IGNORECASE)
        # Extract title
        title_match = re.search(r'title\s*=\s*\{([^}]+)\}', entry, re.IGNORECASE)
        
        if doi_match and title_match:
            doi = doi_match.group(1).strip()
            title = title_match.group(1).strip()
            # Clean up LaTeX formatting
            title = re.sub(r'\\["\'^`]?\{?([a-zA-Z])\}?', r'\1', title)
            title_to_doi[normalize_title(title)] = normalize_doi(doi)
    
    return title_to_doi


def query_crossref_for_doi(title, authors=None, year=None):
    """Query CrossRef API to find DOI for a paper by title."""
    base_url = "https://api.crossref.org/works"
    
    # Build query
    query = title
    if authors and len(authors) > 0:
        # Add first author's last name
        first_author = authors[0].split(',')[0].strip()
        query += f" {first_author}"
    
    params = {
        'query.bibliographic': query,
        'rows': 5,
        'mailto': 'research@example.com'  # Polite pool
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            items = data.get('message', {}).get('items', [])
            
            for item in items:
                item_title = ' '.join(item.get('title', []))
                similarity = title_similarity(title, item_title)
                
                # Check year match if available
                year_match = True
                if year:
                    item_year = str(item.get('published-print', {}).get('date-parts', [[None]])[0][0])
                    if item_year == 'None':
                        item_year = str(item.get('published-online', {}).get('date-parts', [[None]])[0][0])
                    year_match = str(year) == item_year
                
                if similarity > 0.85 and year_match:
                    return normalize_doi(item.get('DOI'))
        
        time.sleep(0.5)  # Rate limiting
    except Exception as e:
        print(f"  CrossRef error: {e}")
    
    return None


def enrich_human_dataset_with_dois():
    """
    Step 1: Enrich human_dataset.jsonl with DOIs from BibTeX file and CrossRef.
    Returns list of entries that couldn't be matched.
    """
    print("\n" + "="*60)
    print("STEP 1: Enriching human_dataset.jsonl with DOIs")
    print("="*60)
    
    # Load existing entries
    entries = load_jsonl(HUMAN_DATASET)
    print(f"Loaded {len(entries)} entries from human_dataset.jsonl")
    
    # Parse DOIs from BibTeX
    bibtex_dois = parse_bibtex_dois(HUMAN_DOI_FILE)
    print(f"Parsed {len(bibtex_dois)} DOIs from human_dataset_doi.txt")
    
    # Track statistics
    already_have_doi = 0
    matched_from_bibtex = 0
    matched_from_crossref = 0
    unmatched = []
    
    for i, entry in enumerate(entries):
        title = entry.get('title', '')
        existing_doi = normalize_doi(entry.get('doi', ''))
        
        if existing_doi:
            already_have_doi += 1
            continue
        
        # Try to match from BibTeX
        normalized_title = normalize_title(title)
        
        # Try exact match first
        if normalized_title in bibtex_dois:
            entry['doi'] = bibtex_dois[normalized_title]
            matched_from_bibtex += 1
            print(f"  [BibTeX] Matched: {title[:60]}...")
            continue
        
        # Try fuzzy match
        best_match = None
        best_score = 0
        for bibtex_title, doi in bibtex_dois.items():
            score = title_similarity(title, bibtex_title)
            if score > best_score and score > 0.85:
                best_score = score
                best_match = doi
        
        if best_match:
            entry['doi'] = best_match
            matched_from_bibtex += 1
            print(f"  [BibTeX fuzzy] Matched: {title[:60]}...")
            continue
        
        # Try CrossRef
        print(f"  [CrossRef] Querying for: {title[:60]}...")
        doi = query_crossref_for_doi(
            title, 
            entry.get('authors', []),
            entry.get('year')
        )
        
        if doi:
            entry['doi'] = doi
            matched_from_crossref += 1
            print(f"    -> Found: {doi}")
        else:
            unmatched.append({
                'id': entry.get('id'),
                'title': title,
                'year': entry.get('year'),
                'authors': entry.get('authors', [])[:2]  # First 2 authors
            })
            print(f"    -> NOT FOUND")
    
    # Save updated entries
    save_jsonl(HUMAN_DATASET, entries)
    
    # Print summary
    print("\nSummary:")
    print(f"  Already had DOI: {already_have_doi}")
    print(f"  Matched from BibTeX: {matched_from_bibtex}")
    print(f"  Matched from CrossRef: {matched_from_crossref}")
    print(f"  Unmatched (need manual review): {len(unmatched)}")
    
    return unmatched


def deduplicate_datasets():
    """
    Step 2 & 3: Deduplicate across datasets.
    - Remove from fullset if exists in subset
    - Remove from subset if exists in human
    """
    print("\n" + "="*60)
    print("STEP 2: Deduplicating datasets")
    print("="*60)
    
    # Load all datasets
    human = load_jsonl(HUMAN_DATASET)
    subset = load_jsonl(SUBSET_DATASET)
    fullset = load_jsonl(FULLSET_DATASET)
    
    print(f"Loaded: human={len(human)}, subset={len(subset)}, fullset={len(fullset)}")
    
    # Build DOI sets for deduplication
    def get_doi_set(entries):
        dois = set()
        for entry in entries:
            doi = normalize_doi(entry.get('doi', ''))
            if doi:
                dois.add(doi)
        return dois
    
    human_dois = get_doi_set(human)
    subset_dois = get_doi_set(subset)
    
    print(f"\nDOIs: human={len(human_dois)}, subset={len(subset_dois)}")
    
    # Step 2a: Remove from fullset if in subset
    print("\n--- Deduplicating fullset vs subset ---")
    original_fullset_count = len(fullset)
    fullset_filtered = []
    removed_from_fullset_subset = 0
    
    for entry in fullset:
        doi = normalize_doi(entry.get('doi', ''))
        if doi and doi in subset_dois:
            removed_from_fullset_subset += 1
        else:
            fullset_filtered.append(entry)
    
    fullset = fullset_filtered
    print(f"  Removed {removed_from_fullset_subset} entries from fullset (duplicates in subset)")
    
    # Step 2b: Remove from fullset if in human
    print("\n--- Deduplicating fullset vs human ---")
    fullset_filtered = []
    removed_from_fullset_human = 0
    
    for entry in fullset:
        doi = normalize_doi(entry.get('doi', ''))
        if doi and doi in human_dois:
            removed_from_fullset_human += 1
        else:
            fullset_filtered.append(entry)
    
    fullset = fullset_filtered
    print(f"  Removed {removed_from_fullset_human} entries from fullset (duplicates in human)")
    
    # Step 3: Remove from subset if in human
    print("\n--- Deduplicating subset vs human ---")
    original_subset_count = len(subset)
    subset_filtered = []
    removed_from_subset = 0
    
    for entry in subset:
        doi = normalize_doi(entry.get('doi', ''))
        if doi and doi in human_dois:
            removed_from_subset += 1
        else:
            subset_filtered.append(entry)
    
    subset = subset_filtered
    print(f"  Removed {removed_from_subset} entries from subset (duplicates in human)")
    
    # Also deduplicate within each dataset (internal duplicates)
    print("\n--- Removing internal duplicates ---")
    
    def remove_internal_duplicates(entries, name):
        seen_dois = set()
        unique = []
        duplicates = 0
        for entry in entries:
            doi = normalize_doi(entry.get('doi', ''))
            if doi:
                if doi in seen_dois:
                    duplicates += 1
                    continue
                seen_dois.add(doi)
            unique.append(entry)
        if duplicates > 0:
            print(f"  Removed {duplicates} internal duplicates from {name}")
        else:
            print(f"  No internal duplicates in {name}")
        return unique
    
    # Deduplicate all three datasets internally
    original_human_count = len(human)
    human = remove_internal_duplicates(human, 'human')
    fullset = remove_internal_duplicates(fullset, 'fullset')
    subset = remove_internal_duplicates(subset, 'subset')
    
    # Create backups before saving
    BACKUP_DIR.mkdir(exist_ok=True)
    import shutil
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if HUMAN_DATASET.exists():
        shutil.copy(HUMAN_DATASET, BACKUP_DIR / f'human_dataset_{timestamp}.jsonl')
    if SUBSET_DATASET.exists():
        shutil.copy(SUBSET_DATASET, BACKUP_DIR / f'subset_dataset_{timestamp}.jsonl')
    if FULLSET_DATASET.exists():
        shutil.copy(FULLSET_DATASET, BACKUP_DIR / f'fullset_dataset_{timestamp}.jsonl')
    
    # Save deduplicated datasets (including human if it had internal duplicates)
    save_jsonl(HUMAN_DATASET, human)
    save_jsonl(SUBSET_DATASET, subset)
    save_jsonl(FULLSET_DATASET, fullset)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"human_dataset.jsonl: {original_human_count} -> {len(human)} entries")
    print(f"subset_dataset.jsonl: {original_subset_count} -> {len(subset)} entries")
    print(f"fullset_dataset.jsonl: {original_fullset_count} -> {len(fullset)} entries")
    print(f"\nBackups saved to: {BACKUP_DIR}")
    
    # Verify no overlap
    print("\n--- Verification ---")
    subset_dois_new = get_doi_set(subset)
    fullset_dois_new = get_doi_set(fullset)
    
    overlap_human_subset = human_dois & subset_dois_new
    overlap_human_fullset = human_dois & fullset_dois_new
    overlap_subset_fullset = subset_dois_new & fullset_dois_new
    
    print(f"Overlap human-subset: {len(overlap_human_subset)}")
    print(f"Overlap human-fullset: {len(overlap_human_fullset)}")
    print(f"Overlap subset-fullset: {len(overlap_subset_fullset)}")
    
    if overlap_human_subset or overlap_human_fullset or overlap_subset_fullset:
        print("⚠️  WARNING: Some overlaps remain (entries without DOIs)")
    else:
        print("✅ No overlaps - all datasets are now disjoint!")


def main():
    print("="*60)
    print("DATASET DEDUPLICATION TOOL")
    print("="*60)
    
    # Step 1: Enrich human dataset with DOIs
    unmatched = enrich_human_dataset_with_dois()
    
    # Report unmatched entries
    if unmatched:
        print("\n" + "="*60)
        print("ENTRIES NEEDING MANUAL DOI LOOKUP")
        print("="*60)
        for i, entry in enumerate(unmatched, 1):
            print(f"\n{i}. {entry['title']}")
            print(f"   Year: {entry['year']}")
            print(f"   Authors: {', '.join(entry['authors'][:2])}")
            print(f"   ID: {entry['id']}")
    
    # Step 2-3: Deduplicate
    deduplicate_datasets()


if __name__ == '__main__':
    main()
