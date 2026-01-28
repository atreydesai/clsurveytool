#!/usr/bin/env python3
"""
Batch import subset PDFs with progress output.
Processes PDFs in batches and saves after each batch.
"""

import os
import json
import uuid
from datetime import datetime
import pdfplumber

# Configuration
SOURCE_PDFS_DIR = 'data/source_pdfs'
SOURCE_METADATA_DIR = 'data/source_metadata'
PENDING_FILE = 'data/pending.json'
BATCH_SIZE = 25  # Process 25 PDFs at a time

def load_pending():
    if os.path.exists(PENDING_FILE):
        with open(PENDING_FILE, 'r') as f:
            return json.load(f)
    return []

def save_pending(entries):
    with open(PENDING_FILE, 'w') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

def extract_pdf_text(pdf_path, max_pages=20, max_chars=50000):
    """Extract text from PDF."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = []
            for page in pdf.pages[:max_pages]:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            pdf_text = '\n\n'.join(pages_text)
            if len(pdf_text) > max_chars:
                pdf_text = pdf_text[:max_chars] + '\n\n[TRUNCATED...]'
            return pdf_text
    except Exception as e:
        return f'[Error extracting PDF: {str(e)}]'

def main():
    print("Loading metadata...")
    
    # Load fullset metadata for paper details
    fullset_file = os.path.join(SOURCE_METADATA_DIR, 'final_relevant_papers.jsonl')
    fullset_by_doi = {}
    if os.path.exists(fullset_file):
        with open(fullset_file, 'r') as f:
            content = f.read().strip()
            if content.startswith('['):
                fullset_entries = json.loads(content)
            else:
                fullset_entries = [json.loads(line) for line in content.split('\n') if line.strip()]
        
        for entry in fullset_entries:
            doi = entry.get('doi', '').strip()
            if doi:
                normalized_doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '').lower()
                fullset_by_doi[normalized_doi] = entry
    print(f"  Loaded {len(fullset_by_doi)} entries from fullset metadata")
    
    # Load download metadata
    download_file = os.path.join(SOURCE_METADATA_DIR, 'download_metadata.jsonl')
    download_entries = []
    with open(download_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    download_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"  Found {len(download_entries)} PDFs to process")
    
    # Load existing pending and get already imported DOIs
    pending = load_pending()
    existing_dois = set()
    for e in pending:
        if e.get('doi'):
            normalized = e['doi'].replace('https://doi.org/', '').replace('http://doi.org/', '').lower().strip()
            existing_dois.add(normalized)
    print(f"  {len(existing_dois)} DOIs already imported")
    
    # Filter to only unprocessed entries
    to_process = []
    for entry in download_entries:
        doi = entry.get('doi', '').strip()
        normalized_doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '').lower()
        if normalized_doi not in existing_dois:
            to_process.append(entry)
    
    print(f"\nProcessing {len(to_process)} new PDFs in batches of {BATCH_SIZE}...")
    
    imported = 0
    not_found = 0
    errors = 0
    
    for i, entry in enumerate(to_process):
        doi = entry.get('doi', '').strip()
        normalized_doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '').lower()
        
        # Get paper info
        paper_info = fullset_by_doi.get(normalized_doi, {})
        if not paper_info:
            not_found += 1
        
        # Extract PDF text
        pdf_filename = entry.get('filename', '')
        pdf_text = ''
        if pdf_filename:
            pdf_path = os.path.join(SOURCE_PDFS_DIR, pdf_filename)
            if os.path.exists(pdf_path):
                pdf_text = extract_pdf_text(pdf_path)
                if pdf_text.startswith('[Error'):
                    errors += 1
            else:
                pdf_text = f'[PDF not found: {pdf_filename}]'
                errors += 1
        
        # Create entry
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
            'analysis_notes': pdf_text,
            'affiliations': [],
            'species_categories': [],
            'specialized_species': [],
            'computational_stages': [],
            'linguistic_features': []
        }
        pending.append(pending_entry)
        existing_dois.add(normalized_doi)
        imported += 1
        
        # Progress update every 10
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(to_process)}] Processed {imported} PDFs...")
        
        # Save every batch
        if (i + 1) % BATCH_SIZE == 0:
            save_pending(pending)
            print(f"  Saved batch ({i+1} entries)")
    
    # Final save
    save_pending(pending)
    
    print(f"\n✓ Complete!")
    print(f"  Imported: {imported}")
    print(f"  No metadata: {not_found}")
    print(f"  Errors: {errors}")
    print(f"  Total pending entries: {len(pending)}")

if __name__ == '__main__':
    main()
