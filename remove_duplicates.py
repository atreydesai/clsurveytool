import json
import os

def normalize_title(title):
    return " ".join(title.lower().split())

def load_titles(filename, is_jsonl=False):
    titles = set()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            if is_jsonl:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if 'title' in entry:
                            titles.add(normalize_title(entry['title']))
            else:
                data = json.load(f)
                for entry in data:
                    if 'title' in entry:
                        titles.add(normalize_title(entry['title']))
    except FileNotFoundError:
        print(f"File not found: {filename}")
    return titles

def remove_duplicates(pending_file, human_file):
    human_titles = load_titles(human_file, is_jsonl=True)
    
    with open(pending_file, 'r', encoding='utf-8') as f:
        pending_data = json.load(f)
    
    original_count = len(pending_data)
    new_pending_data = []
    removed_titles = []

    for entry in pending_data:
        title = normalize_title(entry.get('title', ''))
        if title and title in human_titles:
            removed_titles.append(entry.get('title'))
        else:
            new_pending_data.append(entry)
    
    with open(pending_file, 'w', encoding='utf-8') as f:
        json.dump(new_pending_data, f, indent=2)
    
    print(f"Removed {len(removed_titles)} duplicates from {pending_file}")
    for t in removed_titles:
        print(f" - Removed: {t}")
    print(f"Remaining entries: {len(new_pending_data)}")

if __name__ == "__main__":
    remove_duplicates('data/pending.json', 'data/human_dataset.jsonl')
