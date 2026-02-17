import json

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
    except json.JSONDecodeError:
        print(f"Error decoding JSON in {filename}")
    return titles

pending_titles = load_titles('data/pending.json', is_jsonl=False)
human_titles = load_titles('data/human_dataset.jsonl', is_jsonl=True)

duplicates = pending_titles.intersection(human_titles)

print(f"Found {len(duplicates)} duplicate titles between pending.json and human_dataset.jsonl")
if duplicates:
    print("Duplicate Titles:")
    for title in list(duplicates)[:10]: # Print first 10
        print(f"- {title}")
