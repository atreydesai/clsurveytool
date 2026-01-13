# CL Survey Tool

A web app for annotating computational linguistics research papers with AI-assisted classification.

## Features

- **BibTeX Import**: Bulk import papers from BibTeX files
- **AI Analysis**: OpenAI-powered extraction of species, computational stages, and linguistic features
- **Smart Autocomplete**: Dynamic autocomplete for universities, countries, and disciplines
- **Dual-Pane Selectors**: Click-to-select UI for multi-value fields
- **Auto-Save**: Changes are saved automatically as you type
- **Analytics Dashboard**: Visualize your dataset statistics

## Setup

1. Create a `.env` file:

   ```
   OPENAI_API_KEY=your-api-key-here
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   python app.py
   ```

4. Open <http://127.0.0.1:5000>

## Data Files

- `data/pending.json` - Entries being annotated
- `data/dataset.jsonl` - Committed entries (final dataset)
- `ai_logs.txt` - Last 10 AI analysis responses

## Tech Stack

- **Backend**: Flask + OpenAI API
- **Frontend**: Vanilla HTML/CSS/JS
- **Data**: JSON/JSONL files
