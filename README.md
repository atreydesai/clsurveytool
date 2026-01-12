# CL Survey Tool

A Streamlit application for annotating, managing, and analyzing our computational animal linguistics research papers.

## Features

- **BibTeX Import**: Bulk import papers directly from `.bib` files.
- **AI-Assisted Annotation**: Automatically extract linguistic features, species, and computational stages from analysis notes using OpenAI (fill in API key from .env).
- **Analytics Dashboard**: Visualize trends in publication years, geographic distribution, and research topics.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/clsurveytool.git
    cd clsurveytool
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Configure Environment**:
    - Add your OpenAI API Key to `.env`:

      ```
      OPENAI_API_KEY=sk-...
      ```

## Usage

1. **Run the App**:

    ```bash
    streamlit run app.py
    ```

2. **Import Data**: Use the "BibTeX Import" expander to paste and parse citations.
3. **Annotate**: Click "Edit" on any pending entry. Add notes, run AI analysis, and refine metadata.
4. **Commit**: Save finalized entries to the dataset.
5. **Analyze**: Switch to the **Analytics** page to view charts and statistics.

## Data Structure

- `pending_entries.json`: Stores draft/in-progress annotations.
- `research_data.jsonl`: The permanent dataset of committed entries.
