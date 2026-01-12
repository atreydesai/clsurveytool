# CL Survey Tool

A comprehensive Streamlit application for annotating, managing, and analyzing Computational Linguistics research papers.

## Features

- **BibTeX Import**: Bulk import papers directly from `.bib` files.
- **AI-Assisted Annotation**: Automatically extract linguistic features, species, and computational stages from analysis notes using OpenAI.
- **Workflow Management**:
  - **Pending**: Import and organize raw citations.
  - **Auto-Save**: All edits (titles, year, notes, classifications) are saved instantly.
  - **Commit**: Finalize entries to the permanent dataset.
- **Dynamic Data Model**:
  - Support for multiple affiliations (University, Country, Discipline) per paper.
  - Custom disciplines and specialized species lists.
- **Analytics Dashboard**: Visualize trends in publication years, geographic distribution, and research topics.
- **Data Persistence**: Uses local JSONL files for storage (`research_data.jsonl`), making it lightweight and easy to version control.

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
    - Copy `.env.example` to `.env`:

      ```bash
      cp .env.example .env
      ```

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

## Tech Stack

- **Streamlit**: UI Framework
- **Pandas**: Data manipulation
- **OpenAI**: AI analysis
- **Plotly**: Interactive charts
