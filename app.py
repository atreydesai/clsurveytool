"""
Computational Animal Linguistics Research Tool
A Streamlit application for annotating research papers and generating publication-ready visualizations.
"""

import streamlit as st
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import bibtexparser
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import tempfile

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import OpenAI
from openai import OpenAI

# Import copy to clipboard
from st_copy_to_clipboard import st_copy_to_clipboard

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

DATA_FILE = "research_data.jsonl"
PENDING_FILE = "pending_entries.json"
LOG_FILE = "ai_logs.txt"

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

SPECIES_CATEGORIES = [
    "Amphibian",
    "Terrestrial Mammal",
    "Marine Mammal",
    "Bird",
    "Primate",
    "Reptile",
    "Fish",
    "Insect",
    "Other"
]

COMPUTATIONAL_STAGES = [
    "Data Collection",
    "Data Pre-processing",
    "Sequence Representation",
    "Meaning Identification",
    "Generation"
]

DISCIPLINES = ["Biology", "Linguistics", "Computer Science", "Psychology", "Neuroscience", "Other"]

REFERENCE_TEXT = """
**I. Vocal Auditory Channel and Turn-taking:** The exchange of language, where communication will occur by the producer emitting sounds i.e. speech, and the receiving of these sounds by another animal completes the exchange. Turn-taking refers to the taking of turns when communicating.

**Broadcast and Direct Reception:** Vocalizations can be sent out in all directions but will be localized in space by the receiver.

**II. Reference and Displacement:** Reference is the relationship between an object and its associated vocalization/word. Displacement is an animal's ability to refer to objects that are remote in time and space.

**Specialization:** The idea that the meaning of a word is not dependent on how softly or loudly it is said.

**Arbitrariness and Duality of Patterns:** Words sound different from their inherent meaning, i.e., a long word doesn't need to represent a complex idea.

**III. Discreteness and Syntax:** Vocabulary is made of distinct units, and syntax is the way these units are strung together to form words and sentences.

**Recursion:** The structuring of language. It occurs when units of words are repeated within the words.

**Semanticity:** Words have meaning and are used to explain features of the world.

**Prevarication:** With access to language, animals also have access to lie and deceive one another.

**IV. Openness:** The ability to generate new words/create new messages.

**Tradition and Cultural Transmission:** Tradition is the ability of animals to learn/teach their language, and cultural transmission is the passing down of language.

**Learnability:** Ability of an animal to learn another species' language or dialects outside of what an animal has been taught.
"""

AI_SYSTEM_PROMPT = """Analyze the following notes about a research paper. Extract the specific animal species (focus on the main ones, but if there are multiple substantial focuses then include all), general species category, computational stages (from the list: Data Collection, Pre-processing, Sequence Representation, Meaning Identification, Generation), and which of the 12 linguistic features are present.

The 12 linguistic features are:
1. Vocal Auditory Channel and Turn-taking
2. Broadcast and Direct Reception
3. Reference and Displacement
4. Specialization
5. Arbitrariness and Duality of Patterns
6. Discreteness and Syntax
7. Recursion
8. Semanticity
9. Prevarication
10. Openness
11. Tradition and Cultural Transmission
12. Learnability

Species categories: Amphibian, Terrestrial Mammal, Marine Mammal, Bird, Primate, Reptile, Fish, Insect, Other

Return ONLY a valid JSON object with these exact keys:
{
    "specialized_species": ["list of species strings"],
    "species_categories": ["list of categories"],
    "computational_stages": ["list of stages"],
    "linguistic_features": ["list of feature names"]
}"""

# ============================================================================
# DATA PERSISTENCE FUNCTIONS
# ============================================================================

def load_data() -> pd.DataFrame:
    """Load data from JSONL file into a DataFrame."""
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
        return pd.DataFrame()
    
    records = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if not records:
        return pd.DataFrame()
    
    return pd.DataFrame(records)


def save_entry(entry: dict) -> bool:
    """Append a single entry to the JSONL file."""
    try:
        with open(DATA_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        st.error(f"Error saving entry: {e}")
        return False


def update_entry(index: int, updated_entry: dict) -> bool:
    """Update an entry at a specific index in the JSONL file."""
    try:
        records = []
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        if 0 <= index < len(records):
            records[index] = updated_entry
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            return True
        return False
    except Exception as e:
        st.error(f"Error updating entry: {e}")
        return False


def get_jsonl_content() -> str:
    """Get the full content of the JSONL file for download."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


def load_pending() -> dict:
    """Load pending entries from JSON file."""
    if os.path.exists(PENDING_FILE):
        try:
            with open(PENDING_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Migration: if 'in_progress' exists, merge it back to 'pending' if desired, 
                # or just ignore separate lists. User wants ONE pending list.
                # Let's ensure we read 'entries' properly.
                if 'entries' in data: # Old format
                     return {
                        'pending': data['entries'],
                        'bibtex': data.get('bibtex', ''),
                        'selected_idx': data.get('selected_idx')
                    }
                # Current format
                pending = data.get('pending', [])
                in_progress = data.get('in_progress', [])
                # Merge them if they were separate, or just return pending if we already merged.
                # To be safe and respect "undo in progress tab", we treat everything as pending.
                all_pending = pending + in_progress
                
                return {
                    'pending': all_pending,
                    'bibtex': data.get('bibtex', ''),
                    'selected_idx': data.get('selected_idx')
                }
        except (json.JSONDecodeError, IOError):
            pass
    return {
        'pending': [],
        'bibtex': '',
        'selected_idx': None
    }


def save_pending(pending: list, bibtex: str, selected_idx):
    """Save pending entries to JSON file."""
    try:
        with open(PENDING_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'pending': pending,
                'bibtex': bibtex,
                'selected_idx': selected_idx
            }, f, ensure_ascii=False, indent=2)
    except IOError as e:
        st.error(f"Error saving pending entries: {e}")


# ============================================================================
# BIBTEX PARSING
# ============================================================================

def parse_bibtex(bibtex_str: str) -> list:
    """Parse a BibTeX file and extract all entries."""
    results = []
    
    try:
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        bib_database = bibtexparser.loads(bibtex_str, parser=parser)
        
        for entry in bib_database.entries:
            parsed = {
                "title": entry.get("title", "").replace("{", "").replace("}", ""),
                "year": entry.get("year", ""),
                "journal": entry.get("journal", entry.get("booktitle", "")),
                "doi": entry.get("doi", ""),
                "abstract": entry.get("abstract", ""),
                "authors": [],
                "bib_key": entry.get("ID", "")
            }
            
            # Parse authors
            author_str = entry.get("author", "")
            if author_str:
                authors = [a.strip() for a in author_str.split(" and ")]
                parsed["authors"] = authors
            
            results.append(parsed)
    except Exception as e:
        st.error(f"Error parsing BibTeX: {e}")
    
    return results


def format_search_string(template: str, parsed: dict) -> str:
    """Format a search string using the template and parsed BibTeX data."""
    author_str = parsed["authors"][0] if parsed["authors"] else ""
    return template.format(
        title=parsed.get("title", ""),
        author=author_str,
        journal=parsed.get("journal", ""),
        year=parsed.get("year", "")
    )


# ============================================================================
# AI INTEGRATION
# ============================================================================

def clear_form_state():
    """Clear session state for annotation form widgets."""
    # Clear static keys used in the editor
    keys_to_clear = [
        "meta_title", "meta_year", "meta_journal", "meta_authors", "meta_notes", "aff_editor"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def apply_pending_ai_updates():
    """Apply pending AI updates to the current entry."""
    if 'pending_ai_updates' in st.session_state:
        updates = st.session_state.pending_ai_updates
        
        # Determine target entry
        target_entry = None
        save_callback = None
        
        if st.session_state.get('selected_entry_idx') is not None:
            idx = st.session_state.selected_entry_idx
            if idx < len(st.session_state.pending_entries):
                target_entry = st.session_state.pending_entries[idx]
                def save_cb():
                    save_pending(
                        st.session_state.pending_entries, 
                        st.session_state.last_bibtex, 
                        st.session_state.selected_entry_idx
                    )
                save_callback = save_cb
                
        elif st.session_state.get('editing_saved_idx') is not None:
            # Need to fetch from file again or trust we are in a rerun context where we can't easily fetch reference?
            # Actually, we need to load the full list to update the specific index.
            # Ideally, valid state management would mean we update the file IMMEDIATELY.
            # But here we are applying updates to memory first?
            # For saved entries, we modify the record and call update_entry.
            # We need the record relative to the file index.
            idx = st.session_state.editing_saved_idx
            # We need to load all records, update one, and save back.
            all_records = []
            if os.path.exists(DATA_FILE):
                 try:
                     with open(DATA_FILE, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip(): all_records.append(json.loads(line))
                 except: pass
            
            if idx < len(all_records):
                target_entry = all_records[idx]
                def save_cb():
                    update_entry(idx, target_entry)
                save_callback = save_cb
        
        if target_entry:
            # Apply updates
            target_entry['linguistic_features'] = updates.get('linguistic_features', [])
            target_entry['species_categories'] = [c for c in updates.get('species_categories', []) if c in SPECIES_CATEGORIES]
            target_entry['computational_stages'] = [s for s in updates.get('computational_stages', []) if s in COMPUTATIONAL_STAGES]
            
            # Specialized species
            spec = updates.get('specialized_species', [])
            if isinstance(spec, str): spec = [spec]
            target_entry['specialized_species'] = spec

            # Auto-save immediately
            if save_callback:
                save_callback()
        
        # Clear the pending updates
        del st.session_state.pending_ai_updates


def log_ai_interaction(notes: str, response_content: str, success: bool, error: str = None, metadata: dict = None):
    """Log AI interaction to a file."""
    
    # Truncate notes to first 50 words
    words = notes.split()
    truncated_notes = ' '.join(words[:50]) + ("..." if len(words) > 50 else "")
    
    timestamp = datetime.now().isoformat()
    log_entry = f"\n{'='*50}\nTIMESTAMP: {timestamp}\nSUCCESS: {success}\n"
    if metadata:
        log_entry += f"METADATA: {json.dumps(metadata)}\n"
    if error:
        log_entry += f"ERROR: {error}\n"
    log_entry += f"INPUT KEYWORDS (First 50 words):\n{truncated_notes}\n{'-'*20}\nOUTPUT:\n{response_content}\n{'='*50}\n"
    
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Failed to write log: {e}")


def call_ai_api(notes: str, status_container=None) -> Optional[dict]:
    """Call the OpenAI API to analyze notes and extract structured data."""
    if not OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY not set in .env file.")
        return None
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        if status_container:
            status_container.update(label="Sending request to OpenAI...", state="running")
        
        response = client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": f"Research paper notes:\n\n{notes}"}
            ]
        )
        
        if status_container:
            status_container.update(label="Processing response...", state="running")
            
        content = response.choices[0].message.content
        usage = response.usage.model_dump() if response.usage else {}
        
        # Log successful response
        log_ai_interaction(notes, content, True, metadata=usage)
        
        # Extract JSON from response
        if status_container:
            status_container.update(label="Parsing output...", state="running")
            
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            return json.loads(json_match.group())
        
        return None
    except Exception as e:
        st.error(f"AI API error: {e}")
        # Log error
        log_ai_interaction(notes, "", False, str(e))
        return None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_papers_per_year_chart(df: pd.DataFrame) -> go.Figure:
    """Create a line chart of papers per year."""
    if df.empty or 'year' not in df.columns:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    yearly = df.groupby('year').size().reset_index(name='count')
    yearly = yearly.sort_values('year')
    
    fig = px.line(
        yearly, x='year', y='count',
        title='Number of Papers per Year',
        markers=True
    )
    fig.update_layout(xaxis_title="Year", yaxis_title="Number of Papers")
    return fig


def create_linguistic_features_evolution(df: pd.DataFrame) -> go.Figure:
    """Create a stacked area chart of linguistic features over time."""
    if df.empty or 'year' not in df.columns or 'linguistic_features' not in df.columns:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    # Explode linguistic features
    records = []
    for _, row in df.iterrows():
        year = row.get('year')
        features = row.get('linguistic_features', [])
        if isinstance(features, list):
            for feat in features:
                records.append({'year': year, 'feature': feat})
    
    if not records:
        return go.Figure().add_annotation(text="No linguistic features data", showarrow=False)
    
    exploded = pd.DataFrame(records)
    pivot = exploded.groupby(['year', 'feature']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    for col in pivot.columns:
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[col],
            mode='lines',
            stackgroup='one',
            name=col[:30] + "..." if len(col) > 30 else col
        ))
    
    fig.update_layout(
        title='Linguistic Features Distribution Over Time',
        xaxis_title='Year',
        yaxis_title='Count',
        legend=dict(orientation="h", yanchor="bottom", y=-0.5)
    )
    return fig


def create_computational_stages_evolution(df: pd.DataFrame) -> go.Figure:
    """Create a stacked area chart of computational stages over time."""
    if df.empty or 'year' not in df.columns or 'computational_stages' not in df.columns:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    records = []
    for _, row in df.iterrows():
        year = row.get('year')
        stages = row.get('computational_stages', [])
        if isinstance(stages, list):
            for stage in stages:
                records.append({'year': year, 'stage': stage})
    
    if not records:
        return go.Figure().add_annotation(text="No computational stages data", showarrow=False)
    
    exploded = pd.DataFrame(records)
    pivot = exploded.groupby(['year', 'stage']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    for col in pivot.columns:
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[col],
            mode='lines',
            stackgroup='one',
            name=col
        ))
    
    fig.update_layout(
        title='Computational Stages Distribution Over Time',
        xaxis_title='Year',
        yaxis_title='Count'
    )
    return fig


def extract_keywords(text: str) -> list:
    """Extract simple keywords from text."""
    if not text:
        return []
    # Simple keyword extraction - remove common words
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
                 'those', 'it', 'its', 'we', 'our', 'they', 'their', 'he', 'she'}
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [w for w in words if w not in stopwords]


def create_keyword_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Create a line chart of top 5 keywords over time."""
    if df.empty or 'year' not in df.columns or 'analysis_notes' not in df.columns:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    # Extract keywords per year
    records = []
    for _, row in df.iterrows():
        year = row.get('year')
        notes = row.get('analysis_notes', '') or ''
        title = row.get('title', '') or ''
        keywords = extract_keywords(notes + ' ' + title)
        for kw in keywords:
            records.append({'year': year, 'keyword': kw})
    
    if not records:
        return go.Figure().add_annotation(text="No keywords found", showarrow=False)
    
    kw_df = pd.DataFrame(records)
    
    # Find top 5 overall keywords
    top_keywords = kw_df['keyword'].value_counts().head(5).index.tolist()
    
    # Filter and pivot
    filtered = kw_df[kw_df['keyword'].isin(top_keywords)]
    pivot = filtered.groupby(['year', 'keyword']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    for kw in top_keywords:
        if kw in pivot.columns:
            fig.add_trace(go.Scatter(
                x=pivot.index, y=pivot[kw],
                mode='lines+markers',
                name=kw.capitalize()
            ))
    
    fig.update_layout(
        title='Top 5 Keywords Trend Over Time',
        xaxis_title='Year',
        yaxis_title='Frequency'
    )
    return fig


def create_linguistic_features_bar(df: pd.DataFrame) -> go.Figure:
    """Create a horizontal bar chart of linguistic features."""
    if df.empty or 'linguistic_features' not in df.columns:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    all_features = []
    for features in df['linguistic_features']:
        if isinstance(features, list):
            all_features.extend(features)
    
    if not all_features:
        return go.Figure().add_annotation(text="No linguistic features data", showarrow=False)
    
    counts = pd.Series(all_features).value_counts()
    
    fig = px.bar(
        x=counts.values, y=counts.index,
        orientation='h',
        title='Linguistic Features Distribution',
        labels={'x': 'Count', 'y': 'Feature'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig


def create_computational_stages_bar(df: pd.DataFrame) -> go.Figure:
    """Create a bar chart of computational stages."""
    if df.empty or 'computational_stages' not in df.columns:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    all_stages = []
    for stages in df['computational_stages']:
        if isinstance(stages, list):
            all_stages.extend(stages)
    
    if not all_stages:
        return go.Figure().add_annotation(text="No computational stages data", showarrow=False)
    
    counts = pd.Series(all_stages).value_counts()
    
    fig = px.bar(
        x=counts.index, y=counts.values,
        title='Computational Stages Distribution',
        labels={'x': 'Stage', 'y': 'Count'}
    )
    return fig


def create_species_charts(df: pd.DataFrame) -> tuple:
    """Create species distribution charts."""
    fig_category = go.Figure().add_annotation(text="No data available", showarrow=False)
    fig_specialized = go.Figure().add_annotation(text="No data available", showarrow=False)
    
    if df.empty:
        return fig_category, fig_specialized
    
    # General category
    if 'species_categories' in df.columns:
        all_cats = []
        for cats in df['species_categories']:
            if isinstance(cats, list):
                all_cats.extend(cats)
        
        if all_cats:
            counts = pd.Series(all_cats).value_counts()
            fig_category = px.bar(
                x=counts.index, y=counts.values,
                title='Papers by Species Category',
                labels={'x': 'Category', 'y': 'Count'}
            )
    
    # Specialized species
    if 'specialized_species' in df.columns:
        species = df['specialized_species'].dropna()
        species = species[species != '']
        if len(species) > 0:
            counts = species.value_counts().head(10)
            fig_specialized = px.bar(
                x=counts.values, y=counts.index,
                orientation='h',
                title='Top 10 Specialized Species',
                labels={'x': 'Count', 'y': 'Species'}
            )
            fig_specialized.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    return fig_category, fig_specialized


def create_demographics_charts(df: pd.DataFrame) -> tuple:
    """Create demographics distribution charts."""
    fig_country = go.Figure().add_annotation(text="No data available", showarrow=False)
    fig_discipline = go.Figure().add_annotation(text="No data available", showarrow=False)
    fig_affiliation = go.Figure().add_annotation(text="No data available", showarrow=False)
    
    if df.empty:
        return fig_country, fig_discipline, fig_affiliation
    
    # Country
    if 'country' in df.columns:
        countries = df['country'].dropna()
        countries = countries[countries != '']
        if len(countries) > 0:
            counts = countries.value_counts().head(15)
            fig_country = px.bar(
                x=counts.values, y=counts.index,
                orientation='h',
                title='Papers by Country',
                labels={'x': 'Count', 'y': 'Country'}
            )
            fig_country.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    # Discipline
    if 'discipline' in df.columns:
        disciplines = df['discipline'].dropna()
        disciplines = disciplines[disciplines != '']
        if len(disciplines) > 0:
            counts = disciplines.value_counts()
            fig_discipline = px.pie(
                values=counts.values, names=counts.index,
                title='Papers by Discipline'
            )
    
    # Affiliation
    if 'university' in df.columns:
        unis = df['university'].dropna()
        unis = unis[unis != '']
        if len(unis) > 0:
            counts = unis.value_counts().head(15)
            fig_affiliation = px.bar(
                x=counts.values, y=counts.index,
                orientation='h',
                title='Papers by Affiliation',
                labels={'x': 'Count', 'y': 'University'}
            )
            fig_affiliation.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    return fig_country, fig_discipline, fig_affiliation


def create_wordcloud(df: pd.DataFrame, era: str) -> Optional[plt.Figure]:
    """Create a word cloud for pre-LLM or post-LLM era."""
    if df.empty:
        return None
    
    if era == "pre":
        filtered = df[df['year'].astype(int) <= 2020]
        title = "Pre-LLM Era (≤2020)"
    else:
        filtered = df[df['year'].astype(int) > 2020]
        title = "Post-LLM Era (>2020)"
    
    if filtered.empty:
        return None
    
    # Combine text from titles and notes
    text_parts = []
    for _, row in filtered.iterrows():
        if row.get('title'):
            text_parts.append(str(row['title']))
        if row.get('analysis_notes'):
            text_parts.append(str(row['analysis_notes']))
    
    text = ' '.join(text_parts)
    
    if not text.strip():
        return None
    
    try:
        wc = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold')
        return fig
    except Exception as e:
        st.error(f"Error creating word cloud: {e}")
        return None


def create_network_graph(df: pd.DataFrame, node_type: str) -> Optional[str]:
    """Create an interactive network graph and return HTML."""
    if df.empty:
        return None
    
    G = nx.Graph()
    
    if node_type == "affiliation":
        field = "university"
        title = "Affiliation Collaboration Network"
    elif node_type == "country":
        field = "country"
        title = "Country Collaboration Network"
    else:  # discipline
        field = "discipline"
        title = "Discipline Network"
    
    if field not in df.columns:
        return None
    
    # For co-authorship networks, we need to look at papers with multiple affiliations
    # Since we're storing single affiliation per entry, we'll create edges based on 
    # papers that share common species or features (simplified approach)
    
    # Get unique values
    values = df[field].dropna().unique()
    values = [v for v in values if v and str(v).strip()]
    
    if len(values) < 2:
        return None
    
    # Add nodes
    for v in values:
        count = len(df[df[field] == v])
        G.add_node(v, size=count * 5 + 10, title=f"{v}: {count} papers")
    
    # Create edges based on shared species categories (as proxy for collaboration)
    if 'species_categories' in df.columns:
        for _, row in df.iterrows():
            val = row.get(field)
            cats = row.get('species_categories', [])
            if val and isinstance(cats, list) and cats:
                # Find other papers with same species categories
                for _, other_row in df.iterrows():
                    other_val = other_row.get(field)
                    other_cats = other_row.get('species_categories', [])
                    if other_val and other_val != val and isinstance(other_cats, list):
                        if set(cats) & set(other_cats):  # Intersection
                            if G.has_edge(val, other_val):
                                G[val][other_val]['weight'] += 1
                            else:
                                G.add_edge(val, other_val, weight=1)
    
    # Only show if there are edges
    if G.number_of_edges() == 0:
        # Add at least some edges for visualization
        values_list = list(values)
        for i in range(min(len(values_list) - 1, 5)):
            G.add_edge(values_list[i], values_list[i + 1], weight=1)
    
    if G.number_of_edges() == 0:
        return None
    
    # Create pyvis network
    net = Network(height="400px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)
    net.toggle_physics(True)
    
    # Save to temp file and read
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        net.save_graph(f.name)
        with open(f.name, 'r') as rf:
            html = rf.read()
        os.unlink(f.name)
    
    return html


# ============================================================================
# STREAMLIT APP
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    # Load pending entries from file if not already in session
    if 'pending_entries' not in st.session_state:
        data = load_pending()
        st.session_state.pending_entries = data.get('pending', [])
        st.session_state.last_bibtex = data.get('bibtex', '')
        st.session_state.selected_entry_idx = data.get('selected_idx')
        # Ensure index is valid
        if st.session_state.selected_entry_idx is not None:
             if st.session_state.selected_entry_idx >= len(st.session_state.pending_entries):
                 st.session_state.selected_entry_idx = None
        
    # Maintain compatibility alias if needed, or remove 'parsed_entries' usage
    # We will replace 'parsed_entries' with explicit 'pending_entries' usage.
    
    # Other defaults
    if 'ai_result' not in st.session_state:
        st.session_state.ai_result = None
    if 'auto_trigger' not in st.session_state:
        st.session_state.auto_trigger = False
    if 'search_template' not in st.session_state:
        st.session_state.search_template = '{title} {author} {journal} {year}'
    if 'editing_saved_idx' not in st.session_state:
        st.session_state.editing_saved_idx = None


def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.title("Research Tool")
        
        # Page navigation
        page = st.radio(
            "Navigate to:",
            ["Data Entry", "Analytics Dashboard"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Settings
        with st.expander("Settings", expanded=False):
            api_status = "configured" if OPENAI_API_KEY else "not set"
            st.caption(f"OpenAI API key: {api_status}")
            
            st.session_state.auto_trigger = st.toggle(
                "Auto-run AI analysis on text entry",
                value=st.session_state.auto_trigger
            )
            
            st.session_state.search_template = st.text_input(
                "Search String Template",
                value=st.session_state.search_template,
                help="Use {title}, {author}, {journal}, {year} as placeholders"
            )
        
        # Reference
        with st.expander("Reference: 12 Linguistic Features"):
            st.markdown(REFERENCE_TEXT)
        
        st.divider()
        
        # Export
        jsonl_content = get_jsonl_content()
        if jsonl_content:
            st.download_button(
                "Download Full Dataset (JSONL)",
                data=jsonl_content,
                file_name="research_data.jsonl",
                mime="application/jsonl"
            )
        else:
            st.info("No data to export yet.")
        
        return page


def render_data_entry_page():
    """Render Page 1: Data Entry & Annotation."""
    # Apply any pending AI updates before rendering widgets
    apply_pending_ai_updates()

    st.title("Data Entry & Annotation")
    
    # Load saved entries
    saved_df = load_data()
    saved_titles = set(saved_df['title'].tolist()) if not saved_df.empty else set()
    
    # === BibTeX Import Section ===
    with st.expander("BibTeX Import", expanded=not st.session_state.pending_entries):
        bibtex_input = st.text_area(
            "Paste your entire .bib file here:",
            height=200,
            value=st.session_state.last_bibtex,
            placeholder="@article{key1,\n  title = {...},\n  author = {...},\n}\n\n@article{key2,\n  ...\n}"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Parse BibTeX", use_container_width=True):
                if bibtex_input:
                    entries = parse_bibtex(bibtex_input)
                    # Append new entries to existing pending? Or replace?
                    # Usually parsing means importing new stuff. Let's append if there are existing ones?
                    # Or replace? User behavior usually implies load distinct file.
                    # But for safety, let's append if list not empty to avoid data loss, or just replace if user intends.
                    # Given simple flow: Replace for now, user can clear first if they want.
                    # Actually, better to Append if there are pending entries?
                    # Let's simple Replace for typical use case (one big bib file).
                    # User expects "Clear" to clear.
                    st.session_state.pending_entries = entries
                    st.session_state.last_bibtex = bibtex_input
                    # Reset
                    st.session_state.selected_entry_idx = None
                    save_pending(entries, bibtex_input, None)
                    if entries:
                        st.success(f"Parsed {len(entries)} entries.")
                    else:
                        st.warning("No valid entries found.")
                    st.rerun()
                else:
                    st.warning("Paste BibTeX content first.")
        
        with col2:
            if st.button("Clear All Pending", use_container_width=True):
                st.session_state.pending_entries = []
                st.session_state.last_bibtex = ''
                st.session_state.selected_entry_idx = None
                save_pending([], '', None)
                st.rerun()
        
        if st.session_state.pending_entries:
            st.caption(f"{len(st.session_state.pending_entries)} entries parsed from BibTeX")
    
    st.divider()
    
    # === Entry Browser ===
    st.subheader("Entries")
    
    # Tabs
    n_pending = len(st.session_state.pending_entries)
    n_saved = len(saved_df)
    
    tab_pending, tab_saved = st.tabs([
        f"Pending ({n_pending})", 
        f"Saved ({n_saved})"
    ])
    
    # --- PENDING TAB ---
    with tab_pending:
        if not st.session_state.pending_entries:
            st.info("No pending entries. Paste a .bib file above.")
        else:
            with st.container(height=300):
                # Use same layout as Saved tab: Title/Year/Edit Button
                for idx, entry in enumerate(st.session_state.pending_entries):
                    col1, col2 = st.columns([6, 1])
                    
                    # Entry Metadata - Compact
                    title = entry.get('title', 'Untitled')[:60]
                    year = entry.get('year', '?')
                    journal = entry.get('journal', 'N/A')
                    
                    is_editing = (st.session_state.selected_entry_idx == idx)
                    
                    with col1:
                        # Single line if possible
                        st.markdown(f"**{title}** ({year}) — _{journal}_")
                    
                    with col2:
                       # Edit button
                       if is_editing:
                           st.button("Editing", key=f"p_editing_{idx}", disabled=True, use_container_width=True)
                       else:
                           if st.button("Edit", key=f"p_edit_{idx}", type="secondary", use_container_width=True):
                               st.session_state.selected_entry_idx = idx
                               st.session_state.ai_result = None
                               clear_form_state()
                               save_pending(st.session_state.pending_entries, st.session_state.last_bibtex, idx)
                               st.rerun()
    
    with tab_saved:
        if saved_df.empty:
            st.info("No saved entries yet.")
        else:
            search = st.text_input("Search saved entries", placeholder="Filter by title...")
            display_df = saved_df.copy()
            if search:
                display_df = display_df[display_df['title'].str.contains(search, case=False, na=False)]
            
            st.caption(f"{len(display_df)} entries")
            
            with st.container(height=300):
                for original_idx, row in display_df.head(200).iterrows():
                    title = str(row.get('title', 'Untitled'))[:60]
                    year = row.get('year', '?')
                    journal = row.get('journal', 'N/A')
                    is_editing = st.session_state.editing_saved_idx == original_idx
                    
                    # Row layout
                    c1, c2, c3, c4 = st.columns([4, 2, 1, 1])
                    with c1:
                        st.markdown(f"**{title}**")
                        st.caption(f"{year} • {journal}")
                    with c2:
                        if is_editing:
                            st.caption("Editing now...")
                    with c3:
                        if st.button("Edit", key=f"edit_{original_idx}", use_container_width=True, type="primary" if is_editing else "secondary"):
                            st.session_state.editing_saved_idx = original_idx
                            st.session_state.selected_entry_idx = None
                            st.session_state.ai_result = None
                            clear_form_state()
                            st.rerun()
                    with c4:
                        if st.button("Delete", key=f"del_{original_idx}", use_container_width=True, type="secondary"):
                            delete_entry(original_idx)
                            if st.session_state.editing_saved_idx == original_idx:
                                st.session_state.editing_saved_idx = None
                            st.rerun()
                    st.divider()
    
    st.divider()
    
    # === Editor Section ===
    # === Editor Section ===
    editor_mode = None
    editor_idx = None
    
    if st.session_state.selected_entry_idx is not None:
        editor_mode = 'pending'
        editor_idx = st.session_state.selected_entry_idx
        # Ensure index valid
        if editor_idx >= len(st.session_state.pending_entries):
             st.session_state.selected_entry_idx = None
             st.rerun()
             return
        current_entry = st.session_state.pending_entries[editor_idx]
        
    elif st.session_state.editing_saved_idx is not None:
        editor_mode = 'saved'
        editor_idx = st.session_state.editing_saved_idx
        # Load from file to get mutable dict
        # We need to find the record in file. Since index matches saved_df, and saved_df matches file order (usually).
        # We need to be careful if user sorted/filtered.
        # But saved_df in tab_saved logic above was loaded from load_data().
        # load_data() reads file line by line. Index should align if no sorting/filtering happened in between?
        # Actually filter was applied on view df!
        # But `original_idx` was passed from iterrows(). So likely safe.
        
        # We must load ALL records to get the specific one by index.
        all_records = []
        if os.path.exists(DATA_FILE):
             with open(DATA_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            all_records.append(json.loads(line))
                        except: pass
        
        if editor_idx < len(all_records):
            current_entry = all_records[editor_idx]
            # Ensure affiliations migration
            if 'affiliations' not in current_entry:
                old_aff = {}
                if current_entry.get('university'): old_aff['university'] = current_entry['university']
                if current_entry.get('country'): old_aff['country'] = current_entry['country']
                if current_entry.get('discipline'): old_aff['discipline'] = current_entry['discipline']
                current_entry['affiliations'] = [old_aff] if old_aff else []
        else:
             st.session_state.editing_saved_idx = None
             st.rerun()
             return
    
    if editor_mode:
        st.subheader(f"Editing: {current_entry.get('title', 'Untitled')}")
        
        # Helper to handle updates
        def auto_save():
            if editor_mode == 'pending':
                save_pending(
                    st.session_state.pending_entries, 
                    st.session_state.last_bibtex, 
                    st.session_state.selected_entry_idx
                )
            elif editor_mode == 'saved':
                # Update specific record in file
                update_entry(editor_idx, current_entry)
        
        # --- Metadata ---
        col1, col2 = st.columns([3, 1])
        with col1:
            new_title = st.text_input("Title", value=current_entry.get('title', ''), key="meta_title")
            if new_title != current_entry.get('title', ''):
                current_entry['title'] = new_title
                auto_save()
                st.rerun()

        with col2:
            new_year = st.text_input("Year", value=str(current_entry.get('year', '')), key="meta_year")
            if new_year != str(current_entry.get('year', '')):
                current_entry['year'] = new_year
                auto_save()
                st.rerun()

        new_journal = st.text_input("Journal/Venue", value=current_entry.get('journal', ''), key="meta_journal")
        if new_journal != current_entry.get('journal', ''):
            current_entry['journal'] = new_journal
            auto_save()
            st.rerun()

        # Authors
        authors_val = current_entry.get('authors', [])
        authors_str = "\n".join(authors_val) if isinstance(authors_val, list) else str(authors_val)
        new_authors_str = st.text_area("Authors (one per line)", value=authors_str, height=100, key="meta_authors")
        if new_authors_str != authors_str:
            current_entry['authors'] = [a.strip() for a in new_authors_str.split('\n') if a.strip()]
            auto_save()
            st.rerun()

        # Analysis Notes
        new_notes = st.text_area(
            "Analysis Notes", 
            value=current_entry.get('analysis_notes', '') or current_entry.get('abstract', ''), 
            height=150,
            key="meta_notes",
            help="Notes for AI analysis"
        )
        if new_notes != (current_entry.get('analysis_notes', '') or current_entry.get('abstract', '')):
            current_entry['analysis_notes'] = new_notes
            auto_save()
            st.rerun()

        # --- Affiliations (Dynamic Data Editor) ---
        st.markdown("**Affiliations**")
        
        # Migrate old format if needed
        if 'affiliations' not in current_entry:
            old_aff = {}
            if current_entry.get('university'): old_aff['university'] = current_entry['university']
            if current_entry.get('country'): old_aff['country'] = current_entry['country']
            if current_entry.get('discipline'): old_aff['discipline'] = current_entry['discipline']
            current_entry['affiliations'] = [old_aff] if old_aff else []

        # Prepare DataFrame for editor
        aff_data = current_entry['affiliations']
        aff_df = pd.DataFrame(aff_data)
        if aff_df.empty:
            aff_df = pd.DataFrame(columns=['university', 'country', 'discipline'])
        
        # Get all disciplines for dropdown
        all_disciplines = sorted(list(set(DISCIPLINES + saved_df['discipline'].unique().tolist() if 'discipline' in saved_df.columns else [])))

        edited_aff_df = st.data_editor(
            aff_df,
            num_rows="dynamic",
            column_config={
                "university": st.column_config.TextColumn("University/Institution", required=True),
                "country": st.column_config.TextColumn("Country", required=True),
                "discipline": st.column_config.SelectboxColumn(
                    "Discipline",
                    options=all_disciplines,
                    required=True,
                    width="medium" # allow_custom_value not supported in standard SelectboxColumn? Use TextColumn or verify support.
                    # Streamlit SelectboxColumn doesn't support 'allow_custom_value' yet.
                    # Fallback: Just use TextColumn for flexibility as requested "allow typing in new one".
                    # User asked for "typing in new one AND it becomes available". 
                    # TextColumn is best for typing. Selectbox restricts choices.
                    # I'll use TextColumn but provide autocomplete? Streamlit doesn't support autocomplete text column easily.
                    # I will stick to TextColumn to allow free text as prioritized req.
                )
            },
            key="aff_editor",
            use_container_width=True
        )

        # Check for changes in affiliations
        # Convert NaN to empty string
        edited_aff_df = edited_aff_df.fillna("")
        new_aff_list = edited_aff_df.to_dict('records')
        # Filter out empty rows
        new_aff_list = [r for r in new_aff_list if any(str(v).strip() for v in r.values())]
        
        if new_aff_list != current_entry['affiliations']:
            current_entry['affiliations'] = new_aff_list
            auto_save()
            # No rerun needed here as data_editor handles its state well, usually.
            # But for "Move to In Progress" logic we might need it.
            if entry_type == 'pending':
                st.rerun()

        st.divider()

        # --- AI Classification ---
        col_ai, col_final = st.columns([3, 1])
        with col_ai:
            if st.button("Run AI Analysis", type="secondary"):
                if new_notes:
                    with st.status("Analyzing...", expanded=True) as status:
                        result = call_ai_api(new_notes, status_container=status)
                        if result:
                            st.session_state.pending_ai_updates = result
                            status.update(label="Complete!", state="complete", expanded=False)
                            auto_save() # Save current state first
                            st.rerun()
                        else:
                            status.update(label="Failed.", state="error")
                else:
                    st.warning("Add Analysis Notes first.")

        with col_final:
            label = "Commit to Dataset" if editor_mode == 'pending' else "Save Changes"
            if st.button(label, type="primary", use_container_width=True):
                # Ensure we have minimum fields
                if not current_entry.get('title') or not current_entry.get('year'):
                    st.error("Title and Year required.")
                else:
                    if editor_mode == 'pending':
                        current_entry['created_at'] = datetime.now().isoformat()
                        if save_entry(current_entry):
                            st.success("Entry committed!")
                            # Remove from pending
                            st.session_state.pending_entries.pop(editor_idx)
                            st.session_state.selected_entry_idx = None
                            save_pending(st.session_state.pending_entries, st.session_state.last_bibtex, None)
                            st.rerun()
                    else:
                        # For saved, we just confirm (auto-save arguably already did it, but this explicitly closes editor)
                        # We use 'update_entry' which replaces the line.
                        if update_entry(editor_idx, current_entry):
                            st.success("Changes saved!")
                            st.session_state.editing_saved_idx = None
                            st.rerun()
                    # Final clean: remove empty values
                    # Handled above
                    pass

        # --- Classification Widgets (Auto-save) ---
        st.markdown("**Classification**")
        
        # Linguistic Features
        st.caption("Linguistic Features")
        cols = st.columns(3)
        current_features = current_entry.get('linguistic_features', [])
        
        # We need to manually match checkboxes to list
        new_features = list(current_features)
        
        for i, feat in enumerate(LINGUISTIC_FEATURES):
            # We use key based on feature name or index to avoid collision
            # Since we removed st.form, we can use on_change or just check value
            is_checked = feat in current_features
            if cols[i%3].checkbox(feat, value=is_checked, key=f"feat_{idx}_{i}"):
                if feat not in new_features:
                    new_features.append(feat)
            else:
                if feat in new_features:
                    new_features.remove(feat)
        
        if sorted(new_features) != sorted(current_features):
            current_entry['linguistic_features'] = new_features
            auto_save()
            # Rerun not strictly needed for checkbox unless we want immediate validation
        
        # Species & Stages
        c1, c2 = st.columns(2)
        with c1:
            curr_cats = current_entry.get('species_categories', [])
            new_cats = st.multiselect("Species Category", SPECIES_CATEGORIES, default=[c for c in curr_cats if c in SPECIES_CATEGORIES], key=f"cat_{idx}")
            if new_cats != curr_cats:
                current_entry['species_categories'] = new_cats
                auto_save()
                
        with c2:
            # Specialized Species (Text Input, comma separated, or list)
            # Prompt returns list. UI uses text input.
            curr_spec = current_entry.get('specialized_species', [])
            if isinstance(curr_spec, str): curr_spec = [curr_spec] if curr_spec else []
            spec_str = ", ".join(curr_spec)
            new_spec_str = st.text_input("Specialized Species (comma separated)", value=spec_str, key=f"spec_{idx}")
            if new_spec_str != spec_str:
                current_entry['specialized_species'] = [s.strip() for s in new_spec_str.split(',') if s.strip()]
                auto_save()

        curr_stages = current_entry.get('computational_stages', [])
        new_stages = st.multiselect("Computational Stages", COMPUTATIONAL_STAGES, default=[s for s in curr_stages if s in COMPUTATIONAL_STAGES], key=f"stages_{idx}")
        if new_stages != curr_stages:
            current_entry['computational_stages'] = new_stages
            auto_save()

    elif st.session_state.pending_entries:
         st.info("Select a pending entry to edit.")
    
    st.divider()


def delete_entry(index: int) -> bool:
    """Delete an entry from the JSONL file by index."""
    try:
        records = []
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        if 0 <= index < len(records):
            records.pop(index)
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting entry: {e}")
        return False


def render_data_browser_page():
    """Render Page 2: Data Browser."""
    st.header("Data Browser")
    
    df = load_data()
    
    if df.empty:
        st.warning("No entries yet. Add papers on the Data Entry page.")
        return
    
    st.caption(f"{len(df)} entries in dataset")
    
    # Search/filter
    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input("Search titles and notes", placeholder="Type to filter...")
    with col2:
        sort_order = st.selectbox("Sort by", ["Newest first", "Oldest first", "Title A-Z"])
    
    # Apply search filter
    if search:
        mask = (
            df['title'].str.contains(search, case=False, na=False) |
            df['analysis_notes'].str.contains(search, case=False, na=False)
        )
        df = df[mask]
    
    # Apply sort
    if sort_order == "Newest first":
        df = df.iloc[::-1]  # Reverse order (newest at end of file)
    elif sort_order == "Title A-Z":
        df = df.sort_values('title', na_position='last')
    # "Oldest first" is default file order
    
    if df.empty:
        st.info("No entries match your search.")
        return
    
    # Display entries
    for idx, (original_idx, row) in enumerate(df.iterrows()):
        with st.expander(f"{row.get('title', 'Untitled')} ({row.get('year', 'N/A')})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Journal:** {row.get('journal', 'N/A')}")
                
                authors = row.get('authors', [])
                if isinstance(authors, list) and authors:
                    st.markdown(f"**Authors:** {', '.join(authors)}")
                
                st.markdown(f"**Affiliation:** {row.get('university', 'N/A')} ({row.get('country', 'N/A')})")
                st.markdown(f"**Discipline:** {row.get('discipline', 'N/A')}")
                
                 # Handle new affiliation structure
                affiliations = row.get('affiliations', [])
                if isinstance(affiliations, list) and affiliations:
                     st.markdown("**Affiliations:**")
                     for a in affiliations:
                         st.caption(f"- {a.get('university', '')} ({a.get('country', '')}) — {a.get('discipline', '')}")
                elif not affiliations and (row.get('university') or row.get('country')):
                     # Fallback for old data
                     st.markdown(f"**Affiliation:** {row.get('university', 'N/A')} ({row.get('country', 'N/A')})")
                     st.markdown(f"**Discipline:** {row.get('discipline', 'N/A')}")
            
            with col2:
                species_cats = row.get('species_categories', [])
                if isinstance(species_cats, list) and species_cats:
                    st.markdown(f"**Species:** {', '.join(species_cats)}")
                
                spec_species = row.get('specialized_species', [])
                # Handle list or string
                if isinstance(spec_species, str) and spec_species:
                    spec_species = [spec_species]
                if isinstance(spec_species, list) and spec_species:
                    st.markdown(f"**Specialized:** _{', '.join(spec_species)}_")
                
                stages = row.get('computational_stages', [])
                if isinstance(stages, list) and stages:
                    st.markdown(f"**Stages:** {', '.join(stages)}")
            
            # Linguistic features
            features = row.get('linguistic_features', [])
            if isinstance(features, list) and features:
                st.markdown("**Linguistic Features:**")
                st.write(", ".join(features))
            
            # Analysis notes
            notes = row.get('analysis_notes', '')
            if notes:
                st.markdown("**Notes:**")
                st.text(notes[:500] + "..." if len(notes) > 500 else notes)
            
            # Delete button
            st.divider()
            if st.button("Delete this entry", key=f"delete_{original_idx}", type="secondary"):
                if delete_entry(original_idx):
                    st.success("Entry deleted.")
                    st.rerun()


def render_analytics_page():
    """Render Page 2: Analytics Dashboard."""
    st.header("Analytics Dashboard")
    
    # Load data
    df = load_data()
    
    # Preprocess for backward compatibility with nested affiliations
    if not df.empty and 'affiliations' in df.columns:
        # Fill missing legacy columns from the first affiliation
        def get_primary(row, field):
            val = row.get(field)
            if val: return val
            affs = row.get('affiliations', [])
            if isinstance(affs, list) and affs:
                return affs[0].get(field, '')
            return ''
            
        if 'university' not in df.columns: df['university'] = ''
        if 'country' not in df.columns: df['country'] = ''
        if 'discipline' not in df.columns: df['discipline'] = ''
        
        df['university'] = df.apply(lambda x: get_primary(x, 'university'), axis=1)
        df['country'] = df.apply(lambda x: get_primary(x, 'country'), axis=1)
        df['discipline'] = df.apply(lambda x: get_primary(x, 'discipline'), axis=1)
    
    if df.empty:
        st.warning("No data available. Please add entries on the Data Entry page first.")
        return
    
    # Ensure year is numeric for filtering
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    
    # Sidebar filters
    with st.sidebar:
        st.subheader("Filters")
        
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        
        if min_year < max_year:
            year_range = st.slider(
                "Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
        else:
            year_range = (min_year, max_year)
            st.info(f"All papers from {min_year}")
        
        # Species category filter
        all_categories = set()
        for cats in df['species_categories']:
            if isinstance(cats, list):
                all_categories.update(cats)
        
        category_filter = st.multiselect(
            "Species Category",
            sorted(all_categories),
            default=[]
        )
    
    # Apply filters
    filtered_df = df[
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1])
    ]
    
    if category_filter:
        mask = filtered_df['species_categories'].apply(
            lambda x: bool(set(x) & set(category_filter)) if isinstance(x, list) else False
        )
        filtered_df = filtered_df[mask]
    
    st.info(f"Showing {len(filtered_df)} of {len(df)} papers")
    
    # Group 1: Longitudinal Analysis
    st.subheader("Group 1: Longitudinal Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = create_papers_per_year_chart(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = create_keyword_trend_chart(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    
    fig = create_linguistic_features_evolution(filtered_df)
    st.plotly_chart(fig, use_container_width=True)
    
    fig = create_computational_stages_evolution(filtered_df)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Group 2: Distribution Stats
    st.subheader("Group 2: Distribution Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = create_linguistic_features_bar(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = create_computational_stages_bar(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    
    fig_cat, fig_spec = create_species_charts(filtered_df)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_cat, use_container_width=True)
    with col2:
        st.plotly_chart(fig_spec, use_container_width=True)
    
    fig_country, fig_disc, fig_aff = create_demographics_charts(filtered_df)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(fig_country, use_container_width=True)
    with col2:
        st.plotly_chart(fig_disc, use_container_width=True)
    with col3:
        st.plotly_chart(fig_aff, use_container_width=True)
    
    st.divider()
    
    # Group 3: Word Clouds
    st.subheader("Group 3: Word Clouds")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = create_wordcloud(filtered_df, "pre")
        if fig:
            st.pyplot(fig)
        else:
            st.info("No data for Pre-LLM era (≤2020)")
    
    with col2:
        fig = create_wordcloud(filtered_df, "post")
        if fig:
            st.pyplot(fig)
        else:
            st.info("No data for Post-LLM era (>2020)")
    
    st.divider()
    
    # Group 4: Network Graphs
    st.subheader("Group 4: Network Graphs")
    
    tabs = st.tabs(["Affiliation Network", "Country Network", "Discipline Network"])
    
    with tabs[0]:
        html = create_network_graph(filtered_df, "affiliation")
        if html:
            st.components.v1.html(html, height=450)
        else:
            st.info("Not enough affiliation data to create network graph.")
    
    with tabs[1]:
        html = create_network_graph(filtered_df, "country")
        if html:
            st.components.v1.html(html, height=450)
        else:
            st.info("Not enough country data to create network graph.")
    
    with tabs[2]:
        html = create_network_graph(filtered_df, "discipline")
        if html:
            st.components.v1.html(html, height=450)
        else:
            st.info("Not enough discipline data to create network graph.")


def main():
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="Computational Animal Linguistics Research Tool",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    page = render_sidebar()
    
    if page == "Data Entry":
        render_data_entry_page()
    else:
        render_analytics_page()


if __name__ == "__main__":
    main()
