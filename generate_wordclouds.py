#!/usr/bin/env python3
"""
Generate word clouds for the survey tool dataset.
Creates three word clouds:
1. Pre-LLM era (≤2020)
2. Post-LLM era (>2020)
3. Difference word cloud (words that increased/decreased in frequency)
"""

import json
import re
import os
import io
import base64
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    try:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    except:
        nltk.download('averaged_perceptron_tagger', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = '/Users/ndesai-air/Documents/GitHub/clsurveytool/data'
OUTPUT_DIR = os.path.join(DATA_DIR, 'graphs/wordclouds')

# Dataset files
HUMAN_DATASET_FILE = os.path.join(DATA_DIR, 'human_dataset.jsonl')
SUBSET_DATASET_FILE = os.path.join(DATA_DIR, 'subset_dataset.jsonl')
FULLSET_DATASET_FILE = os.path.join(DATA_DIR, 'fullset_dataset.jsonl')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_dataset_file(filepath):
    """Load a JSONL dataset file."""
    if not os.path.exists(filepath):
        return []
    
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries

def load_data(sources):
    """Load data from specified sources, deduplicating by ID."""
    source_map = {
        'human': HUMAN_DATASET_FILE,
        'subset': SUBSET_DATASET_FILE,
        'fullset': FULLSET_DATASET_FILE
    }
    
    seen_ids = set()
    unique_entries = []
    
    for source in sources:
        if source in source_map:
            file_entries = load_dataset_file(source_map[source])
            for e in file_entries:
                eid = e.get('id')
                if eid and eid not in seen_ids:
                    seen_ids.add(eid)
                    unique_entries.append(e)
                elif not eid:
                    unique_entries.append(e)
    
    return unique_entries

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    from nltk.corpus import wordnet
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    """Lemmatize words in text."""
    if not text:
        return ""
    tokens = word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
    return ' '.join(lemmatized)

def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def get_word_frequencies(text, stopwords):
    """Get word frequencies from text."""
    words = text.split()
    # Filter out stopwords and short words
    filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
    return Counter(filtered_words)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_wordclouds():
    """Generate word clouds for pre-LLM, post-LLM, and difference."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("WORD CLOUD GENERATION")
    print("="*80)
    
    # Load data from human + subset
    print("\nLoading data from human + subset datasets...")
    entries = load_data(['human', 'subset'])
    print(f"Loaded {len(entries)} total papers")
    
    # Custom stopwords
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update([
        'paper', 'study', 'research', 'analysis', 'using', 'used', 'use',
        'based', 'results', 'show', 'found', 'also', 'however', 'may',
        'one', 'two', 'three', 'et', 'al', 'can', 'well', 'via',
        'moreover', 'furthermore', 'therefore', 'thus',
        'abstract', 'introduction', 'conclusion', 'discussion',
        'method', 'methods', 'approach', 'approaches', 'data',
        'present', 'presented', 'propose', 'proposed', 'although',
        'either', 's', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
        'l', 'm', 'n', 'o', 'p', 'q', 'r', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        'set', 'know', 'first', 'new', 'different', 'many', 'will', 'able',
        'doi', 'org', 'https', 'http', 'www', 'cid', 'preprint', 'fig',
        'bioarxiv', 'arxiv', 'pdf', 'com', 'site', 'version', 'posted',
        'author', 'funded', 'granted', 'peer', 'review', 'nature',
        'information', 'biorxiv', 'xxxx','published','jstor','sticas',
        'journal','para','california','andrews','license','copyright',
        
    ])
    
    # Collect text by era
    print("\nCollecting text by era...")
    text_pre_llm = []
    text_post_llm = []
    papers_pre_llm = 0
    papers_post_llm = 0
    
    for entry in entries:
        year_str = str(entry.get('year', '')).strip()
        try:
            year = int(year_str.split('-')[0]) if year_str else None
        except ValueError:
            year = None
        
        if year and 1900 <= year <= 2100:
            # Collect text from title and abstract only (not full PDF text - too slow)
            title = entry.get('title', '') or ''
            abstract = entry.get('abstract', '') or ''
            # Limit analysis notes to first 1000 chars to avoid processing huge PDFs
            notes = (entry.get('analysis_notes', '') or '')[:1000]
            text = clean_text(title + ' ' + abstract + ' ' + notes)
            
            if year <= 2020:
                text_pre_llm.append(text)
                papers_pre_llm += 1
            else:
                text_post_llm.append(text)
                papers_post_llm += 1
    
    print(f"Pre-LLM era (≤2020): {papers_pre_llm} papers")
    print(f"Post-LLM era (>2020): {papers_post_llm} papers")
    
    # Combine text (skip lemmatization - it's too slow and not necessary for word clouds)
    print("\nCombining text...")
    all_text_pre_llm = ' '.join(text_pre_llm)
    all_text_post_llm = ' '.join(text_post_llm)
    
    # ========================================================================
    # 1. PRE-LLM WORD CLOUD
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING PRE-LLM WORD CLOUD (≤2020)")
    print("="*80)
    
    wc_pre = WordCloud(
        width=1600,
        height=800,
        background_color='white',
        stopwords=custom_stopwords,
        max_words=100,
        colormap='viridis',
        relative_scaling=0.5,
        min_font_size=10,
        prefer_horizontal=0.7
    ).generate(all_text_pre_llm)
    
    plt.figure(figsize=(20, 10))
    plt.imshow(wc_pre, interpolation='bilinear')
    plt.axis('off')
    plt.title('Pre-LLM Era (≤2020)', fontsize=24, pad=20)
    plt.tight_layout(pad=0)
    
    # output_file = os.path.join(OUTPUT_DIR, 'wordcloud_pre_llm.png')
    # plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    # print(f"Saved: {output_file}")
    
    output_pdf = os.path.join(OUTPUT_DIR, 'wordcloud_pre_llm.pdf')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_pdf}")
    plt.close()
    
    # ========================================================================
    # 2. POST-LLM WORD CLOUD
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING POST-LLM WORD CLOUD (>2020)")
    print("="*80)
    
    wc_post = WordCloud(
        width=1600,
        height=800,
        background_color='white',
        stopwords=custom_stopwords,
        max_words=100,
        colormap='viridis',
        relative_scaling=0.5,
        min_font_size=10,
        prefer_horizontal=0.7
    ).generate(all_text_post_llm)
    
    plt.figure(figsize=(20, 10))
    plt.imshow(wc_post, interpolation='bilinear')
    plt.axis('off')
    plt.title('Post-LLM Era (>2020)', fontsize=24, pad=20)
    plt.tight_layout(pad=0)
    
    # output_file = os.path.join(OUTPUT_DIR, 'wordcloud_post_llm.png')
    # plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    # print(f"Saved: {output_file}")
    
    output_pdf = os.path.join(OUTPUT_DIR, 'wordcloud_post_llm.pdf')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_pdf}")
    plt.close()
    
    # ========================================================================
    # 3. DIFFERENCE WORD CLOUD
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING DIFFERENCE WORD CLOUD")
    print("="*80)
    
    # Calculate word frequencies
    print("\nCalculating word frequencies...")
    freq_pre_llm = get_word_frequencies(all_text_pre_llm, custom_stopwords)
    freq_post_llm = get_word_frequencies(all_text_post_llm, custom_stopwords)
    
    # Normalize by number of papers
    norm_freq_pre_llm = {word: count / papers_pre_llm for word, count in freq_pre_llm.items()}
    norm_freq_post_llm = {word: count / papers_post_llm for word, count in freq_post_llm.items()}
    
    # Calculate differences
    all_words = set(norm_freq_pre_llm.keys()) | set(norm_freq_post_llm.keys())
    
    frequency_diff = {}
    for word in all_words:
        pre_freq = norm_freq_pre_llm.get(word, 0)
        post_freq = norm_freq_post_llm.get(word, 0)
        # Only include words that appear in at least one period with reasonable frequency
        if pre_freq > 0.01 or post_freq > 0.01:
            frequency_diff[word] = post_freq - pre_freq
    
    # Separate into positive and negative differences
    positive_diff = {word: diff for word, diff in frequency_diff.items() if diff > 0}
    negative_diff = {word: abs(diff) for word, diff in frequency_diff.items() if diff < 0}
    
    print(f"\nWords with increased frequency: {len(positive_diff)}")
    print(f"Words with decreased frequency: {len(negative_diff)}")
    
    # Show top changes
    print("\nTop 30 words with INCREASED frequency:")
    sorted_positive = sorted(positive_diff.items(), key=lambda x: x[1], reverse=True)[:30]
    for word, diff in sorted_positive:
        print(f"  {word}: +{diff:.4f} (pre: {norm_freq_pre_llm.get(word, 0):.4f}, post: {norm_freq_post_llm.get(word, 0):.4f})")
    
    print("\nTop 30 words with DECREASED frequency:")
    sorted_negative = sorted(negative_diff.items(), key=lambda x: x[1], reverse=True)[:30]
    for word, diff in sorted_negative:
        print(f"  {word}: -{diff:.4f} (pre: {norm_freq_pre_llm.get(word, 0):.4f}, post: {norm_freq_post_llm.get(word, 0):.4f})")
    
    # Generate word cloud for increased frequency
    if positive_diff:
        print("\nGenerating word cloud for increased frequency words...")
        wc_increased = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10,
            colormap='Greens',  # Green for positive/increase (darker colors for accessibility)
            prefer_horizontal=0.7
        ).generate_from_frequencies(positive_diff)
        
        plt.figure(figsize=(20, 10))
        plt.imshow(wc_increased, interpolation='bilinear')
        plt.axis('off')
        plt.title('Words with Increased Frequency (Post-LLM vs Pre-LLM)', 
                  fontsize=24, pad=20)
        plt.tight_layout(pad=0)
        
        # output_file = os.path.join(OUTPUT_DIR, 'wordcloud_increased_frequency.png')
        # plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        # print(f"Saved: {output_file}")
        
        output_pdf = os.path.join(OUTPUT_DIR, 'wordcloud_increased_frequency.pdf')
        plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_pdf}")
        plt.close()
    
    # Generate word cloud for decreased frequency
    if negative_diff:
        print("\nGenerating word cloud for decreased frequency words...")
        wc_decreased = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10,
            colormap='Reds',  # Red for negative/decrease (darker colors for accessibility)
            prefer_horizontal=0.7
        ).generate_from_frequencies(negative_diff)
        
        plt.figure(figsize=(20, 10))
        plt.imshow(wc_decreased, interpolation='bilinear')
        plt.axis('off')
        plt.title('Words with Decreased Frequency (Post-LLM vs Pre-LLM)', 
                  fontsize=24, pad=20)
        plt.tight_layout(pad=0)
        
        # output_file = os.path.join(OUTPUT_DIR, 'wordcloud_decreased_frequency.png')
        # plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        # print(f"Saved: {output_file}")
        
        output_pdf = os.path.join(OUTPUT_DIR, 'wordcloud_decreased_frequency.pdf')
        plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_pdf}")
        plt.close()
    
    # ========================================================================
    # GENERATE LATEX CODE
    # ========================================================================
    # latex_code_pre = r"""\begin{figure}[th]
# \centering
# \includegraphics[width=0.9\textwidth]{wordcloud_pre_llm.pdf}
# \caption{Word cloud showing most frequent terms in papers published up to 2020 (Pre-LLM era).}
# \label{fig:wordcloud_pre_llm}
# \end{figure}"""
    
    # latex_code_post = r"""\begin{figure}[th]
# \centering
# \includegraphics[width=0.9\textwidth]{wordcloud_post_llm.pdf}
# \caption{Word cloud showing most frequent terms in papers published after 2020 (Post-LLM era).}
# \label{fig:wordcloud_post_llm}
# \end{figure}"""
    
    # latex_code_increased = r"""\begin{figure}[th]
# \centering
# \includegraphics[width=0.9\textwidth]{wordcloud_increased_frequency.pdf}
# \caption{Word cloud showing terms with increased frequency in papers published after 2020 compared to papers published up to 2020. Word size represents the magnitude of the frequency increase.}
# \label{fig:wordcloud_increased_frequency}
# \end{figure}"""
    
    # latex_code_decreased = r"""\begin{figure}[th]
# \centering
# \includegraphics[width=0.9\textwidth]{wordcloud_decreased_frequency.pdf}
# \caption{Word cloud showing terms with decreased frequency in papers published after 2020 compared to papers published up to 2020. Word size represents the magnitude of the frequency decrease.}
# \label{fig:wordcloud_decreased_frequency}
# \end{figure}"""
    
    # # Save LaTeX code
    # latex_file = os.path.join(OUTPUT_DIR, 'wordcloud_latex.tex')
    # with open(latex_file, 'w') as f:
    #     f.write("% Pre-LLM word cloud\n")
    #     f.write(latex_code_pre)
    #     f.write("\n\n% Post-LLM word cloud\n")
    #     f.write(latex_code_post)
    #     f.write("\n\n% Increased frequency\n")
    #     f.write(latex_code_increased)
    #     f.write("\n\n% Decreased frequency\n")
    #     f.write(latex_code_decreased)
    
    # print("\n" + "="*80)
    # print("LATEX CODE")
    # print("="*80)
    # print(f"\nLaTeX code saved to: {latex_file}")
    # print("\nPre-LLM:")
    # print(latex_code_pre)
    # print("\nPost-LLM:")
    # print(latex_code_post)
    # print("\nIncreased Frequency:")
    # print(latex_code_increased)
    # print("\nDecreased Frequency:")
    # print(latex_code_decreased)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nAll word clouds saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - wordcloud_pre_llm.png/pdf")
    print("  - wordcloud_post_llm.png/pdf")
    print("  - wordcloud_increased_frequency.png/pdf")
    print("  - wordcloud_decreased_frequency.png/pdf")
    print("  - wordcloud_latex.tex")

if __name__ == "__main__":
    generate_wordclouds()
