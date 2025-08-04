import streamlit as st
import pandas as pd
import re
from collections import defaultdict
import io

# Page configuration
st.set_page_config(
    page_title="CSV Keyword Search Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E7D4F;
        margin-bottom: 2rem;
    }
    .stat-card {
        background-color: #f0f9f4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10b981;
        margin: 0.5rem 0;
    }
    .keyword-found {
        background-color: #d1fae5;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border-left: 3px solid #10b981;
        margin: 0.25rem 0;
    }
    .keyword-not-found {
        background-color: #fef2f2;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border-left: 3px solid #ef4444;
        margin: 0.25rem 0;
    }
    .footer {
        text-align: center;
        color: #d97706;
        font-weight: bold;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'default_keywords' not in st.session_state:
        st.session_state.default_keywords = [
            "innovation", "sustainability", "growth", "technology", 
            "efficiency", "collaboration", "strategy", "quality",
            "customer", "market", "digital", "transformation"
        ]
    
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    if 'id_level_data' not in st.session_state:
        st.session_state.id_level_data = None

def parse_keywords(input_text):
    """Parse keywords from input text"""
    if not input_text:
        return []
    
    # Check if quotes are used
    if '"' in input_text:
        keywords = [k.strip().strip('"') for k in input_text.split(',')]
    else:
        # Split by comma or newline
        keywords = re.split(r'[,\n]', input_text)
    
    return [k.strip() for k in keywords if k.strip()]

def count_keyword_in_text(text, keyword, case_sensitive=False, exact_match=False):
    """Count occurrences of keyword in text"""
    search_text = text if case_sensitive else text.lower()
    search_term = keyword if case_sensitive else keyword.lower()
    
    if exact_match:
        words = re.findall(r'\b\w+\b', search_text)
        return words.count(search_term)
    else:
        # Escape special regex characters
        escaped_keyword = re.escape(search_term)
        flags = 0 if case_sensitive else re.IGNORECASE
        matches = re.findall(escaped_keyword, search_text, flags)
        return len(matches)

def calculate_id_level_data(data, keywords, case_sensitive=False, exact_match=False):
    """Calculate ID-level aggregated statistics"""
    # Group by ID
    id_groups = defaultdict(list)
    for _, row in data.iterrows():
        id_groups[row['ID']].append(row)
    
    total_ids = len(id_groups)
    total_words = 0
    id_stats = {}
    
    # Calculate stats for each ID
    for id_val, rows in id_groups.items():
        combined_text = ' '.join([str(row['Statement']) for row in rows])
        word_count = len(re.findall(r'\b\w+\b', combined_text))
        total_words += word_count
        
        # Find present keywords
        present_keywords = []
        for keyword in keywords:
            search_text = combined_text if case_sensitive else combined_text.lower()
            search_term = keyword if case_sensitive else keyword.lower()
            
            if exact_match:
                words = re.findall(r'\b\w+\b', search_text)
                if search_term in words:
                    present_keywords.append(keyword)
            else:
                if search_term in search_text:
                    present_keywords.append(keyword)
        
        id_stats[id_val] = {
            'word_count': word_count,
            'statement_count': len(rows),
            'present_keywords': present_keywords,
            'combined_text': combined_text
        }
    
    avg_words_per_id = total_words / total_ids if total_ids > 0 else 0
    avg_statements_per_id = len(data) / total_ids if total_ids > 0 else 0
    
    # Calculate classifier breakdown
    classifier_breakdown = []
    for i, keyword in enumerate(keywords):
        ids_with_keyword = [stats for stats in id_stats.values() 
                           if keyword in stats['present_keywords']]
        
        total_words_in_keyword_ids = sum(stats['word_count'] for stats in ids_with_keyword)
        percent_of_corpus = (total_words_in_keyword_ids / total_words * 100) if total_words > 0 else 0
        
        # Calculate average percentage within ID
        avg_percent_within_id = 0
        if ids_with_keyword:
            percentages = []
            for stats in ids_with_keyword:
                keyword_matches = count_keyword_in_text(stats['combined_text'], keyword, case_sensitive, exact_match)
                if stats['word_count'] > 0:
                    percentages.append(keyword_matches / stats['word_count'] * 100)
            
            if percentages:
                avg_percent_within_id = sum(percentages) / len(percentages)
        
        classifier_breakdown.append({
            'index': i,
            'classifier': keyword,
            'ids_with_classifier': len(ids_with_keyword),
            'total_words': total_words_in_keyword_ids,
            'percent_of_corpus': f"{percent_of_corpus:.2f}%",
            'avg_percent_within_id': f"{avg_percent_within_id:.2f}%"
        })
    
    # Sort by number of IDs with classifier
    classifier_breakdown.sort(key=lambda x: x['ids_with_classifier'], reverse=True)
    
    return {
        'aggregate_stats': {
            'total_ids': total_ids,
            'total_words': f"{total_words:,}",
            'avg_words_per_id': f"{avg_words_per_id:.2f}",
            'avg_statements_per_id': f"{avg_statements_per_id:.2f}"
        },
        'classifier_breakdown': classifier_breakdown
    }

def search_keywords(data, keywords, case_sensitive=False, exact_match=False):
    """Search for keywords in the data"""
    results = {}
    matching_rows = []
    
    for keyword in keywords:
        matches = []
        for idx, row in data.iterrows():
            statement = str(row['Statement'])
            search_text = statement if case_sensitive else statement.lower()
            search_term = keyword if case_sensitive else keyword.lower()
            
            found = False
            if exact_match:
                words = re.findall(r'\b\w+\b', search_text)
                found = search_term in words
            else:
                found = search_term in search_text
            
            if found:
                matches.append({
                    'row_index': idx,
                    'id': row['ID'],
                    'statement': statement,
                    'matched_keywords': [keyword]
                })
        
        results[keyword] = {
            'found': len(matches) > 0,
            'count': len(matches),
            'matches': matches
        }
        
        # Add to matching rows
        for match in matches:
            existing_row = next((r for r in matching_rows if r['row_index'] == match['row_index']), None)
            if existing_row:
                if keyword not in existing_row['matched_keywords']:
                    existing_row['matched_keywords'].append(keyword)
            else:
                matching_rows.append(match)
    
    # Calculate statistics
    total_statements = len(data)
    statements_with_matches = len(set(row['row_index'] for row in matching_rows))
    match_percentage = (statements_with_matches / total_statements * 100) if total_statements > 0 else 0
    
    return {
        'keywords': results,
        'matching_rows': sorted(matching_rows, key=lambda x: x['row_index']),
        'stats': {
            'total_keywords': len(keywords),
            'found_keywords': sum(1 for r in results.values() if r['found']),
            'total_statements': total_statements,
            'statements_with_matches': statements_with_matches,
            'match_percentage': f"{match_percentage:.1f}"
        }
    }

def create_results_csv(results, id_level_data):
    """Create CSV content for download"""
    lines = []
    
    # Search results summary
    lines.append("SEARCH RESULTS SUMMARY")
    lines.append(f"Keywords Found,{results['stats']['found_keywords']}")
    lines.append(f"Statements with Matches,{results['stats']['statements_with_matches']}")
    lines.append(f"Match Rate,{results['stats']['match_percentage']}%")
    lines.append(f"Total Statements,{results['stats']['total_statements']}")
    lines.append("")
    
    # ID-level aggregated data
    lines.append("ID-LEVEL AGGREGATED DATA")
    lines.append(f"Total IDs,{id_level_data['aggregate_stats']['total_ids']}")
    lines.append(f"Total Words,{id_level_data['aggregate_stats']['total_words']}")
    lines.append(f"Avg Words/ID,{id_level_data['aggregate_stats']['avg_words_per_id']}")
    lines.append(f"Avg Statements/ID,{id_level_data['aggregate_stats']['avg_statements_per_id']}")
    lines.append("")
    
    # Classifier breakdown
    lines.append("ID-LEVEL CLASSIFIER BREAKDOWN")
    lines.append("Index,Classifier,IDs with Classifier,Total Words,% of Total Corpus,Avg % within ID")
    
    for classifier in id_level_data['classifier_breakdown']:
        lines.append(f"{classifier['index']},{classifier['classifier']},{classifier['ids_with_classifier']},{classifier['total_words']},{classifier['percent_of_corpus']},{classifier['avg_percent_within_id']}")
    
    lines.append("")
    
    # Individual keyword results
    lines.append("INDIVIDUAL KEYWORD RESULTS")
    lines.append("Keyword,Found,Statement Matches")
    for keyword, data in results['keywords'].items():
        lines.append(f"{keyword},{'Yes' if data['found'] else 'No'},{data['count']}")
    
    return '\n'.join(lines)

def main():
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 CSV Keyword Search Tool</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #16a34a;">Upload a CSV file and search for keywords in the Statement column</p>', unsafe_allow_html=True)
    st.markdown('<p class="footer">Brought to you by Ricky Woznichak</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Default keywords management
        st.subheader("📝 Default Keywords")
        
        # Display current default keywords
        st.write("Current default keywords:")
        for i, keyword in enumerate(st.session_state.default_keywords):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(keyword)
            with col2:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.default_keywords.pop(i)
                    st.rerun()
        
        # Add new keyword
        new_keyword = st.text_input("Add new keyword:")
        if st.button("➕ Add Keyword") and new_keyword:
            if new_keyword not in st.session_state.default_keywords:
                st.session_state.default_keywords.append(new_keyword)
                st.rerun()
        
        # Reset to original defaults
        if st.button("🔄 Reset to Defaults"):
            st.session_state.default_keywords = [
                "innovation", "sustainability", "growth", "technology", 
                "efficiency", "collaboration", "strategy", "quality",
                "customer", "market", "digital", "transformation"
            ]
            st.rerun()
        
        st.divider()
        
        # Search options
        st.subheader("🔍 Search Options")
        case_sensitive = st.checkbox("Case sensitive", value=False)
        exact_match = st.checkbox("Exact word match only", value=False)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="File must contain 'Statement' and 'ID' columns"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                data = pd.read_csv(uploaded_file)
                
                # Validate required columns
                if 'Statement' not in data.columns:
                    st.error("❌ CSV file must contain a 'Statement' column")
                    return
                
                if 'ID' not in data.columns:
                    st.error("❌ CSV file must contain an 'ID' column for aggregation")
                    return
                
                # Filter out empty statements
                data = data.dropna(subset=['Statement'])
                data = data[data['Statement'].str.strip() != '']
                
                st.success(f"✅ File loaded: {len(data)} statements found")
                
                # Store data in session state
                st.session_state.data = data
                
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
                return
    
    with col2:
        st.subheader("🔍 Enter Keywords")
        
        # Option to use default keywords
        use_defaults = st.checkbox("Use default keywords", value=True)
        
        if use_defaults:
            keywords_text = ', '.join(st.session_state.default_keywords)
        else:
            keywords_text = ""
        
        keywords_input = st.text_area(
            "Keywords (comma-separated or one per line):",
            value=keywords_text,
            height=150,
            help="Enter keywords in any of these formats:\n- Comma-separated: innovation, sustainability, growth\n- With quotes: \"innovation\",\"sustainability\",\"growth\"\n- One per line"
        )
        
        # Search button
        if st.button("🔍 Search Keywords", type="primary", use_container_width=True):
            if 'data' not in st.session_state:
                st.error("❌ Please upload a CSV file first")
            elif not keywords_input.strip():
                st.error("❌ Please enter keywords to search")
            else:
                with st.spinner("🔍 Searching keywords..."):
                    keywords = parse_keywords(keywords_input)
                    
                    # Perform search
                    results = search_keywords(st.session_state.data, keywords, case_sensitive, exact_match)
                    id_level_data = calculate_id_level_data(st.session_state.data, keywords, case_sensitive, exact_match)
                    
                    # Store results
                    st.session_state.results = results
                    st.session_state.id_level_data = id_level_data
                
                st.success("✅ Search completed!")
    
    # Display results
    if st.session_state.results is not None:
        st.divider()
        
        # Results summary
        st.subheader("📊 Search Results Summary")
        
        # Download button
        if st.session_state.id_level_data is not None:
            csv_content = create_results_csv(st.session_state.results, st.session_state.id_level_data)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_content,
                file_name="keyword_search_results.csv",
                mime="text/csv"
            )
        
        # Stats cards
        results = st.session_state.results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <h3 style="margin: 0; color: #059669;">{results['stats']['found_keywords']}</h3>
                <p style="margin: 0; color: #047857;">Keywords Found</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <h3 style="margin: 0; color: #d97706;">{results['stats']['statements_with_matches']}</h3>
                <p style="margin: 0; color: #b45309;">Statements with Matches</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <h3 style="margin: 0; color: #059669;">{results['stats']['match_percentage']}%</h3>
                <p style="margin: 0; color: #047857;">Match Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <h3 style="margin: 0; color: #d97706;">{results['stats']['total_statements']}</h3>
                <p style="margin: 0; color: #b45309;">Total Statements</p>
            </div>
            """, unsafe_allow_html=True)
        
        # ID-Level Aggregated Data
        if st.session_state.id_level_data is not None:
            st.subheader("📈 ID-Level Aggregated Data")
            
            id_data = st.session_state.id_level_data
            
            # Aggregate stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total IDs", id_data['aggregate_stats']['total_ids'])
            with col2:
                st.metric("Total Words", id_data['aggregate_stats']['total_words'])
            with col3:
                st.metric("Avg Words/ID", id_data['aggregate_stats']['avg_words_per_id'])
            with col4:
                st.metric("Avg Statements/ID", id_data['aggregate_stats']['avg_statements_per_id'])
            
            # Classifier breakdown
            st.subheader("📋 ID-Level Classifier Breakdown")
            
            # Column definitions
            with st.expander("📖 Column Definitions"):
                st.write("**% of Total Corpus:** What percentage of all words in your dataset come from IDs containing this keyword. Higher percentages indicate keywords associated with longer, more detailed content.")
                st.write("**Avg % within ID:** On average, what percentage of an individual ID's content is made up of this keyword. Shows how frequently the keyword appears relative to other words within each ID.")
            
            # Create DataFrame for display
            classifier_df = pd.DataFrame(id_data['classifier_breakdown'])
            classifier_df = classifier_df[['index', 'classifier', 'ids_with_classifier', 'total_words', 'percent_of_corpus', 'avg_percent_within_id']]
            classifier_df.columns = ['Index', 'Classifier', 'IDs with Classifier', 'Total Words', '% of Total Corpus', 'Avg % within ID']
            
            st.dataframe(classifier_df, use_container_width=True)
        
        # Keyword Analysis
        st.subheader("🎯 Keyword Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**✅ Found Keywords**")
            found_keywords = {k: v for k, v in results['keywords'].items() if v['found']}
            
            if found_keywords:
                for keyword, data in found_keywords.items():
                    st.markdown(f"""
                    <div class="keyword-found">
                        <strong>{keyword}</strong><br>
                        <small>{data['count']} match{'es' if data['count'] != 1 else ''}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("*No keywords found*")
        
        with col2:
            st.write("**❌ Not Found Keywords**")
            not_found_keywords = {k: v for k, v in results['keywords'].items() if not v['found']}
            
            if not_found_keywords:
                for keyword, data in not_found_keywords.items():
                    st.markdown(f"""
                    <div class="keyword-not-found">
                        <strong>{keyword}</strong><br>
                        <small>{data['count']} match{'es' if data['count'] != 1 else ''}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("*All keywords found*")

if __name__ == "__main__":
    main()
