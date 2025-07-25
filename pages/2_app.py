import streamlit as st
import pandas as pd
import re
import io
from collections import defaultdict
import numpy as np

def parse_keywords(keywords_input):
    """Parse keywords from various input formats"""
    if not keywords_input.strip():
        return []
    
    if '"' in keywords_input:
        # Handle quoted keywords
        keywords = []
        for item in keywords_input.split(','):
            keyword = item.strip().strip('"').strip("'")
            if keyword:
                keywords.append(keyword)
        return keywords
    else:
        # Handle comma-separated or newline-separated keywords
        keywords = re.split(r'[,\n]', keywords_input)
        return [k.strip() for k in keywords if k.strip()]

def count_keyword_in_text(text, keyword, case_sensitive=False, exact_match=False):
    """Count occurrences of a keyword in text"""
    if pd.isna(text):
        text = ""
    
    search_text = text if case_sensitive else text.lower()
    search_term = keyword if case_sensitive else keyword.lower()
    
    if exact_match:
        words = re.findall(r'\b\w+\b', search_text)
        return words.count(search_term)
    else:
        # Escape special regex characters
        escaped_keyword = re.escape(search_term)
        pattern = re.compile(escaped_keyword, re.IGNORECASE if not case_sensitive else 0)
        matches = pattern.findall(search_text)
        return len(matches)

def calculate_id_level_data(df, keywords, case_sensitive=False, exact_match=False):
    """Calculate ID-level aggregated statistics"""
    # Group by ID
    id_groups = df.groupby('ID')
    
    total_ids = len(id_groups)
    total_words = 0
    
    id_stats = {}
    
    for id_val, group in id_groups:
        combined_text = ' '.join(group['Statement'].astype(str))
        word_count = len(re.findall(r'\b\w+\b', combined_text))
        total_words += word_count
        
        # Find which keywords are present in this ID
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
            'statement_count': len(group),
            'present_keywords': present_keywords,
            'combined_text': combined_text
        }
    
    avg_words_per_id = total_words / total_ids if total_ids > 0 else 0
    avg_statements_per_id = len(df) / total_ids if total_ids > 0 else 0
    
    # Calculate classifier breakdown
    classifier_breakdown = []
    for i, keyword in enumerate(keywords):
        ids_with_keyword = [stats for stats in id_stats.values() 
                          if keyword in stats['present_keywords']]
        
        total_words_in_keyword_ids = sum(stats['word_count'] for stats in ids_with_keyword)
        percent_of_corpus = (total_words_in_keyword_ids / total_words * 100) if total_words > 0 else 0
        
        # Calculate average percentage within ID
        if ids_with_keyword:
            percentages = []
            for stats in ids_with_keyword:
                keyword_matches = count_keyword_in_text(stats['combined_text'], keyword, 
                                                      case_sensitive, exact_match)
                if stats['word_count'] > 0:
                    percentages.append(keyword_matches / stats['word_count'] * 100)
                else:
                    percentages.append(0)
            avg_percent_within_id = np.mean(percentages)
        else:
            avg_percent_within_id = 0
        
        classifier_breakdown.append({
            'Index': i,
            'Classifier': keyword,
            'IDs with Classifier': len(ids_with_keyword),
            'Total Words': total_words_in_keyword_ids,
            '% of Total Corpus': f"{percent_of_corpus:.2f}%",
            'Avg % within ID': f"{avg_percent_within_id:.2f}%"
        })
    
    # Sort by number of IDs with classifier (descending)
    classifier_breakdown.sort(key=lambda x: x['IDs with Classifier'], reverse=True)
    
    return {
        'aggregate_stats': {
            'total_ids': total_ids,
            'total_words': total_words,
            'avg_words_per_id': f"{avg_words_per_id:.2f}",
            'avg_statements_per_id': f"{avg_statements_per_id:.2f}"
        },
        'classifier_breakdown': classifier_breakdown
    }

def search_keywords_in_dataframe(df, keywords, case_sensitive=False, exact_match=False):
    """Search for keywords in the dataframe"""
    results = {}
    matching_rows = []
    
    for keyword in keywords:
        matches = []
        for idx, row in df.iterrows():
            statement = str(row['Statement']) if pd.notna(row['Statement']) else ""
            search_text = statement if case_sensitive else statement.lower()
            search_term = keyword if case_sensitive else keyword.lower()
            
            found = False
            if exact_match:
                words = re.findall(r'\b\w+\b', search_text)
                found = search_term in words
            else:
                found = search_term in search_text
            
            if found:
                matches.append(idx)
        
        results[keyword] = {
            'found': len(matches) > 0,
            'count': len(matches),
            'matches': matches
        }
        
        # Add to matching rows
        for match_idx in matches:
            existing_row = next((r for r in matching_rows if r['index'] == match_idx), None)
            if existing_row:
                existing_row['matched_keywords'].append(keyword)
            else:
                matching_rows.append({
                    'index': match_idx,
                    'matched_keywords': [keyword]
                })
    
    return results, matching_rows

# Streamlit App
def main():
    st.set_page_config(
        page_title="CSV Keyword Search Tool",
        page_icon="🔍",
        layout="wide"
    )
    
    # Header
    st.title("🔍 CSV Keyword Search Tool")
    st.markdown("Upload a CSV file and search for keywords in the Statement column")
    st.markdown("**Brought to you by Ricky Woznichak**")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Default keywords dictionary (editable)
    st.sidebar.subheader("Default Keywords Dictionary")
    default_keywords_text = st.sidebar.text_area(
        "Edit default keywords (one per line or comma-separated):",
        value="innovation\nsustainability\ngrowth\ntechnology\ncollaboration",
        height=150,
        help="You can modify these default keywords and use them as a starting point"
    )
    
    if st.sidebar.button("Load Default Keywords"):
        st.session_state.keywords_input = default_keywords_text
    
    # Search options
    st.sidebar.subheader("Search Options")
    case_sensitive = st.sidebar.checkbox("Case sensitive", value=False)
    exact_match = st.sidebar.checkbox("Exact word match only", value=False)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="File must contain 'Statement' and 'ID' columns"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                if 'Statement' not in df.columns:
                    st.error("❌ CSV file must contain a 'Statement' column")
                    return
                
                if 'ID' not in df.columns:
                    st.error("❌ CSV file must contain an 'ID' column for aggregation")
                    return
                
                # Remove rows with empty statements
                df = df.dropna(subset=['Statement'])
                
                st.success(f"✅ File loaded: {len(df)} statements found")
                st.info(f"Columns: {', '.join(df.columns.tolist())}")
                
                # Store dataframe in session state
                st.session_state.df = df
                
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
                return
    
    with col2:
        st.header("🔍 Enter Keywords")
        
        # Initialize keywords input in session state if not exists
        if 'keywords_input' not in st.session_state:
            st.session_state.keywords_input = ""
        
        keywords_input = st.text_area(
            "Enter keywords in any of these formats:",
            value=st.session_state.keywords_input,
            height=150,
            help="""
            • Comma-separated: innovation, sustainability, growth
            • With quotes: "innovation","sustainability","growth"
            • One per line:
              innovation
              sustainability
              growth
            """,
            key="keywords_input"
        )
        
        search_button = st.button("🔍 Search Keywords", type="primary")
    
    # Search functionality
    if search_button:
        if 'df' not in st.session_state:
            st.error("❌ Please upload a CSV file first")
            return
        
        if not keywords_input.strip():
            st.error("❌ Please enter keywords to search")
            return
        
        df = st.session_state.df
        keywords = parse_keywords(keywords_input)
        
        if not keywords:
            st.error("❌ No valid keywords found")
            return
        
        with st.spinner("Searching keywords..."):
            # Perform search
            keyword_results, matching_rows = search_keywords_in_dataframe(
                df, keywords, case_sensitive, exact_match
            )
            
            # Calculate statistics
            total_statements = len(df)
            statements_with_matches = len(set(row['index'] for row in matching_rows))
            match_percentage = (statements_with_matches / total_statements * 100) if total_statements > 0 else 0
            
            # Calculate ID-level data
            id_level_data = calculate_id_level_data(df, keywords, case_sensitive, exact_match)
        
        # Display results
        st.header("📊 Search Results")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        found_keywords = sum(1 for result in keyword_results.values() if result['found'])
        
        with col1:
            st.metric("Keywords Found", found_keywords)
        with col2:
            st.metric("Statements with Matches", statements_with_matches)
        with col3:
            st.metric("Match Rate", f"{match_percentage:.1f}%")
        with col4:
            st.metric("Total Statements", total_statements)
        
        # ID-Level Aggregated Data
        st.subheader("📈 ID-Level Aggregated Data")
        
        agg_stats = id_level_data['aggregate_stats']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total IDs", agg_stats['total_ids'])
        with col2:
            st.metric("Total Words", f"{agg_stats['total_words']:,}")
        with col3:
            st.metric("Avg Words/ID", agg_stats['avg_words_per_id'])
        with col4:
            st.metric("Avg Statements/ID", agg_stats['avg_statements_per_id'])
        
        # Classifier Breakdown
        st.subheader("📋 ID-Level Classifier Breakdown")
        
        st.info("""
        **Column Definitions:**
        - **% of Total Corpus:** What percentage of all words in your dataset come from IDs containing this keyword
        - **Avg % within ID:** On average, what percentage of an individual ID's content is made up of this keyword
        """)
        
        classifier_df = pd.DataFrame(id_level_data['classifier_breakdown'])
        st.dataframe(classifier_df, use_container_width=True)
        
        # Individual keyword results
        st.subheader("🔍 Individual Keyword Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**✅ Found Keywords**")
            found_keywords_list = [(k, v) for k, v in keyword_results.items() if v['found']]
            if found_keywords_list:
                for keyword, data in found_keywords_list:
                    st.write(f"• **{keyword}**: {data['count']} matches")
            else:
                st.write("*No keywords found*")
        
        with col2:
            st.write("**❌ Not Found Keywords**")
            not_found_keywords = [(k, v) for k, v in keyword_results.items() if not v['found']]
            if not_found_keywords:
                for keyword, data in not_found_keywords:
                    st.write(f"• **{keyword}**: {data['count']} matches")
            else:
                st.write("*All keywords found*")
        
        # Export results
        st.subheader("📥 Export Results")
        
        # Create export data
        export_data = []
        
        # Summary
        export_data.append(["SEARCH RESULTS SUMMARY"])
        export_data.append(["Keywords Found", found_keywords])
        export_data.append(["Statements with Matches", statements_with_matches])
        export_data.append(["Match Rate", f"{match_percentage:.1f}%"])
        export_data.append(["Total Statements", total_statements])
        export_data.append([""])
        
        # ID-Level data
        export_data.append(["ID-LEVEL AGGREGATED DATA"])
        export_data.append(["Total IDs", agg_stats['total_ids']])
        export_data.append(["Total Words", agg_stats['total_words']])
        export_data.append(["Avg Words/ID", agg_stats['avg_words_per_id']])
        export_data.append(["Avg Statements/ID", agg_stats['avg_statements_per_id']])
        export_data.append([""])
        
        # Classifier breakdown
        export_data.append(["ID-LEVEL CLASSIFIER BREAKDOWN"])
        export_data.append(["Index", "Classifier", "IDs with Classifier", "Total Words", "% of Total Corpus", "Avg % within ID"])
        
        for item in id_level_data['classifier_breakdown']:
            export_data.append([
                item['Index'],
                item['Classifier'],
                item['IDs with Classifier'],
                item['Total Words'],
                item['% of Total Corpus'],
                item['Avg % within ID']
            ])
        
        export_data.append([""])
        
        # Individual keyword results
        export_data.append(["INDIVIDUAL KEYWORD RESULTS"])
        export_data.append(["Keyword", "Found", "Statement Matches"])
        
        for keyword, data in keyword_results.items():
            export_data.append([
                keyword,
                "Yes" if data['found'] else "No",
                data['count']
            ])
        
        # Convert to CSV string
        export_df = pd.DataFrame(export_data)
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False, header=False)
        csv_string = csv_buffer.getvalue()
        
        st.download_button(
            label="💾 Download Results as CSV",
            data=csv_string,
            file_name="keyword_search_results.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
