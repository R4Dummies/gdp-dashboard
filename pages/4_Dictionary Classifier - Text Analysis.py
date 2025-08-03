import streamlit as st
import pandas as pd
import json
import re
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="Dictionary Classifier",
    page_icon="🧠",
    layout="wide"
)

def parse_dictionary(dict_input):
    """Parse dictionary input from JSON or line format"""
    dict_text = dict_input.strip()
    categories = {}
    
    if not dict_text:
        return categories
    
    # Try JSON first
    try:
        parsed = json.loads(dict_text)
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                if isinstance(v, str):
                    v = [v]
                elif isinstance(v, list):
                    v = [str(x) for x in v]
                else:
                    continue
                categories[k] = [term.strip().lower() for term in v if term.strip()]
        return categories
    except json.JSONDecodeError:
        pass
    
    # Parse line format
    for line in dict_text.splitlines():
        line = line.strip()
        if ':' in line:
            cat, terms = line.split(':', 1)
            cat = cat.strip()
            if cat:
                term_list = [w.strip().lower() for w in terms.split(',') if w.strip()]
                if term_list:
                    categories[cat] = term_list
    
    return categories

def classify_text(text, categories, method="count"):
    """Classify text based on keyword matches"""
    if pd.isna(text):
        return None, 0
    
    text_lower = str(text).lower()
    
    best_category = None
    max_score = 0
    
    for category, keywords in categories.items():
        if method == "count":
            # Count total keyword occurrences
            score = sum(text_lower.count(keyword) for keyword in keywords)
        elif method == "unique":
            # Count unique keywords found
            score = sum(1 for keyword in keywords if keyword in text_lower)
        elif method == "weighted":
            # Weighted by keyword length (longer keywords get more weight)
            score = sum(text_lower.count(keyword) * len(keyword.split()) 
                       for keyword in keywords)
        
        if score > max_score:
            best_category = category
            max_score = score
    
    return best_category, max_score

def get_classification_stats(df, text_col, categories):
    """Get statistics about the classification"""
    stats = {}
    for category in categories.keys():
        stats[category] = 0
    stats['Unclassified'] = 0
    
    for _, row in df.iterrows():
        pred_cat = row.get('Predicted_Category')
        if pred_cat:
            stats[pred_cat] += 1
        else:
            stats['Unclassified'] += 1
    
    return stats

# Main app
st.title("🧠 Dictionary-Based Text Classifier")

st.markdown("""
This app classifies text based on keyword dictionaries. Upload your CSV dataset, 
define your categories and keywords, then analyze your text data!

### How to use:
1. **Upload** your CSV file
2. **Select** the text column to analyze
3. **Define** your dictionary (JSON format or simple lines)
4. **Choose** classification method
5. **Run** the analysis and download results
""")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    classification_method = st.selectbox(
        "Classification Method:",
        ["count", "unique", "weighted"],
        help="""
        - **Count**: Total keyword occurrences
        - **Unique**: Number of unique keywords found
        - **Weighted**: Weighted by keyword phrase length
        """
    )
    
    min_score_threshold = st.number_input(
        "Minimum Score Threshold:",
        min_value=0,
        value=1,
        help="Minimum score needed for classification (0 = classify all)"
    )

# File upload section
st.header("📁 Upload Data")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload a CSV file containing text data to classify"
)

df_loaded = None
if uploaded_file:
    try:
        # Load the CSV
        df_loaded = pd.read_csv(uploaded_file)
        
        st.success(f"✅ CSV uploaded successfully! Shape: {df_loaded.shape}")
        
        # Show preview
        with st.expander("📊 Data Preview"):
            st.dataframe(df_loaded.head())
            
        # Column selection
        text_columns = df_loaded.select_dtypes(include=['object']).columns.tolist()
        if not text_columns:
            st.error("❌ No text columns found in the dataset!")
            st.stop()
            
        text_col = st.selectbox(
            "📝 Select Text Column:",
            text_columns,
            help="Choose the column containing text to classify"
        )
        
        # Show sample text
        if text_col:
            st.info(f"**Sample text from '{text_col}':**")
            sample_texts = df_loaded[text_col].dropna().head(3)
            for i, text in enumerate(sample_texts, 1):
                st.write(f"{i}. {str(text)[:200]}{'...' if len(str(text)) > 200 else ''}")
                
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        st.stop()

# Dictionary definition section
st.header("📚 Define Classification Dictionary")

col1, col2 = st.columns([2, 1])

with col1:
    # Default dictionary
    default_dict = """{
    "Luxury": ["elegant", "timeless", "refined", "classic", "sophisticated", "luxury", "polished", "premium"],
    "Technology": ["digital", "AI", "machine learning", "software", "tech", "innovation", "algorithm"],
    "Sustainability": ["eco-friendly", "sustainable", "green", "environmental", "renewable", "carbon neutral"]
}"""
    
    dict_input = st.text_area(
        "Dictionary Definition:",
        value=default_dict,
        height=200,
        help="""
        Enter your dictionary in JSON format or simple lines like:
        
        **JSON Format:**
        ```json
        {
            "Category1": ["keyword1", "keyword2"],
            "Category2": ["keyword3", "keyword4"]
        }
        ```
        
        **Simple Format:**
        ```
        Category1: keyword1, keyword2, keyword3
        Category2: keyword4, keyword5, keyword6
        ```
        """
    )

with col2:
    st.markdown("### 💡 Tips:")
    st.markdown("""
    - Use specific, relevant keywords
    - Include variations and synonyms
    - Consider plural/singular forms
    - Test with sample data first
    - Keywords are case-insensitive
    """)

# Parse and validate dictionary
if dict_input.strip():
    categories = parse_dictionary(dict_input)
    
    if categories:
        st.success(f"✅ Dictionary parsed successfully! Found {len(categories)} categories:")
        
        # Display parsed dictionary
        with st.expander("🔍 View Parsed Dictionary"):
            for cat, keywords in categories.items():
                st.write(f"**{cat}:** {', '.join(keywords)}")
    else:
        st.error("❌ Invalid dictionary format. Please check your input.")
        st.stop()
else:
    st.warning("⚠️ Please define your classification dictionary.")
    st.stop()

# Analysis section
if df_loaded is not None and categories:
    st.header("🔍 Run Analysis")
    
    if st.button("🚀 Start Classification", type="primary"):
        with st.spinner("Classifying texts..."):
            results = []
            scores = []
            
            # Progress bar
            progress_bar = st.progress(0)
            total_rows = len(df_loaded)
            
            for idx, text in enumerate(df_loaded[text_col]):
                category, score = classify_text(text, categories, classification_method)
                
                # Apply threshold
                if score < min_score_threshold:
                    category = None
                    
                results.append(category)
                scores.append(score)
                
                # Update progress
                progress_bar.progress((idx + 1) / total_rows)
            
            # Create result dataframe
            result_df = df_loaded.copy()
            result_df['Predicted_Category'] = results
            result_df['Classification_Score'] = scores
            
            st.success("✅ Classification complete!")
            
            # Show statistics
            stats = get_classification_stats(result_df, text_col, categories)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Classification Statistics")
                stats_df = pd.DataFrame(
                    list(stats.items()), 
                    columns=['Category', 'Count']
                )
                st.dataframe(stats_df)
                
            with col2:
                st.subheader("📈 Distribution")
                st.bar_chart(stats_df.set_index('Category'))
            
            # Show sample results
            st.subheader("📋 Sample Results")
            
            # Filter options
            filter_col1, filter_col2 = st.columns([1, 1])
            
            with filter_col1:
                show_category = st.selectbox(
                    "Filter by Category:",
                    ['All'] + list(categories.keys()) + ['Unclassified']
                )
                
            with filter_col2:
                min_display_score = st.number_input(
                    "Minimum Score to Display:",
                    min_value=0,
                    value=0
                )
            
            # Apply filters
            display_df = result_df.copy()
            
            if show_category != 'All':
                if show_category == 'Unclassified':
                    display_df = display_df[display_df['Predicted_Category'].isna()]
                else:
                    display_df = display_df[display_df['Predicted_Category'] == show_category]
                    
            display_df = display_df[display_df['Classification_Score'] >= min_display_score]
            
            # Show results
            st.dataframe(
                display_df[[text_col, 'Predicted_Category', 'Classification_Score']].head(20),
                use_container_width=True
            )
            
            if len(display_df) > 20:
                st.info(f"Showing first 20 of {len(display_df)} matching records.")
            
            # Download section
            st.subheader("💾 Download Results")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                # Full results
                csv_full = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download Full Results",
                    data=csv_full,
                    file_name='classification_results_full.csv',
                    mime='text/csv'
                )
                
            with col2:
                # Summary only
                summary_df = result_df[[text_col, 'Predicted_Category', 'Classification_Score']]
                csv_summary = summary_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📋 Download Summary",
                    data=csv_summary,
                    file_name='classification_summary.csv',
                    mime='text/csv'
                )
                
            with col3:
                # Statistics
                csv_stats = stats_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📈 Download Statistics",
                    data=csv_stats,
                    file_name='classification_stats.csv',
                    mime='text/csv'
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    Built with Streamlit • Dictionary-Based Text Classification
</div>
""", unsafe_allow_html=True)
