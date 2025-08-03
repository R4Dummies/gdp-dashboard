import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import re
from collections import defaultdict

# Set page config
st.set_page_config(
    page_title="Dictionary Classification Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'dictionary' not in st.session_state:
    st.session_state.dictionary = []
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = None
if 'keyword_analysis' not in st.session_state:
    st.session_state.keyword_analysis = None

def parse_dictionary_text(text):
    """Parse dictionary text in various formats"""
    keywords = []
    
    # Try parsing quoted format first
    quoted_matches = re.findall(r'"([^"]+)"', text)
    if quoted_matches:
        keywords = [k.strip() for k in quoted_matches if k.strip()]
    else:
        # Try comma-separated format
        keywords = [k.strip() for k in text.split(',') if k.strip()]
    
    return keywords

def classify_statements(df, text_column, dictionary, ground_truth_column=None):
    """Classify statements using dictionary keywords"""
    results = []
    
    # Initialize confusion matrix counters
    tp = fp = fn = tn = 0
    
    for idx, row in df.iterrows():
        statement = str(row[text_column]).lower()
        
        # Find matched keywords
        matched_keywords = [kw for kw in dictionary if kw.lower() in statement]
        predicted = 1 if matched_keywords else 0
        
        # Calculate confusion matrix if ground truth is available
        category = None
        ground_truth = None
        
        if ground_truth_column and ground_truth_column in df.columns:
            try:
                ground_truth = int(row[ground_truth_column])
                if predicted == 1 and ground_truth == 1:
                    tp += 1
                    category = 'TP'
                elif predicted == 1 and ground_truth == 0:
                    fp += 1
                    category = 'FP'
                elif predicted == 0 and ground_truth == 1:
                    fn += 1
                    category = 'FN'
                else:
                    tn += 1
                    category = 'TN'
            except (ValueError, TypeError):
                pass
        
        results.append({
            'index': idx,
            'statement': row[text_column],
            'predicted': predicted,
            'ground_truth': ground_truth,
            'category': category,
            'matched_keywords': matched_keywords,
            'score': len(matched_keywords)
        })
    
    # Calculate metrics
    metrics = None
    if ground_truth_column and ground_truth_column in df.columns:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        metrics = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'accuracy': accuracy * 100,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    return results, metrics

def analyze_keywords(df, text_column, ground_truth_column, dictionary):
    """Analyze individual keyword performance"""
    keyword_metrics = []
    
    for keyword in dictionary:
        tp_examples = []
        fp_examples = []
        total_positives = 0
        
        for idx, row in df.iterrows():
            statement = str(row[text_column]).lower()
            contains_keyword = keyword.lower() in statement
            
            try:
                ground_truth = int(row[ground_truth_column])
                
                if ground_truth == 1:
                    total_positives += 1
                    if contains_keyword:
                        tp_examples.append(row[text_column])
                elif ground_truth == 0 and contains_keyword:
                    fp_examples.append(row[text_column])
            except (ValueError, TypeError):
                continue
        
        # Calculate metrics
        tp_count = len(tp_examples)
        fp_count = len(fp_examples)
        
        recall = tp_count / total_positives if total_positives > 0 else 0
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
        
        keyword_metrics.append({
            'keyword': keyword,
            'tp_count': tp_count,
            'fp_count': fp_count,
            'recall': recall * 100,
            'precision': precision * 100,
            'f1_score': f1 * 100,
            'tp_examples': tp_examples[:3],  # Top 3 examples
            'fp_examples': fp_examples[:3]   # Top 3 examples
        })
    
    return keyword_metrics

# Main UI
st.title("📚 Dictionary Classification Bot")
st.markdown("Enter keywords and classify statements to analyze their effectiveness")

# Sidebar for dictionary management
with st.sidebar:
    st.header("🔧 Dictionary Management")
    
    # Dictionary input
    st.subheader("Keyword Dictionary")
    dictionary_text = st.text_area(
        "Enter keywords (comma-separated or quoted):",
        value='"keyword1","keyword2","keyword3"' if not st.session_state.dictionary else ','.join(f'"{kw}"' for kw in st.session_state.dictionary),
        height=100,
        help='Format: "custom","customized","customization" or simply: custom, customized, customization'
    )
    
    if st.button("Save Dictionary"):
        keywords = parse_dictionary_text(dictionary_text)
        if keywords:
            st.session_state.dictionary = keywords
            st.success(f"Saved {len(keywords)} keywords!")
        else:
            st.error("Please enter valid keywords")
    
    # Display current dictionary
    if st.session_state.dictionary:
        st.subheader("Current Dictionary")
        st.write(f"**{len(st.session_state.dictionary)} keywords:**")
        for i, kw in enumerate(st.session_state.dictionary, 1):
            st.write(f"{i}. {kw}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Data Input")
    
    # Sample data option
    if st.button("Load Sample Data"):
        sample_data = """ID,Statement,Answer
1,It's SPRING TRUNK SHOW week!,1
2,I am offering 4 shirts styled the way you want & the 5th is free!,1
3,In recognition of Earth Day I would like to showcase our collection of Earth Fibers!,0
4,It is now time to do some wardrobe crunches and check your basics! Never on sale.,1
5,He's a hard worker and always willing to lend a hand. The prices are the best I've seen.,0"""
        st.session_state.sample_data = sample_data
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    # Text area for manual input
    csv_text = st.text_area(
        "Or paste CSV data here:",
        value=getattr(st.session_state, 'sample_data', ''),
        height=200,
        help="Paste your CSV data with headers"
    )
    
    # Process data
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} rows from uploaded file")
    elif csv_text.strip():
        try:
            df = pd.read_csv(StringIO(csv_text))
            st.success(f"Loaded {len(df)} rows from text input")
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")
    
    if df is not None:
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Column selection
        st.subheader("Column Selection")
        col_sel1, col_sel2 = st.columns(2)
        
        with col_sel1:
            text_column = st.selectbox(
                "Text Column for Analysis:",
                options=df.columns.tolist(),
                index=df.columns.tolist().index('Statement') if 'Statement' in df.columns else 0
            )
        
        with col_sel2:
            ground_truth_column = st.selectbox(
                "Ground Truth Column (0/1 values):",
                options=['None'] + df.columns.tolist(),
                index=df.columns.tolist().index('Answer') + 1 if 'Answer' in df.columns else 0
            )
            if ground_truth_column == 'None':
                ground_truth_column = None

with col2:
    st.header("🎯 Actions")
    
    # Classification button
    if st.button("🚀 Classify Statements", type="primary", disabled=not (df is not None and st.session_state.dictionary and text_column)):
        if df is not None and st.session_state.dictionary and text_column:
            with st.spinner("Classifying statements..."):
                results, metrics = classify_statements(df, text_column, st.session_state.dictionary, ground_truth_column)
                st.session_state.classification_results = results
                st.session_state.metrics = metrics
            st.success("Classification completed!")
        else:
            st.error("Please upload data, set dictionary, and select text column")
    
    # Keyword analysis button
    if st.button("📈 Analyze Keywords", disabled=not (st.session_state.classification_results and ground_truth_column)):
        if st.session_state.classification_results and ground_truth_column:
            with st.spinner("Analyzing keywords..."):
                keyword_analysis = analyze_keywords(df, text_column, ground_truth_column, st.session_state.dictionary)
                st.session_state.keyword_analysis = keyword_analysis
            st.success("Keyword analysis completed!")
        else:
            st.error("Please classify statements first with ground truth column")

# Results section
if st.session_state.classification_results:
    st.header("📊 Classification Results")
    
    # Metrics display
    if hasattr(st.session_state, 'metrics') and st.session_state.metrics:
        metrics = st.session_state.metrics
        
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        with met_col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.1f}%")
        with met_col2:
            st.metric("Precision", f"{metrics['precision']:.1f}%")
        with met_col3:
            st.metric("Recall", f"{metrics['recall']:.1f}%")
        with met_col4:
            st.metric("F1 Score", f"{metrics['f1_score']:.1f}%")
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        st.write(f"**True Positives:** {metrics['tp']} | **False Positives:** {metrics['fp']} | "
                f"**False Negatives:** {metrics['fn']} | **True Negatives:** {metrics['tn']}")
    
    # Results dataframe
    results_df = pd.DataFrame(st.session_state.classification_results)
    
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        show_category = st.selectbox("Filter by category:", ['All', 'TP', 'FP', 'FN', 'TN'])
    with filter_col2:
        show_predicted = st.selectbox("Filter by prediction:", ['All', 'Positive (1)', 'Negative (0)'])
    
    # Apply filters
    filtered_df = results_df.copy()
    if show_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == show_category]
    if show_predicted == 'Positive (1)':
        filtered_df = filtered_df[filtered_df['predicted'] == 1]
    elif show_predicted == 'Negative (0)':
        filtered_df = filtered_df[filtered_df['predicted'] == 0]
    
    st.subheader("Detailed Results")
    st.dataframe(filtered_df[['statement', 'predicted', 'ground_truth', 'category', 'matched_keywords', 'score']])
    
    # Download results
    csv_results = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_results,
        file_name="classification_results.csv",
        mime="text/csv"
    )

# Keyword analysis section
if st.session_state.keyword_analysis:
    st.header("🔍 Keyword Analysis")
    
    # Metric selection
    analysis_metric = st.selectbox("Sort by metric:", ['Recall', 'Precision', 'F1 Score'])
    
    # Sort keywords by selected metric
    metric_key = analysis_metric.lower().replace(' ', '_')
    sorted_keywords = sorted(st.session_state.keyword_analysis, 
                           key=lambda x: x[metric_key], reverse=True)
    
    # Display top 10 keywords
    st.subheader(f"Top Keywords by {analysis_metric}")
    
    for i, kw_data in enumerate(sorted_keywords[:10], 1):
        with st.expander(f"#{i} - {kw_data['keyword']} ({kw_data[metric_key]:.1f}% {analysis_metric})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Recall", f"{kw_data['recall']:.1f}%")
            with col2:
                st.metric("Precision", f"{kw_data['precision']:.1f}%")
            with col3:
                st.metric("F1 Score", f"{kw_data['f1_score']:.1f}%")
            
            # Examples
            ex_col1, ex_col2 = st.columns(2)
            
            with ex_col1:
                st.write("**True Positives** (sample):")
                for example in kw_data['tp_examples']:
                    st.write(f"• {example[:100]}{'...' if len(example) > 100 else ''}")
                if not kw_data['tp_examples']:
                    st.write("No examples")
            
            with ex_col2:
                st.write("**False Positives** (sample):")
                for example in kw_data['fp_examples']:
                    st.write(f"• {example[:100]}{'...' if len(example) > 100 else ''}")
                if not kw_data['fp_examples']:
                    st.write("No examples")
    
    # Download keyword analysis
    keyword_df = pd.DataFrame(st.session_state.keyword_analysis)
    keyword_csv = keyword_df[['keyword', 'recall', 'precision', 'f1_score', 'tp_count', 'fp_count']].to_csv(index=False)
    st.download_button(
        label="📥 Download Keyword Analysis as CSV",
        data=keyword_csv,
        file_name="keyword_analysis.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Upload your dataset and start classifying!")
