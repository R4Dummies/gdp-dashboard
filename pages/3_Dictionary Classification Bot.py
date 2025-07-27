import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import re
from datetime import datetime
import base64

# Set page configuration
st.set_page_config(
    page_title="Dictionary Classification Bot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #166534;
        margin-bottom: 2rem;
    }
    .step-container {
        border-left: 4px solid #16a34a;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: #f9fafb;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0fdf4;
        border: 2px solid #bbf7d0;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    .keyword-tag {
        background-color: #eab308;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
        display: inline-block;
    }
    .success-msg {
        background-color: #dcfce7;
        border: 1px solid #bbf7d0;
        color: #166534;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .error-msg {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        color: #dc2626;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'dictionary' not in st.session_state:
    st.session_state.dictionary = []
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = None
if 'keyword_analysis' not in st.session_state:
    st.session_state.keyword_analysis = None

def parse_csv_data(csv_text):
    """Parse CSV text and return DataFrame"""
    try:
        df = pd.read_csv(StringIO(csv_text))
        return df, None
    except Exception as e:
        return None, f"Error parsing CSV: {str(e)}"

def parse_dictionary(dict_text):
    """Parse dictionary text and return list of keywords"""
    if not dict_text.strip():
        return []
    
    # Try parsing quoted format first
    quoted_pattern = r'"([^"]+)"'
    keywords = re.findall(quoted_pattern, dict_text)
    
    if not keywords:
        # Try comma-separated format
        keywords = [k.strip() for k in dict_text.split(',') if k.strip()]
    
    return keywords

def classify_statements(df, text_column, ground_truth_column, dictionary):
    """Classify statements using dictionary"""
    results = []
    
    for idx, row in df.iterrows():
        statement = str(row[text_column]).lower() if pd.notna(row[text_column]) else ""
        
        # Find matched keywords
        matched_keywords = [kw for kw in dictionary if kw.lower() in statement]
        predicted = 1 if matched_keywords else 0
        
        # Get ground truth if available
        ground_truth = None
        category = None
        if ground_truth_column and ground_truth_column in df.columns:
            try:
                ground_truth = int(row[ground_truth_column])
                # Determine classification category
                if predicted == 1 and ground_truth == 1:
                    category = 'TP'
                elif predicted == 1 and ground_truth == 0:
                    category = 'FP'
                elif predicted == 0 and ground_truth == 1:
                    category = 'FN'
                else:
                    category = 'TN'
            except (ValueError, TypeError):
                ground_truth = None
        
        result = {
            'index': idx,
            'predicted': predicted,
            'ground_truth': ground_truth,
            'category': category,
            'matched_keywords': matched_keywords,
            'score': len(matched_keywords)
        }
        
        # Add original data
        for col in df.columns:
            result[col] = row[col]
        
        results.append(result)
    
    return pd.DataFrame(results)

def calculate_metrics(results_df):
    """Calculate classification metrics"""
    if 'category' not in results_df.columns or results_df['category'].isna().all():
        return None
    
    tp = len(results_df[results_df['category'] == 'TP'])
    fp = len(results_df[results_df['category'] == 'FP'])
    fn = len(results_df[results_df['category'] == 'FN'])
    tn = len(results_df[results_df['category'] == 'TN'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'accuracy': accuracy * 100,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn
    }

def analyze_keywords(df, text_column, ground_truth_column, dictionary):
    """Analyze individual keyword performance"""
    keyword_metrics = []
    
    for keyword in dictionary:
        tp_examples = []
        fp_examples = []
        total_positives = 0
        
        for idx, row in df.iterrows():
            statement = str(row[text_column]).lower() if pd.notna(row[text_column]) else ""
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
        
        recall = len(tp_examples) / total_positives if total_positives > 0 else 0
        precision = len(tp_examples) / (len(tp_examples) + len(fp_examples)) if (len(tp_examples) + len(fp_examples)) > 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
        
        keyword_metrics.append({
            'keyword': keyword,
            'true_positives_count': len(tp_examples),
            'false_positives_count': len(fp_examples),
            'recall': recall * 100,
            'precision': precision * 100,
            'f1_score': f1 * 100,
            'tp_examples': tp_examples[:3],
            'fp_examples': fp_examples[:3]
        })
    
    return keyword_metrics

def create_download_link(df, filename, link_text):
    """Create a download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Main app
def main():
    st.markdown('<h1 class="main-header">🔍 Dictionary Classification Bot</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #16a34a;">Enter keywords and classify statements to analyze their effectiveness</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Sample data option
        if st.button("Load Sample Data", type="secondary"):
            sample_csv = """ID,Statement,Answer
1,It's SPRING TRUNK SHOW week!,1
2,I am offering 4 shirts styled the way you want & the 5th is Also tossing in MAGNETIC COLLAR STAY to help keep your collars in place!,1
3,In recognition of Earth Day I would like to showcase our collection of Earth Fibers!,0
4,It is now time to do some wardrobe crunches and check your basics! Never on sale.,1
5,He's a hard worker and always willing to lend a hand. The prices are the best I've seen in 17 years of servicing my clients.,0"""
            st.session_state.csv_data, _ = parse_csv_data(sample_csv)
            st.success("Sample data loaded!")
    
    # Step 1: Input CSV Data
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("📊 Step 1: Input Sample Data (CSV Format)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader("Upload CSV File", type="csv")
        if uploaded_file is not None:
            csv_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            df, error = parse_csv_data(csv_text)
            if error:
                st.markdown(f'<div class="error-msg">❌ {error}</div>', unsafe_allow_html=True)
            else:
                st.session_state.csv_data = df
                st.markdown(f'<div class="success-msg">✅ Loaded {len(df)} rows</div>', unsafe_allow_html=True)
    
    with col2:
        # Manual input
        csv_input = st.text_area(
            "Or paste your CSV data here:",
            height=150,
            placeholder="ID,Statement,Answer\n1,Sample statement,1\n2,Another statement,0"
        )
        
        if csv_input.strip():
            df, error = parse_csv_data(csv_input)
            if error:
                st.markdown(f'<div class="error-msg">❌ {error}</div>', unsafe_allow_html=True)
            else:
                st.session_state.csv_data = df
                st.markdown(f'<div class="success-msg">✅ Loaded {len(df)} rows</div>', unsafe_allow_html=True)
    
    # Column selection
    if st.session_state.csv_data is not None:
        st.subheader("Column Selection")
        col1, col2 = st.columns(2)
        
        with col1:
            text_column = st.selectbox(
                "Text Column for Analysis",
                options=st.session_state.csv_data.columns.tolist(),
                index=0 if 'Statement' not in st.session_state.csv_data.columns else st.session_state.csv_data.columns.tolist().index('Statement')
            )
        
        with col2:
            ground_truth_column = st.selectbox(
                "Ground Truth Column (0/1 values) - Optional",
                options=['None'] + st.session_state.csv_data.columns.tolist(),
                index=0 if 'Answer' not in st.session_state.csv_data.columns else st.session_state.csv_data.columns.tolist().index('Answer') + 1
            )
            if ground_truth_column == 'None':
                ground_truth_column = None
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(st.session_state.csv_data.head(), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 2: Enter Dictionary
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("📝 Step 2: Enter Keyword Dictionary")
    
    # Default dictionary
    default_dict = '"spring","trunk show","customized","customization","earth day","wardrobe","sale","prices"'
    
    dict_input = st.text_area(
        "Enter keywords (format: \"word1\",\"word2\",\"word3\" or word1, word2, word3):",
        value=default_dict,
        height=100,
        help="You can use quoted format or simple comma-separated format"
    )
    
    if st.button("Save Dictionary", type="primary"):
        keywords = parse_dictionary(dict_input)
        if keywords:
            st.session_state.dictionary = keywords
            st.markdown(f'<div class="success-msg">✅ Dictionary saved with {len(keywords)} keywords</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-msg">❌ Please enter valid keywords</div>', unsafe_allow_html=True)
    
    # Display current dictionary
    if st.session_state.dictionary:
        st.subheader(f"Current Dictionary ({len(st.session_state.dictionary)} keywords)")
        keywords_html = ''.join([f'<span class="keyword-tag">{kw}</span>' for kw in st.session_state.dictionary])
        st.markdown(keywords_html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 3: Classify Statements
    if st.session_state.csv_data is not None and st.session_state.dictionary and 'text_column' in locals():
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.header("🎯 Step 3: Classify Statements")
        
        if st.button("Classify Statements", type="primary"):
            with st.spinner("Classifying statements..."):
                results_df = classify_statements(
                    st.session_state.csv_data, 
                    text_column, 
                    ground_truth_column, 
                    st.session_state.dictionary
                )
                st.session_state.classification_results = results_df
                st.success("Classification completed!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Classification Results
    if st.session_state.classification_results is not None:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.header("📈 Classification Results Summary")
        
        results_df = st.session_state.classification_results
        metrics = calculate_metrics(results_df)
        
        if metrics:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Accuracy</h3>
                    <h2>{metrics['accuracy']:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Precision</h3>
                    <h2>{metrics['precision']:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Recall</h3>
                    <h2>{metrics['recall']:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>F1 Score</h3>
                    <h2>{metrics['f1_score']:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Confusion Matrix Summary
            st.subheader("Confusion Matrix Summary")
            st.write(f"**True Positives:** {metrics['true_positives']} | "
                    f"**False Positives:** {metrics['false_positives']} | "
                    f"**False Negatives:** {metrics['false_negatives']} | "
                    f"**True Negatives:** {metrics['true_negatives']}")
            
            # Export buttons
            st.subheader("Export Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Export Full Results"):
                    st.download_button(
                        label="Download Classification Results CSV",
                        data=results_df.to_csv(index=False),
                        file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                metrics_df = pd.DataFrame([metrics])
                if st.button("📥 Export Metrics"):
                    st.download_button(
                        label="Download Metrics CSV",
                        data=metrics_df.to_csv(index=False),
                        file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            # False Positives and Negatives
            false_positives = results_df[results_df['category'] == 'FP']
            false_negatives = results_df[results_df['category'] == 'FN']
            
            if len(false_positives) > 0:
                st.subheader("🔴 False Positives (Incorrectly Classified as Positive)")
                with st.expander(f"View {len(false_positives)} false positives"):
                    for idx, row in false_positives.head(10).iterrows():
                        st.write(f"**Statement:** {row[text_column]}")
                        st.write(f"**Matched keywords:** {', '.join(row['matched_keywords'])}")
                        st.write("---")
            
            if len(false_negatives) > 0:
                st.subheader("🟡 False Negatives (Missed Positive Cases)")
                with st.expander(f"View {len(false_negatives)} false negatives"):
                    for idx, row in false_negatives.head(10).iterrows():
                        st.write(f"**Statement:** {row[text_column]}")
                        st.write("**No keywords matched**")
                        st.write("---")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 4: Keyword Analysis
    if (st.session_state.classification_results is not None and 
        'ground_truth_column' in locals() and ground_truth_column):
        
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.header("🔍 Step 4: Keyword Impact Analysis")
        st.write("Analyze keywords by different metrics to find the optimal set for your classification needs")
        
        if st.button("Analyze Keyword Impact", type="primary"):
            with st.spinner("Analyzing keywords..."):
                keyword_metrics = analyze_keywords(
                    st.session_state.csv_data,
                    text_column,
                    ground_truth_column,
                    st.session_state.dictionary
                )
                st.session_state.keyword_analysis = keyword_metrics
                st.success("Keyword analysis completed!")
        
        if st.session_state.keyword_analysis:
            metrics_data = st.session_state.keyword_analysis
            
            # Metric selection
            metric_option = st.selectbox(
                "Sort keywords by:",
                options=["Recall", "Precision", "F1 Score"],
                index=0
            )
            
            # Sort data based on selected metric
            if metric_option == "Recall":
                sorted_data = sorted(metrics_data, key=lambda x: x['recall'], reverse=True)
                metric_description = "**Recall:** Percentage of true positive cases captured. High recall means the keyword catches most relevant statements."
            elif metric_option == "Precision":
                sorted_data = sorted(metrics_data, key=lambda x: x['precision'], reverse=True)
                metric_description = "**Precision:** Percentage of predictions that are correct. High precision means the keyword rarely triggers on irrelevant statements."
            else:
                sorted_data = sorted(metrics_data, key=lambda x: x['f1_score'], reverse=True)
                metric_description = "**F1 Score:** Harmonic mean of precision and recall. Balances both metrics for overall effectiveness."
            
            st.markdown(f'<div class="success-msg">{metric_description}</div>', unsafe_allow_html=True)
            
            # Display top 10 keywords
            st.subheader(f"Top 10 Keywords by {metric_option}")
            
            for i, analysis in enumerate(sorted_data[:10]):
                with st.expander(f"#{i+1} - {analysis['keyword']} (Recall: {analysis['recall']:.1f}%, Precision: {analysis['precision']:.1f}%, F1: {analysis['f1_score']:.1f}%)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**True Positives ({analysis['true_positives_count']}):**")
                        if analysis['tp_examples']:
                            for example in analysis['tp_examples']:
                                st.write(f"• {example}")
                        else:
                            st.write("No examples")
                    
                    with col2:
                        st.write(f"**False Positives ({analysis['false_positives_count']}):**")
                        if analysis['fp_examples']:
                            for example in analysis['fp_examples']:
                                st.write(f"• {example}")
                        else:
                            st.write("No examples")
            
            # Export keyword analysis
            if st.button("📥 Export Keyword Analysis"):
                analysis_df = pd.DataFrame(sorted_data)
                st.download_button(
                    label="Download Keyword Analysis CSV",
                    data=analysis_df.to_csv(index=False),
                    file_name=f"keyword_analysis_{metric_option.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
