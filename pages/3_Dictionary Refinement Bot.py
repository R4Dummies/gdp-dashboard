import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import json
from typing import Dict, List, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="Dictionary Classification Bot",
    page_icon="🤖",
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
    .step-header {
        color: #166534;
        border-left: 4px solid #22c55e;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0fdf4;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #bbf7d0;
        text-align: center;
    }
    .keyword-badge {
        background-color: #eab308;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
        display: inline-block;
    }
    .false-positive {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .false-negative {
        background-color: #fffbeb;
        border: 1px solid #fed7aa;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = None
    if 'dictionary' not in st.session_state:
        st.session_state.dictionary = []
    if 'classification_results' not in st.session_state:
        st.session_state.classification_results = None
    if 'keyword_analysis' not in st.session_state:
        st.session_state.keyword_analysis = None

def load_sample_data():
    """Load sample CSV data"""
    sample_csv = """ID,Statement,Answer
1,It's SPRING TRUNK SHOW week!,1
2,I am offering 4 shirts styled the way you want & the 5th is Also tossing in MAGNETIC COLLAR STAY to help keep your collars in place!,1
3,In recognition of Earth Day I would like to showcase our collection of Earth Fibers!,0
4,It is now time to do some wardrobe crunches and check your basics! Never on sale.,1
5,He's a hard worker and always willing to lend a hand. The prices are the best I've seen in 17 years of servicing my clients.,0"""
    
    return pd.read_csv(StringIO(sample_csv))

def parse_dictionary_text(dictionary_text: str) -> List[str]:
    """Parse dictionary text into list of keywords"""
    if not dictionary_text.strip():
        return []
    
    # Try parsing quoted format first
    import re
    quoted_keywords = re.findall(r'"([^"]+)"', dictionary_text)
    if quoted_keywords:
        return [k.strip() for k in quoted_keywords if k.strip()]
    
    # Fallback to comma-separated
    keywords = [k.strip() for k in dictionary_text.split(',') if k.strip()]
    return keywords

def classify_statements(df: pd.DataFrame, text_column: str, dictionary: List[str], 
                       ground_truth_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """Classify statements using dictionary and calculate metrics"""
    results = []
    
    # Classification counters
    tp = fp = fn = tn = 0
    
    for _, row in df.iterrows():
        statement = str(row[text_column]).lower()
        
        # Find matching keywords
        matched_keywords = [kw for kw in dictionary if kw.lower() in statement]
        predicted = 1 if matched_keywords else 0
        
        # Calculate confusion matrix if ground truth available
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
                ground_truth = None
        
        results.append({
            **row.to_dict(),
            'predicted': predicted,
            'ground_truth': ground_truth,
            'category': category,
            'matched_keywords': matched_keywords,
            'score': len(matched_keywords)
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate overall metrics
    metrics = {}
    if ground_truth_column and tp + fp + fn + tn > 0:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        metrics = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'accuracy': accuracy * 100,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        }
    
    return results_df, metrics

def analyze_keywords(df: pd.DataFrame, text_column: str, ground_truth_column: str, 
                    dictionary: List[str]) -> Dict:
    """Analyze individual keyword performance"""
    keyword_metrics = []
    
    for keyword in dictionary:
        tp_examples = []
        fp_examples = []
        total_positives = 0
        
        for _, row in df.iterrows():
            statement = str(row[text_column]).lower()
            contains_keyword = keyword.lower() in statement
            ground_truth = int(row[ground_truth_column])
            
            if ground_truth == 1:
                total_positives += 1
                if contains_keyword:
                    tp_examples.append(row)
            elif ground_truth == 0 and contains_keyword:
                fp_examples.append(row)
        
        tp_count = len(tp_examples)
        fp_count = len(fp_examples)
        
        recall = tp_count / total_positives if total_positives > 0 else 0
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
        
        keyword_metrics.append({
            'keyword': keyword,
            'true_positives_count': tp_count,
            'false_positives_count': fp_count,
            'recall': recall * 100,
            'precision': precision * 100,
            'f1_score': f1 * 100,
            'tp_examples': tp_examples[:3],
            'fp_examples': fp_examples[:3]
        })
    
    # Sort by different metrics
    by_recall = sorted(keyword_metrics, key=lambda x: x['recall'], reverse=True)[:10]
    by_precision = sorted(keyword_metrics, key=lambda x: x['precision'], reverse=True)[:10]
    by_f1 = sorted(keyword_metrics, key=lambda x: x['f1_score'], reverse=True)[:10]
    
    return {
        'by_recall': by_recall,
        'by_precision': by_precision,
        'by_f1': by_f1
    }

def render_keyword_analysis_table(keywords: List[Dict], text_column: str, metric_name: str):
    """Render keyword analysis results"""
    st.markdown(f"### Top Keywords by {metric_name.title()}")
    
    # Add download button for this specific metric
    if keywords:
        # Create CSV data for this metric
        csv_data = []
        for i, analysis in enumerate(keywords):
            csv_data.append({
                'Rank': i + 1,
                'Keyword': analysis['keyword'],
                'Recall_%': round(analysis['recall'], 2),
                'Precision_%': round(analysis['precision'], 2),
                'F1_Score_%': round(analysis['f1_score'], 2),
                'True_Positives_Count': analysis['true_positives_count'],
                'False_Positives_Count': analysis['false_positives_count']
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_output = csv_df.to_csv(index=False)
        
        st.download_button(
            label=f"📥 Download {metric_name.title()} Analysis CSV",
            data=csv_output,
            file_name=f"keyword_analysis_{metric_name}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key=f"download_{metric_name}"
        )
    
    for i, analysis in enumerate(keywords):
        with st.expander(f"#{i+1} - {analysis['keyword']} ({analysis[metric_name]:.1f}%)"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Recall", f"{analysis['recall']:.1f}%")
            with col2:
                st.metric("Precision", f"{analysis['precision']:.1f}%")
            with col3:
                st.metric("F1 Score", f"{analysis['f1_score']:.1f}%")
            with col4:
                st.metric("True Positives", analysis['true_positives_count'])
            
            # Examples
            col_tp, col_fp = st.columns(2)
            
            with col_tp:
                st.markdown("**True Positive Examples:**")
                if analysis['tp_examples']:
                    for example in analysis['tp_examples']:
                        st.markdown(f"<div class='false-positive'>{example[text_column]}</div>", 
                                  unsafe_allow_html=True)
                else:
                    st.write("No examples")
            
            with col_fp:
                st.markdown("**False Positive Examples:**")
                if analysis['fp_examples']:
                    for example in analysis['fp_examples']:
                        st.markdown(f"<div class='false-negative'>{example[text_column]}</div>", 
                                  unsafe_allow_html=True)
                else:
                    st.write("No examples")

def main():
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown("<h1 class='main-header'>🤖 Dictionary Classification Bot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #166534;'>Enter keywords and classify statements to analyze their effectiveness</p>", unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Default dictionary
        st.subheader("Default Dictionary")
        default_dict_text = st.text_area(
            "Default Keywords (comma-separated or quoted)",
            value='"spring","trunk show","sale","offer","custom","style"',
            height=100,
            help="Enter keywords in format: keyword1,keyword2 or \"keyword1\",\"keyword2\""
        )
        
        if st.button("💾 Save Default Dictionary"):
            st.session_state.dictionary = parse_dictionary_text(default_dict_text)
            st.success(f"Saved {len(st.session_state.dictionary)} keywords")
    
    # Step 1: Input CSV Data
    st.markdown("<h2 class='step-header'>📁 Step 1: Input Sample Data (CSV Format)</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                st.session_state.csv_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(st.session_state.csv_data)} rows")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
        
        # Manual CSV input
        csv_text = st.text_area(
            "Or paste CSV data here:",
            height=150,
            placeholder="ID,Statement,Answer\n1,Sample statement,1\n2,Another statement,0"
        )
        
        if csv_text.strip():
            try:
                st.session_state.csv_data = pd.read_csv(StringIO(csv_text))
                st.success(f"✅ Loaded {len(st.session_state.csv_data)} rows from text input")
            except Exception as e:
                st.error(f"Error parsing CSV: {str(e)}")
    
    with col2:
        st.markdown("**Sample Data**")
        if st.button("📋 Load Sample Data"):
            st.session_state.csv_data = load_sample_data()
            st.success("✅ Sample data loaded")
    
    # Column selection
    if st.session_state.csv_data is not None:
        st.subheader("Column Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            text_column = st.selectbox(
                "Text Column for Analysis",
                options=st.session_state.csv_data.columns.tolist(),
                index=0 if 'Statement' not in st.session_state.csv_data.columns 
                      else st.session_state.csv_data.columns.tolist().index('Statement')
            )
        
        with col2:
            ground_truth_options = ['None'] + st.session_state.csv_data.columns.tolist()
            ground_truth_column = st.selectbox(
                "Ground Truth Column (0/1 values)",
                options=ground_truth_options,
                index=0 if 'Answer' not in st.session_state.csv_data.columns 
                      else ground_truth_options.index('Answer')
            )
            ground_truth_column = None if ground_truth_column == 'None' else ground_truth_column
        
        # Preview data
        st.subheader("Data Preview")
        st.dataframe(st.session_state.csv_data.head())
    
    # Step 2: Dictionary Management
    st.markdown("<h2 class='step-header'>📝 Step 2: Manage Keyword Dictionary</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Current dictionary display
        if st.session_state.dictionary:
            st.markdown("**Current Dictionary:**")
            keywords_html = " ".join([f"<span class='keyword-badge'>{kw}</span>" for kw in st.session_state.dictionary])
            st.markdown(f"<div>{keywords_html}</div>", unsafe_allow_html=True)
            st.write(f"Total keywords: {len(st.session_state.dictionary)}")
        
        # Dictionary input
        dict_input = st.text_area(
            "Enter Keywords",
            value='"spring","trunk show","sale","offer","custom","style"' if not st.session_state.dictionary 
                  else ','.join([f'"{kw}"' for kw in st.session_state.dictionary]),
            height=100,
            help="Format: keyword1,keyword2 or \"keyword1\",\"keyword2\""
        )
    
    with col2:
        st.markdown("**Actions**")
        if st.button("💾 Update Dictionary"):
            st.session_state.dictionary = parse_dictionary_text(dict_input)
            st.success(f"✅ Updated dictionary with {len(st.session_state.dictionary)} keywords")
        
        if st.button("🗑️ Clear Dictionary"):
            st.session_state.dictionary = []
            st.success("✅ Dictionary cleared")
    
    # Step 3: Classification
    if (st.session_state.csv_data is not None and 
        st.session_state.dictionary and 
        'text_column' in locals()):
        
        st.markdown("<h2 class='step-header'>🎯 Step 3: Classify Statements</h2>", unsafe_allow_html=True)
        
        if st.button("🚀 Classify Statements", type="primary"):
            with st.spinner("Classifying statements..."):
                results_df, metrics = classify_statements(
                    st.session_state.csv_data, 
                    text_column, 
                    st.session_state.dictionary,
                    ground_truth_column
                )
                st.session_state.classification_results = (results_df, metrics)
                st.success("✅ Classification completed!")
    
    # Results Display
    if st.session_state.classification_results:
        results_df, metrics = st.session_state.classification_results
        
        st.markdown("<h2 class='step-header'>📊 Classification Results</h2>", unsafe_allow_html=True)
        
        # Metrics display
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Accuracy", f"{metrics['accuracy']:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Precision", f"{metrics['precision']:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Recall", f"{metrics['recall']:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col4:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("F1 Score", f"{metrics['f1_score']:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            st.write(f"**True Positives:** {metrics['true_positives']} | "
                    f"**False Positives:** {metrics['false_positives']} | "
                    f"**False Negatives:** {metrics['false_negatives']} | "
                    f"**True Negatives:** {metrics['true_negatives']}")
        
        # False positives and negatives
        if ground_truth_column:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("False Positives")
                false_positives = results_df[results_df['category'] == 'FP']
                if not false_positives.empty:
                    for _, row in false_positives.head(5).iterrows():
                        st.markdown(
                            f"<div class='false-positive'>"
                            f"<strong>Text:</strong> {row[text_column][:200]}...<br>"
                            f"<strong>Matched:</strong> {', '.join(row['matched_keywords'])}"
                            f"</div>", 
                            unsafe_allow_html=True
                        )
                    if len(false_positives) > 5:
                        st.write(f"... and {len(false_positives) - 5} more")
                else:
                    st.write("No false positives")
            
            with col2:
                st.subheader("False Negatives")
                false_negatives = results_df[results_df['category'] == 'FN']
                if not false_negatives.empty:
                    for _, row in false_negatives.head(5).iterrows():
                        st.markdown(
                            f"<div class='false-negative'>"
                            f"<strong>Text:</strong> {row[text_column][:200]}..."
                            f"</div>", 
                            unsafe_allow_html=True
                        )
                    if len(false_negatives) > 5:
                        st.write(f"... and {len(false_negatives) - 5} more")
                else:
                    st.write("No false negatives")
        
        # Download results
        st.subheader("Download Results")
        csv_output = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Classification Results",
            data=csv_output,
            file_name="classification_results.csv",
            mime="text/csv"
        )
    
    # Step 4: Keyword Analysis
    if (st.session_state.classification_results and 
        ground_truth_column and 
        'text_column' in locals()):
        
        st.markdown("<h2 class='step-header'>🔍 Step 4: Keyword Impact Analysis</h2>", unsafe_allow_html=True)
        
        if st.button("📈 Analyze Keyword Impact"):
            with st.spinner("Analyzing keyword performance..."):
                analysis = analyze_keywords(
                    st.session_state.csv_data,
                    text_column,
                    ground_truth_column,
                    st.session_state.dictionary
                )
                st.session_state.keyword_analysis = analysis
                st.success("✅ Keyword analysis completed!")
        
        # Display keyword analysis
        if st.session_state.keyword_analysis:
            analysis = st.session_state.keyword_analysis
            
            # Metric selection tabs
            tab1, tab2, tab3 = st.tabs(["📈 Top by Recall", "🎯 Top by Precision", "⚖️ Top by F1 Score"])
            
            with tab1:
                st.info("**Recall:** Percentage of true positive cases captured. High recall means the keyword catches most relevant statements.")
                render_keyword_analysis_table(analysis['by_recall'], text_column, 'recall')
            
            with tab2:
                st.info("**Precision:** Percentage of predictions that are correct. High precision means the keyword rarely triggers on irrelevant statements.")
                render_keyword_analysis_table(analysis['by_precision'], text_column, 'precision')
            
            with tab3:
                st.info("**F1 Score:** Harmonic mean of precision and recall. Balances both metrics for overall effectiveness.")
                render_keyword_analysis_table(analysis['by_f1'], text_column, 'f1_score')
            
            # Download all keyword analysis combined
            st.subheader("📥 Download Complete Analysis")
            if st.button("📥 Download All Keywords Analysis"):
                # Create comprehensive downloadable CSV
                all_keywords = []
                for metric_type, keywords in analysis.items():
                    for i, kw in enumerate(keywords):
                        all_keywords.append({
                            'Metric_Type': metric_type.replace('by_', '').title(),
                            'Rank': i + 1,
                            'Keyword': kw['keyword'],
                            'Recall_%': round(kw['recall'], 2),
                            'Precision_%': round(kw['precision'], 2),
                            'F1_Score_%': round(kw['f1_score'], 2),
                            'True_Positives_Count': kw['true_positives_count'],
                            'False_Positives_Count': kw['false_positives_count']
                        })
                
                analysis_df = pd.DataFrame(all_keywords)
                csv_output = analysis_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Keyword Analysis CSV",
                    data=csv_output,
                    file_name=f"complete_keyword_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_complete_analysis"
                )

if __name__ == "__main__":
    main()
