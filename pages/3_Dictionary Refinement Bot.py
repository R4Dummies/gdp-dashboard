import streamlit as st
import pandas as pd
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Dictionary Classification Bot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2d5016;
        margin-bottom: 2rem;
    }
    .step-header {
        background: linear-gradient(90deg, #22c55e, #16a34a);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0fdf4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #22c55e;
    }
    .keyword-tag {
        background: #fbbf24;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = None
if 'overall_metrics' not in st.session_state:
    st.session_state.overall_metrics = None
if 'keyword_analysis' not in st.session_state:
    st.session_state.keyword_analysis = None
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None

def create_sample_data():
    """Create sample data for demonstration"""
    sample_data = {
        'ID': [1, 2, 3, 4, 5],
        'Statement': [
            "It's SPRING TRUNK SHOW week!",
            "I am offering 4 shirts styled the way you want & the 5th is Also tossing in MAGNETIC COLLAR STAY to help keep your collars in place!",
            "In recognition of Earth Day, I would like to showcase our collection of Earth Fibers!",
            "It is now time to do some \"wardrobe crunches,\" and check your basics! Never on sale.",
            "He's a hard worker and always willing to lend a hand. The prices are the best I've seen in 17 years of servicing my clients."
        ],
        'Answer': [1, 1, 0, 1, 0]
    }
    return pd.DataFrame(sample_data)

def classify_statements(df, text_column, dictionary, ground_truth_column=None):
    """Classify statements based on dictionary keywords"""
    results = []
    
    for idx, row in df.iterrows():
        statement = str(row[text_column]).lower()
        matched_keywords = [keyword for keyword in dictionary if keyword.lower() in statement]
        predicted = 1 if len(matched_keywords) > 0 else 0
        
        result = {
            'index': idx,
            'statement': row[text_column],
            'predicted': predicted,
            'matched_keywords': matched_keywords,
            'score': len(matched_keywords)
        }
        
        if ground_truth_column and ground_truth_column in df.columns:
            ground_truth = int(row[ground_truth_column])
            result['ground_truth'] = ground_truth
            
            # Determine category
            if predicted == 1 and ground_truth == 1:
                result['category'] = 'TP'
            elif predicted == 1 and ground_truth == 0:
                result['category'] = 'FP'
            elif predicted == 0 and ground_truth == 1:
                result['category'] = 'FN'
            else:
                result['category'] = 'TN'
        
        results.append(result)
    
    return results

def calculate_metrics_manually(y_true, y_pred):
    """Calculate precision, recall, f1, and accuracy manually"""
    # Calculate confusion matrix components
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1, tp, fp, fn, tn

def calculate_overall_metrics(results):
    """Calculate overall classification metrics"""
    if not any('ground_truth' in r for r in results):
        return None
    
    y_true = [r['ground_truth'] for r in results if 'ground_truth' in r]
    y_pred = [r['predicted'] for r in results if 'ground_truth' in r]
    
    accuracy, precision, recall, f1, tp, fp, fn, tn = calculate_metrics_manually(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn)
    }
    
    return metrics

def analyze_keywords(df, text_column, dictionary, ground_truth_column):
    """Analyze individual keyword performance"""
    keyword_metrics = []
    
    for keyword in dictionary:
        true_positives = []
        false_positives = []
        total_positives = 0
        
        for idx, row in df.iterrows():
            statement = str(row[text_column]).lower()
            contains_keyword = keyword.lower() in statement
            ground_truth = int(row[ground_truth_column])
            
            if ground_truth == 1:
                total_positives += 1
                if contains_keyword:
                    true_positives.append(row)
            elif ground_truth == 0 and contains_keyword:
                false_positives.append(row)
        
        recall = len(true_positives) / total_positives if total_positives > 0 else 0
        precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
        
        keyword_metrics.append({
            'keyword': keyword,
            'true_positives_count': len(true_positives),
            'false_positives_count': len(false_positives),
            'recall': recall * 100,
            'precision': precision * 100,
            'f1_score': f1 * 100,
            'true_positive_examples': true_positives[:3],
            'false_positive_examples': false_positives[:3]
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

def create_metrics_chart(metrics):
    """Create a simple metrics visualization using Streamlit"""
    # Using Streamlit's built-in chart instead of Plotly
    chart_data = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
    })
    
    st.bar_chart(chart_data.set_index('Metric'))
    return None

def create_confusion_matrix_chart(metrics):
    """Create confusion matrix visualization using Streamlit"""
    # Create a simple table representation
    confusion_df = pd.DataFrame({
        'Predicted Negative': [metrics['true_negatives'], metrics['false_negatives']],
        'Predicted Positive': [metrics['false_positives'], metrics['true_positives']]
    }, index=['Actual Negative', 'Actual Positive'])
    
    st.write("**Confusion Matrix:**")
    st.dataframe(confusion_df)
    return None

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Dictionary Classification Bot</h1>
        <p>Enter keywords and classify statements to analyze their effectiveness</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation and controls
    with st.sidebar:
        st.header("📋 Navigation")
        step = st.selectbox(
            "Select Step:",
            ["Step 1: Upload Data", "Step 2: Configure Dictionary", "Step 3: Classify", "Step 4: Analyze Keywords"],
            index=0
        )
        
        st.markdown("---")
        
        # Sample data option
        if st.button("🔄 Load Sample Data"):
            st.session_state.csv_data = create_sample_data()
            st.success("Sample data loaded!")
            st.rerun()
    
    # Step 1: Input CSV Data
    if step == "Step 1: Upload Data":
        st.markdown('<div class="step-header"><h2>📊 Step 1: Input Sample Data (CSV Format)</h2></div>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload a CSV file with text data and optional ground truth labels"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.csv_data = df
                st.success(f"✅ Successfully loaded {len(df)} rows")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
        
        # Manual CSV input
        st.subheader("Or paste CSV data:")
        csv_text = st.text_area(
            "CSV Data",
            height=200,
            placeholder="ID,Statement,Answer\n1,Sample statement,1\n2,Another statement,0"
        )
        
        if csv_text:
            try:
                df = pd.read_csv(io.StringIO(csv_text))
                st.session_state.csv_data = df
                st.success(f"✅ Successfully parsed {len(df)} rows")
            except Exception as e:
                st.error(f"Error parsing CSV: {str(e)}")
        
        # Display data preview
        if st.session_state.csv_data is not None:
            st.subheader("📋 Data Preview")
            st.dataframe(st.session_state.csv_data.head(10))
            
            # Column selection
            cols = st.session_state.csv_data.columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                text_column = st.selectbox(
                    "Select Text Column for Analysis:",
                    cols,
                    index=1 if len(cols) > 1 else 0
                )
                st.session_state.text_column = text_column
            
            with col2:
                ground_truth_options = ['None'] + cols
                ground_truth_column = st.selectbox(
                    "Select Ground Truth Column (0/1 values):",
                    ground_truth_options
                )
                st.session_state.ground_truth_column = ground_truth_column if ground_truth_column != 'None' else None
    
    # Step 2: Configure Dictionary
    elif step == "Step 2: Configure Dictionary":
        st.markdown('<div class="step-header"><h2>📝 Step 2: Enter Keyword Dictionary</h2></div>', unsafe_allow_html=True)
        
        # Default dictionary
        default_keywords = "spring,trunk,show,shirt,custom,sale,special,offer,discount,promotion"
        
        # Dictionary input
        st.subheader("Enter Keywords")
        st.info("💡 Enter keywords separated by commas. These will be used to classify statements.")
        
        dictionary_text = st.text_area(
            "Keywords (comma-separated):",
            value=default_keywords,
            height=150,
            help="Enter keywords separated by commas, e.g.: custom,customized,customization"
        )
        
        # Parse dictionary
        if dictionary_text:
            keywords = [k.strip().strip('"') for k in dictionary_text.split(',') if k.strip()]
            st.session_state.dictionary = keywords
            
            # Display keywords
            st.subheader(f"📌 Keywords ({len(keywords)}):")
            keyword_html = ""
            for keyword in keywords:
                keyword_html += f'<span class="keyword-tag">{keyword}</span>'
            st.markdown(keyword_html, unsafe_allow_html=True)
        
        # Dictionary modification options
        st.subheader("🔧 Dictionary Modification")
        
        col1, col2 = st.columns(2)
        with col1:
            new_keyword = st.text_input("Add new keyword:")
            if st.button("➕ Add Keyword") and new_keyword:
                if 'dictionary' not in st.session_state:
                    st.session_state.dictionary = []
                if new_keyword not in st.session_state.dictionary:
                    st.session_state.dictionary.append(new_keyword)
                    st.success(f"Added '{new_keyword}' to dictionary")
                    st.rerun()
        
        with col2:
            if 'dictionary' in st.session_state and st.session_state.dictionary:
                remove_keyword = st.selectbox("Remove keyword:", st.session_state.dictionary)
                if st.button("🗑️ Remove Keyword"):
                    st.session_state.dictionary.remove(remove_keyword)
                    st.success(f"Removed '{remove_keyword}' from dictionary")
                    st.rerun()
    
    # Step 3: Classify
    elif step == "Step 3: Classify":
        st.markdown('<div class="step-header"><h2>🎯 Step 3: Classify Statements</h2></div>', unsafe_allow_html=True)
        
        # Check prerequisites
        if st.session_state.csv_data is None:
            st.warning("⚠️ Please upload data first (Step 1)")
            return
        
        if 'dictionary' not in st.session_state or not st.session_state.dictionary:
            st.warning("⚠️ Please configure dictionary first (Step 2)")
            return
        
        if 'text_column' not in st.session_state:
            st.warning("⚠️ Please select text column in Step 1")
            return
        
        # Classification button
        if st.button("🚀 Classify Statements", type="primary"):
            with st.spinner("Classifying statements..."):
                results = classify_statements(
                    st.session_state.csv_data,
                    st.session_state.text_column,
                    st.session_state.dictionary,
                    st.session_state.get('ground_truth_column')
                )
                st.session_state.classification_results = results
                
                # Calculate metrics if ground truth is available
                if st.session_state.get('ground_truth_column'):
                    metrics = calculate_overall_metrics(results)
                    st.session_state.overall_metrics = metrics
                
                st.success("✅ Classification completed!")
        
        # Display results
        if st.session_state.classification_results:
            st.subheader("📊 Classification Results")
            
            # Create results dataframe
            results_data = []
            for r in st.session_state.classification_results:
                statement = r['statement']
                # Truncate long statements
                if len(statement) > 100:
                    display_statement = statement[:100] + '...'
                else:
                    display_statement = statement
                
                results_data.append({
                    'Statement': display_statement,
                    'Predicted': '✅ Positive' if r['predicted'] == 1 else '❌ Negative',
                    'Matched Keywords': ', '.join(r['matched_keywords']) if r['matched_keywords'] else 'None',
                    'Score': r['score']
                })
            
            results_df = pd.DataFrame(results_data)
            
            st.dataframe(results_df, use_container_width=True)
            
            # Display metrics if available
            if st.session_state.overall_metrics:
                st.subheader("📈 Classification Metrics")
                
                # Metrics cards
                col1, col2, col3, col4 = st.columns(4)
                metrics = st.session_state.overall_metrics
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.1f}%")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.1f}%")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.1f}%")
                with col4:
                    st.metric("F1 Score", f"{metrics['f1_score']:.1f}%")
                
                # Charts
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 Metrics Overview")
                    create_metrics_chart(metrics)
                with col2:
                    st.subheader("🔢 Confusion Matrix")
                    create_confusion_matrix_chart(metrics)
                
                # Confusion matrix summary
                st.info(f"""
                **Confusion Matrix Summary:**
                - True Positives: {metrics['true_positives']}
                - False Positives: {metrics['false_positives']}
                - False Negatives: {metrics['false_negatives']}
                - True Negatives: {metrics['true_negatives']}
                """)
                
                # False positives and negatives analysis
                false_positives = [r for r in st.session_state.classification_results if r.get('category') == 'FP']
                false_negatives = [r for r in st.session_state.classification_results if r.get('category') == 'FN']
                
                if false_positives:
                    with st.expander(f"🔍 False Positives ({len(false_positives)})"):
                        for fp in false_positives[:10]:
                            st.write(f"**Statement:** {fp['statement']}")
                            st.write(f"**Matched Keywords:** {', '.join(fp['matched_keywords'])}")
                            st.write("---")
                
                if false_negatives:
                    with st.expander(f"🔍 False Negatives ({len(false_negatives)})"):
                        for fn in false_negatives[:10]:
                            st.write(f"**Statement:** {fn['statement']}")
                            st.write("**Issue:** No keywords matched")
                            st.write("---")
    
    # Step 4: Analyze Keywords
    elif step == "Step 4: Analyze Keywords":
        st.markdown('<div class="step-header"><h2>🔬 Step 4: Keyword Impact Analysis</h2></div>', unsafe_allow_html=True)
        
        # Check prerequisites
        if not st.session_state.classification_results:
            st.warning("⚠️ Please complete classification first (Step 3)")
            return
        
        if not st.session_state.get('ground_truth_column'):
            st.warning("⚠️ Ground truth column is required for keyword analysis")
            return
        
        # Keyword analysis button
        if st.button("🔍 Analyze Keyword Impact", type="primary"):
            with st.spinner("Analyzing keyword performance..."):
                keyword_analysis = analyze_keywords(
                    st.session_state.csv_data,
                    st.session_state.text_column,
                    st.session_state.dictionary,
                    st.session_state.ground_truth_column
                )
                st.session_state.keyword_analysis = keyword_analysis
                st.success("✅ Keyword analysis completed!")
        
        # Display keyword analysis
        if st.session_state.keyword_analysis:
            st.subheader("📊 Keyword Performance Analysis")
            
            # Metric selection
            metric_tabs = st.tabs(["🎯 Top by Recall", "🔧 Top by Precision", "⚖️ Top by F1 Score"])
            
            with metric_tabs[0]:
                st.info("**Recall:** Percentage of true positive cases captured. High recall means the keyword catches most relevant statements.")
                display_keyword_analysis(st.session_state.keyword_analysis['by_recall'], "recall")
            
            with metric_tabs[1]:
                st.info("**Precision:** Percentage of predictions that are correct. High precision means the keyword rarely triggers on irrelevant statements.")
                display_keyword_analysis(st.session_state.keyword_analysis['by_precision'], "precision")
            
            with metric_tabs[2]:
                st.info("**F1 Score:** Harmonic mean of precision and recall. Balances both metrics for overall effectiveness.")
                display_keyword_analysis(st.session_state.keyword_analysis['by_f1'], "f1")
            
            # Export option
            if st.button("📥 Export Keyword Analysis as CSV"):
                csv_data = export_keyword_analysis_csv(st.session_state.keyword_analysis)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"keyword_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def display_keyword_analysis(analysis_data, metric_type):
    """Display keyword analysis results"""
    for i, item in enumerate(analysis_data):
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown(f"**#{i+1}**")
                st.markdown(f'<span class="keyword-tag">{item["keyword"]}</span>', unsafe_allow_html=True)
            
            with col2:
                # Metrics
                col_r, col_p, col_f = st.columns(3)
                with col_r:
                    st.metric("Recall", f"{item['recall']:.1f}%")
                with col_p:
                    st.metric("Precision", f"{item['precision']:.1f}%")
                with col_f:
                    st.metric("F1 Score", f"{item['f1_score']:.1f}%")
                
                # Examples
                if item['true_positive_examples']:
                    with st.expander(f"✅ True Positives ({item['true_positives_count']})"):
                        for example in item['true_positive_examples']:
                            st.write(f"• {example[st.session_state.text_column]}")
                
                if item['false_positive_examples']:
                    with st.expander(f"❌ False Positives ({item['false_positives_count']})"):
                        for example in item['false_positive_examples']:
                            st.write(f"• {example[st.session_state.text_column]}")
            
            st.markdown("---")

def export_keyword_analysis_csv(keyword_analysis):
    """Export keyword analysis as CSV"""
    all_data = []
    
    for metric_type, data in keyword_analysis.items():
        for i, item in enumerate(data):
            all_data.append({
                'Metric_Type': metric_type.replace('by_', '').title(),
                'Rank': i + 1,
                'Keyword': item['keyword'],
                'Recall_%': round(item['recall'], 2),
                'Precision_%': round(item['precision'], 2),
                'F1_Score_%': round(item['f1_score'], 2),
                'True_Positives_Count': item['true_positives_count'],
                'False_Positives_Count': item['false_positives_count']
            })
    
    df = pd.DataFrame(all_data)
    return df.to_csv(index=False)

if __name__ == "__main__":
    main()
