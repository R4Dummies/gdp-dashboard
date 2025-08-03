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

# Custom CSS for better UI
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .success-container {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .info-container {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
    .step-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'dictionary' not in st.session_state:
    st.session_state.dictionary = []
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = None
if 'keyword_analysis' not in st.session_state:
    st.session_state.keyword_analysis = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'step' not in st.session_state:
    st.session_state.step = 1

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
    """Classify statements using dictionary keywords with fixed metrics calculation"""
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
                ground_truth = int(float(row[ground_truth_column]))  # Handle float strings
                if predicted == 1 and ground_truth == 1:
                    tp += 1
                    category = 'TP'
                elif predicted == 1 and ground_truth == 0:
                    fp += 1
                    category = 'FP'
                elif predicted == 0 and ground_truth == 1:
                    fn += 1
                    category = 'FN'
                else:  # predicted == 0 and ground_truth == 0
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
    
    # Calculate metrics with proper formulas
    metrics = None
    if ground_truth_column and ground_truth_column in df.columns and (tp + fp + fn + tn) > 0:
        # Fixed metric calculations
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
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
    """Analyze individual keyword performance with fixed calculations"""
    keyword_metrics = []
    
    # Get total positives for recall calculation
    total_positives = 0
    try:
        total_positives = sum(1 for _, row in df.iterrows() 
                            if int(float(row[ground_truth_column])) == 1)
    except (ValueError, TypeError):
        pass
    
    for keyword in dictionary:
        tp_examples = []
        fp_examples = []
        tp_count = 0
        fp_count = 0
        
        for idx, row in df.iterrows():
            statement = str(row[text_column]).lower()
            contains_keyword = keyword.lower() in statement
            
            try:
                ground_truth = int(float(row[ground_truth_column]))
                
                if ground_truth == 1 and contains_keyword:
                    tp_count += 1
                    tp_examples.append(row[text_column])
                elif ground_truth == 0 and contains_keyword:
                    fp_count += 1
                    fp_examples.append(row[text_column])
            except (ValueError, TypeError):
                continue
        
        # Calculate metrics with proper formulas
        recall = tp_count / total_positives if total_positives > 0 else 0.0
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
        
        keyword_metrics.append({
            'keyword': keyword,
            'tp_count': tp_count,
            'fp_count': fp_count,
            'recall': recall * 100,
            'precision': precision * 100,
            'f1_score': f1 * 100,
            'tp_examples': tp_examples[:3],
            'fp_examples': fp_examples[:3]
        })
    
    return keyword_metrics

# Header with progress indicator
st.title("📚 Dictionary Classification Bot")
st.markdown("**A step-by-step guide to keyword-based text classification**")

# Progress indicator
progress_cols = st.columns(4)
steps = ["📊 Data", "📝 Dictionary", "🎯 Classify", "📈 Analysis"]
for i, (col, step_name) in enumerate(zip(progress_cols, steps)):
    with col:
        if i + 1 < st.session_state.step:
            st.success(f"✅ {step_name}")
        elif i + 1 == st.session_state.step:
            st.info(f"➡️ {step_name}")
        else:
            st.write(f"⭕ {step_name}")

st.markdown("---")

# Step 1: Data Input
st.markdown('<div class="step-header"><h2>Step 1: Load Your Data 📊</h2></div>', unsafe_allow_html=True)

# Quick start with sample data
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Quick Start with Sample Data", type="secondary"):
        sample_data = """ID,Statement,Answer
1,It's SPRING TRUNK SHOW week!,1
2,I am offering 4 shirts styled the way you want & the 5th is free!,1
3,In recognition of Earth Day I would like to showcase our collection of Earth Fibers!,0
4,It is now time to do some wardrobe crunches and check your basics! Never on sale.,1
5,He's a hard worker and always willing to lend a hand. The prices are the best I've seen.,0
6,Check out our new custom tailoring services available now!,1
7,The weather has been really nice lately for outdoor activities.,0
8,Special promotion on personalized clothing items this month!,1
9,I love reading books in my free time especially mysteries.,0
10,Our bespoke suits are crafted with attention to every detail.,1"""
        try:
            st.session_state.df = pd.read_csv(StringIO(sample_data))
            st.session_state.step = max(st.session_state.step, 2)
            st.rerun()
        except Exception as e:
            st.error(f"Error loading sample data: {e}")

# Data input options
input_method = st.radio("Choose your data input method:", 
                       ["📁 Upload CSV File", "✏️ Paste CSV Data"], 
                       horizontal=True)

if input_method == "📁 Upload CSV File":
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], help="Upload a CSV file with text data")
    
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.step = max(st.session_state.step, 2)
            st.success(f"✅ Successfully loaded {len(st.session_state.df)} rows!")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

else:  # Paste CSV Data
    csv_text = st.text_area(
        "Paste your CSV data here:",
        height=150,
        placeholder="ID,Statement,Answer\n1,Your text here,1\n2,Another text,0",
        help="Make sure your data includes headers and is properly formatted"
    )
    
    if csv_text.strip():
        try:
            st.session_state.df = pd.read_csv(StringIO(csv_text))
            st.session_state.step = max(st.session_state.step, 2)
            st.success(f"✅ Successfully parsed {len(st.session_state.df)} rows!")
        except Exception as e:
            st.error(f"❌ Error parsing CSV: {e}")

# Show data preview and column selection
if st.session_state.df is not None:
    st.markdown("### 👀 Data Preview")
    st.dataframe(st.session_state.df.head(), use_container_width=True)
    
    st.markdown("### 🎯 Column Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        text_column = st.selectbox(
            "📝 Select Text Column:",
            options=st.session_state.df.columns.tolist(),
            index=next((i for i, col in enumerate(st.session_state.df.columns) if 'statement' in col.lower() or 'text' in col.lower()), 0),
            help="Choose the column containing the text you want to classify"
        )
    
    with col2:
        ground_truth_options = ['None (no evaluation)'] + st.session_state.df.columns.tolist()
        gt_index = next((i+1 for i, col in enumerate(st.session_state.df.columns) if 'answer' in col.lower() or 'label' in col.lower()), 0)
        ground_truth_column = st.selectbox(
            "🎯 Select Ground Truth Column (optional):",
            options=ground_truth_options,
            index=gt_index,
            help="Choose a column with 0/1 values to evaluate classification performance"
        )
        if ground_truth_column == 'None (no evaluation)':
            ground_truth_column = None

    # Store selections in session state
    st.session_state.text_column = text_column
    st.session_state.ground_truth_column = ground_truth_column

st.markdown("---")

# Step 2: Dictionary Setup
st.markdown('<div class="step-header"><h2>Step 2: Create Your Keyword Dictionary 📝</h2></div>', unsafe_allow_html=True)

if st.session_state.step >= 2:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Provide examples and help
        st.markdown("### 📚 Enter Keywords")
        st.info("💡 **Tip**: Use keywords that appear in positive examples. You can enter them in multiple formats!")
        
        # Format selector
        format_type = st.radio("Choose input format:", 
                              ["🔤 Simple (comma-separated)", "📝 Quoted (with quotes)"], 
                              horizontal=True)
        
        if format_type == "🔤 Simple (comma-separated)":
            placeholder = "custom, customized, customization, bespoke, tailored"
            help_text = "Enter keywords separated by commas"
        else:
            placeholder = '"custom","customized","customization","bespoke","tailored"'
            help_text = "Enter keywords in quotes, separated by commas"
        
        dictionary_text = st.text_area(
            "Keywords:",
            value=','.join(f'"{kw}"' for kw in st.session_state.dictionary) if st.session_state.dictionary else "",
            height=100,
            placeholder=placeholder,
            help=help_text
        )
        
        if st.button("💾 Save Dictionary", type="primary"):
            keywords = parse_dictionary_text(dictionary_text)
            if keywords:
                st.session_state.dictionary = keywords
                st.session_state.step = max(st.session_state.step, 3)
                st.success(f"✅ Saved {len(keywords)} keywords!")
                st.rerun()
            else:
                st.error("❌ Please enter valid keywords")
    
    with col2:
        # Dictionary preview
        st.markdown("### 🔍 Current Dictionary")
        if st.session_state.dictionary:
            st.success(f"**{len(st.session_state.dictionary)} keywords loaded**")
            for i, kw in enumerate(st.session_state.dictionary, 1):
                st.write(f"{i}. `{kw}`")
        else:
            st.warning("No keywords saved yet")

st.markdown("---")

# Step 3: Classification
st.markdown('<div class="step-header"><h2>Step 3: Run Classification 🎯</h2></div>', unsafe_allow_html=True)

if st.session_state.step >= 3 and st.session_state.dictionary and st.session_state.df is not None:
    
    # Show what will be classified
    st.markdown("### 📋 Classification Setup")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Rows", len(st.session_state.df))
    with col2:
        st.metric("Keywords", len(st.session_state.dictionary))
    with col3:
        evaluation_available = "✅ Yes" if st.session_state.ground_truth_column else "❌ No"
        st.metric("Evaluation", evaluation_available)
    
    # Classification button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 **START CLASSIFICATION**", type="primary", use_container_width=True):
            with st.spinner("🔄 Classifying statements..."):
                try:
                    results, metrics = classify_statements(
                        st.session_state.df, 
                        st.session_state.text_column, 
                        st.session_state.dictionary, 
                        st.session_state.ground_truth_column
                    )
                    st.session_state.classification_results = results
                    st.session_state.metrics = metrics
                    st.session_state.step = max(st.session_state.step, 4)
                    st.success("✅ Classification completed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Classification failed: {e}")

elif st.session_state.step >= 3:
    st.warning("⚠️ Please complete Steps 1 and 2 first")

st.markdown("---")

# Step 4: Results
if st.session_state.classification_results:
    st.markdown('<div class="step-header"><h2>Step 4: View Results 📊</h2></div>', unsafe_allow_html=True)
    
    # Performance Metrics (if ground truth available)
    if hasattr(st.session_state, 'metrics') and st.session_state.metrics:
        st.markdown("### 🎯 Performance Metrics")
        
        metrics = st.session_state.metrics
        
        # Main metrics in colored containers
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        
        with met_col1:
            st.markdown(f"""
                <div style="background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="margin: 0; color: white;">Accuracy</h3>
                    <h2 style="margin: 0; color: white;">{metrics['accuracy']:.1f}%</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with met_col2:
            st.markdown(f"""
                <div style="background: linear-gradient(45deg, #f093fb, #f5576c); color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="margin: 0; color: white;">Precision</h3>
                    <h2 style="margin: 0; color: white;">{metrics['precision']:.1f}%</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with met_col3:
            st.markdown(f"""
                <div style="background: linear-gradient(45deg, #4facfe, #00f2fe); color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="margin: 0; color: white;">Recall</h3>
                    <h2 style="margin: 0; color: white;">{metrics['recall']:.1f}%</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with met_col4:
            st.markdown(f"""
                <div style="background: linear-gradient(45def, #43e97b, #38f9d7); color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="margin: 0; color: white;">F1 Score</h3>
                    <h2 style="margin: 0; color: white;">{metrics['f1_score']:.1f}%</h2>
                </div>
            """, unsafe_allow_html=True)
        
        # Confusion Matrix
        st.markdown("### 🔍 Detailed Breakdown")
        conf_col1, conf_col2, conf_col3, conf_col4 = st.columns(4)
        
        with conf_col1:
            st.success(f"✅ **True Positives**\n\n{metrics['tp']}")
        with conf_col2:
            st.error(f"❌ **False Positives**\n\n{metrics['fp']}")
        with conf_col3:
            st.warning(f"⚠️ **False Negatives**\n\n{metrics['fn']}")
        with conf_col4:
            st.info(f"✅ **True Negatives**\n\n{metrics['tn']}")
    
    # Results Table
    st.markdown("### 📋 Classification Results")
    
    results_df = pd.DataFrame(st.session_state.classification_results)
    
    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        show_category = st.selectbox("🔍 Filter by Result:", ['All'] + ['TP', 'FP', 'FN', 'TN'] if st.session_state.ground_truth_column else ['All'])
    
    with filter_col2:
        show_predicted = st.selectbox("📊 Filter by Prediction:", ['All', 'Positive (1)', 'Negative (0)'])
    
    with filter_col3:
        show_matched = st.selectbox("🎯 Filter by Keywords:", ['All', 'With Matches', 'No Matches'])
    
    # Apply filters
    filtered_df = results_df.copy()
    
    if show_category != 'All' and 'category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['category'] == show_category]
    
    if show_predicted == 'Positive (1)':
        filtered_df = filtered_df[filtered_df['predicted'] == 1]
    elif show_predicted == 'Negative (0)':
        filtered_df = filtered_df[filtered_df['predicted'] == 0]
    
    if show_matched == 'With Matches':
        filtered_df = filtered_df[filtered_df['score'] > 0]
    elif show_matched == 'No Matches':
        filtered_df = filtered_df[filtered_df['score'] == 0]
    
    st.info(f"📊 Showing {len(filtered_df)} of {len(results_df)} results")
    
    # Display results
    display_df = filtered_df[['statement', 'predicted', 'ground_truth', 'category', 'matched_keywords', 'score']].copy()
    display_df.columns = ['Statement', 'Predicted', 'Ground Truth', 'Result', 'Matched Keywords', 'Score']
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download button
    csv_results = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_results,
        file_name="classification_results.csv",
        mime="text/csv",
        use_container_width=True
    )

# Step 5: Keyword Analysis
if st.session_state.classification_results and st.session_state.ground_truth_column:
    st.markdown("---")
    st.markdown('<div class="step-header"><h2>Step 5: Keyword Performance Analysis 📈</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🔍 Analyze Individual Keywords")
        st.info("💡 **Tip**: This analysis shows which keywords are most effective for your classification task.")
    
    with col2:
        if st.button("📊 **ANALYZE KEYWORDS**", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing keyword performance..."):
                try:
                    keyword_analysis = analyze_keywords(
                        st.session_state.df, 
                        st.session_state.text_column, 
                        st.session_state.ground_truth_column, 
                        st.session_state.dictionary
                    )
                    st.session_state.keyword_analysis = keyword_analysis
                    st.success("✅ Keyword analysis completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")

# Show keyword analysis results
if st.session_state.keyword_analysis:
    st.markdown("### 🏆 Keyword Performance Rankings")
    
    # Metric selection with explanations
    metric_col1, metric_col2 = st.columns([1, 3])
    
    with metric_col1:
        analysis_metric = st.selectbox("📊 Rank by:", ['Recall', 'Precision', 'F1 Score'])
    
    with metric_col2:
        if analysis_metric == 'Recall':
            st.info("📈 **Recall**: Percentage of positive cases captured by this keyword")
        elif analysis_metric == 'Precision':
            st.info("🎯 **Precision**: Percentage of keyword matches that are actually positive")
        else:
            st.info("⚖️ **F1 Score**: Balanced measure combining precision and recall")
    
    # Sort and display keywords
    metric_key = analysis_metric.lower().replace(' ', '_')
    sorted_keywords = sorted(st.session_state.keyword_analysis, 
                           key=lambda x: x[metric_key], reverse=True)
    
    # Top performers summary
    st.markdown("### 🥇 Top Performing Keywords")
    
    for i, kw_data in enumerate(sorted_keywords[:5], 1):
        with st.expander(f"#{i} - **{kw_data['keyword']}** ({kw_data[metric_key]:.1f}% {analysis_metric})", expanded=i<=2):
            
            # Metrics row
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Recall", f"{kw_data['recall']:.1f}%")
            with metric_col2:
                st.metric("Precision", f"{kw_data['precision']:.1f}%")
            with metric_col3:
                st.metric("F1 Score", f"{kw_data['f1_score']:.1f}%")
            with metric_col4:
                st.metric("TP Count", kw_data['tp_count'])
            
            # Examples
            if kw_data['tp_examples'] or kw_data['fp_examples']:
                ex_col1, ex_col2 = st.columns(2)
                
                with ex_col1:
                    st.markdown("**✅ True Positives (sample):**")
                    if kw_data['tp_examples']:
                        for example in kw_data['tp_examples']:
                            st.write(f"• {example[:100]}{'...' if len(example) > 100 else ''}")
                    else:
                        st.write("*No examples*")
                
                with ex_col2:
                    st.markdown("**❌ False Positives (sample):**")
                    if kw_data['fp_examples']:
                        for example in kw_data['fp_examples']:
                            st.write(f"• {example[:100]}{'...' if len(example) > 100 else ''}")
                    else:
                        st.write("*No examples*")
    
    # Download keyword analysis
    keyword_df = pd.DataFrame(st.session_state.keyword_analysis)
    keyword_csv = keyword_df[['keyword', 'recall', 'precision', 'f1_score', 'tp_count', 'fp_count']].to_csv(index=False)
    
    st.download_button(
        label="📥 Download Keyword Analysis as CSV",
        data=keyword_csv,
        file_name="keyword_analysis.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🛠️ Built with Streamlit | 📊 Dictionary Classification Bot | 🚀 Ready to classify your text data!</p>
        <p><small>💡 Tip: Use the keyword analysis to refine your dictionary for better performance</small></p>
    </div>
""", unsafe_allow_html=True)
