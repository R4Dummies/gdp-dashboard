import streamlit as st
import pandas as pd
import re
import numpy as np
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Dictionary Classifier",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Cal Poly styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #154734, #1e5a3f);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #154734;
        margin: 0.5rem 0;
    }
    
    .keyword-tag {
        background-color: #154734;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.8rem;
    }
    
    .stTab [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'keywords' not in st.session_state:
    st.session_state.keywords = [
        'classic', 'timeless', 'refined', 'elegant', 'tailored', 'heritage', 
        'bespoke', 'polished', 'sophisticated', 'custom', 'understated', 
        'luxurious', 'impeccable', 'premium', 'crafted', 'high-end', 
        'clean lines', 'sharp', 'iconic', 'wardrobe staple', 'minimal', 
        'structured', 'graceful', 'prestigious', 'exquisite', 'exclusive', 
        'luxury', 'traditional', 'enduring', 'sleek'
    ]

if 'classified_data' not in st.session_state:
    st.session_state.classified_data = None

if 'ground_truth_data' not in st.session_state:
    st.session_state.ground_truth_data = None

# Main header
st.markdown("""
<div class="main-header">
    <h1>📊 Dictionary Classifier</h1>
    <p>Cal Poly Pomona • Text Analysis Tool</p>
    <p style="font-size: 1.1em; margin-top: 1rem;">Classify text data using advanced keyword matching algorithms</p>
</div>
""", unsafe_allow_html=True)

# Helper functions
def classify_text(text, keywords, case_sensitive=False, exact_match=False):
    """Classify text based on keyword presence"""
    if not text or not keywords:
        return 0
    
    text_to_search = text if case_sensitive else text.lower()
    keywords_to_search = keywords if case_sensitive else [k.lower() for k in keywords]
    
    for keyword in keywords_to_search:
        if exact_match:
            # Word boundary matching for exact words
            pattern = r'\b' + re.escape(keyword) + r'\b'
            flags = 0 if case_sensitive else re.IGNORECASE
            if re.search(pattern, text, flags):
                return 1
        else:
            # Partial matching
            if keyword in text_to_search:
                return 1
    
    return 0

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics manually"""
    # Convert to numpy arrays for easier processing
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'confusion_matrix': np.array([[tn, fp], [fn, tp]])
    }

def create_confusion_matrix(y_true, y_pred):
    """Create confusion matrix manually"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[tn, fp], [fn, tp]])

def create_download_link(df, filename="classified_data.csv"):
    """Create a download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Sidebar for settings and keyword management
with st.sidebar:
    st.header("🔧 Settings & Keywords")
    
    # Classification settings
    st.subheader("Classification Settings")
    case_sensitive = st.checkbox("Case Sensitive", value=False)
    exact_match = st.checkbox("Exact Word Match", value=False)
    
    st.markdown("---")
    
    # Keyword management
    st.subheader("Keyword Management")
    
    # Display current keywords
    st.write(f"**Current Keywords ({len(st.session_state.keywords)}):**")
    
    # Create a scrollable container for keywords
    keywords_container = st.container()
    with keywords_container:
        cols = st.columns(2)
        for i, keyword in enumerate(st.session_state.keywords):
            col_idx = i % 2
            with cols[col_idx]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f'<span class="keyword-tag">{keyword}</span>', unsafe_allow_html=True)
                with col2:
                    if st.button("❌", key=f"remove_{i}", help=f"Remove {keyword}"):
                        st.session_state.keywords.remove(keyword)
                        st.rerun()
    
    # Add new keywords
    st.subheader("Add Keywords")
    new_keywords_input = st.text_area(
        "Enter keywords (one per line or comma-separated):",
        placeholder="elegant, sophisticated, premium\nor one keyword per line"
    )
    
    if st.button("Add Keywords", type="primary"):
        if new_keywords_input:
            # Handle both comma-separated and line-separated input
            new_keywords = []
            for line in new_keywords_input.split('\n'):
                if ',' in line:
                    new_keywords.extend([k.strip().lower() for k in line.split(',') if k.strip()])
                else:
                    if line.strip():
                        new_keywords.append(line.strip().lower())
            
            # Remove duplicates and add to session state
            for keyword in new_keywords:
                if keyword and keyword not in st.session_state.keywords:
                    st.session_state.keywords.append(keyword)
            
            st.success(f"Added {len(new_keywords)} new keywords!")
            st.rerun()
    
    # Reset to default keywords
    if st.button("Reset to Default Keywords"):
        st.session_state.keywords = [
            'classic', 'timeless', 'refined', 'elegant', 'tailored', 'heritage', 
            'bespoke', 'polished', 'sophisticated', 'custom', 'understated', 
            'luxurious', 'impeccable', 'premium', 'crafted', 'high-end', 
            'clean lines', 'sharp', 'iconic', 'wardrobe staple', 'minimal', 
            'structured', 'graceful', 'prestigious', 'exquisite', 'exclusive', 
            'luxury', 'traditional', 'enduring', 'sleek'
        ]
        st.success("Keywords reset to default!")
        st.rerun()

# Main content area
tab1, tab2, tab3 = st.tabs(["📁 Data Upload", "📊 Analysis Results", "📈 Performance Metrics"])

with tab1:
    st.header("Data Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Upload Main Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Expected columns: ID, Statement"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows")
                
                # Validate required columns
                if 'ID' not in df.columns or 'Statement' not in df.columns:
                    st.error("❌ CSV must contain 'ID' and 'Statement' columns")
                else:
                    # Classify the data
                    with st.spinner("Classifying text data..."):
                        df['tactic_present'] = df['Statement'].apply(
                            lambda x: classify_text(x, st.session_state.keywords, case_sensitive, exact_match)
                        )
                    
                    st.session_state.classified_data = df
                    st.success("✅ Classification complete!")
                    
                    # Show preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
    
    with col2:
        st.subheader("🎯 Upload Ground Truth (Optional)")
        ground_truth_file = st.file_uploader(
            "Choose ground truth CSV file",
            type="csv",
            help="Expected columns: ID, Mode_Researcher",
            key="ground_truth"
        )
        
        if ground_truth_file is not None:
            try:
                gt_df = pd.read_csv(ground_truth_file)
                st.success(f"✅ Loaded {len(gt_df)} ground truth rows")
                
                # Validate required columns
                if 'ID' not in gt_df.columns or 'Mode_Researcher' not in gt_df.columns:
                    st.error("❌ Ground truth CSV must contain 'ID' and 'Mode_Researcher' columns")
                else:
                    st.session_state.ground_truth_data = gt_df
                    st.success("✅ Ground truth data loaded!")
                    
                    # Show preview
                    st.subheader("Ground Truth Preview")
                    st.dataframe(gt_df.head(10), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error reading ground truth CSV: {str(e)}")

with tab2:
    st.header("Analysis Results")
    
    if st.session_state.classified_data is not None:
        df = st.session_state.classified_data
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_rows = len(df)
        matches = len(df[df['tactic_present'] == 1])
        no_matches = total_rows - matches
        match_percentage = (matches / total_rows) * 100 if total_rows > 0 else 0
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #154734; margin: 0;">📊 Total Rows</h3>
                <h2 style="margin: 0.5rem 0;">{total_rows:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #154734; margin: 0;">✅ Matches</h3>
                <h2 style="margin: 0.5rem 0;">{matches:,}</h2>
                <p style="margin: 0; color: #666;">({match_percentage:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #154734; margin: 0;">❌ No Matches</h3>
                <h2 style="margin: 0.5rem 0;">{no_matches:,}</h2>
                <p style="margin: 0; color: #666;">({100-match_percentage:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #154734; margin: 0;">🔍 Keywords</h3>
                <h2 style="margin: 0.5rem 0;">{len(st.session_state.keywords)}</h2>
                <p style="margin: 0; color: #666;">Active keywords</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualization
        st.subheader("📈 Distribution Chart")
        
        # Create data for chart
        chart_data = pd.DataFrame({
            'Classification': ['No Match (0)', 'Match (1)'],
            'Count': [no_matches, matches]
        })
        
        # Display bar chart
        st.bar_chart(chart_data.set_index('Classification'))
        
        # Data table with filtering
        st.subheader("📋 Classified Data")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            show_filter = st.selectbox(
                "Filter by classification:",
                ["All", "Matches only (1)", "No matches only (0)"]
            )
        
        with col2:
            search_term = st.text_input("Search in statements:", placeholder="Enter search term...")
        
        # Apply filters
        filtered_df = df.copy()
        
        if show_filter == "Matches only (1)":
            filtered_df = filtered_df[filtered_df['tactic_present'] == 1]
        elif show_filter == "No matches only (0)":
            filtered_df = filtered_df[filtered_df['tactic_present'] == 0]
        
        if search_term:
            filtered_df = filtered_df[
                filtered_df['Statement'].str.contains(search_term, case=False, na=False)
            ]
        
        st.write(f"Showing {len(filtered_df)} of {len(df)} rows")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download button
        st.subheader("💾 Download Results")
        st.markdown(
            create_download_link(df, "classified_data.csv"),
            unsafe_allow_html=True
        )
        
    else:
        st.info("👆 Please upload a CSV file in the 'Data Upload' tab to see analysis results.")

with tab3:
    st.header("Performance Metrics")
    
    if (st.session_state.classified_data is not None and 
        st.session_state.ground_truth_data is not None):
        
        df = st.session_state.classified_data
        gt_df = st.session_state.ground_truth_data
        
        # Merge datasets on ID
        try:
            merged_df = df.merge(gt_df, on='ID', how='inner')
            
            if len(merged_df) == 0:
                st.error("❌ No matching IDs found between datasets")
            else:
                # Calculate metrics
                y_true = merged_df['Mode_Researcher'].astype(int)
                y_pred = merged_df['tactic_present']
                
                metrics = calculate_metrics(y_true, y_pred)
                
                st.success(f"✅ Metrics calculated for {len(merged_df)} matching samples")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: #154734; margin: 0;">🎯 Precision</h3>
                        <h2 style="margin: 0.5rem 0;">{metrics['precision']:.3f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: #154734; margin: 0;">📊 Recall</h3>
                        <h2 style="margin: 0.5rem 0;">{metrics['recall']:.3f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: #154734; margin: 0;">⚖️ F1 Score</h3>
                        <h2 style="margin: 0.5rem 0;">{metrics['f1']:.3f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: #154734; margin: 0;">✅ Accuracy</h3>
                        <h2 style="margin: 0.5rem 0;">{metrics['accuracy']:.3f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confusion matrix
                st.subheader("📊 Confusion Matrix")
                
                cm = create_confusion_matrix(y_true, y_pred)
                
                # Create a simple confusion matrix display
                cm_df = pd.DataFrame(
                    cm,
                    columns=['Predicted: No Match', 'Predicted: Match'],
                    index=['Actual: No Match', 'Actual: Match']
                )
                
                st.write("**Confusion Matrix:**")
                st.dataframe(cm_df, use_container_width=True)
                
                # Display confusion matrix values in a more readable format
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("True Negatives", int(cm[0,0]))
                    st.metric("False Positives", int(cm[0,1]))
                with col2:
                    st.metric("False Negatives", int(cm[1,0]))
                    st.metric("True Positives", int(cm[1,1]))
                
                # Performance breakdown
                st.subheader("📋 Detailed Analysis")
                
                # Show misclassified examples
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**False Positives (Predicted 1, Actual 0):**")
                    fp_examples = merged_df[(merged_df['tactic_present'] == 1) & 
                                          (merged_df['Mode_Researcher'] == 0)]
                    if len(fp_examples) > 0:
                        st.dataframe(fp_examples[['ID', 'Statement']].head(5), use_container_width=True)
                    else:
                        st.write("None")
                
                with col2:
                    st.write("**False Negatives (Predicted 0, Actual 1):**")
                    fn_examples = merged_df[(merged_df['tactic_present'] == 0) & 
                                          (merged_df['Mode_Researcher'] == 1)]
                    if len(fn_examples) > 0:
                        st.dataframe(fn_examples[['ID', 'Statement']].head(5), use_container_width=True)
                    else:
                        st.write("None")
                
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
    
    elif st.session_state.classified_data is None:
        st.info("👆 Please upload and classify your main dataset first.")
    else:
        st.info("👆 Please upload ground truth data to see performance metrics.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>📊 Dictionary Classifier • Cal Poly Pomona • Text Analysis Tool</p>
    <p style="font-size: 0.9em;">Upload your CSV data, customize keywords, and analyze text classification results</p>
</div>
""", unsafe_allow_html=True)
