import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Dictionary Classifier",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #154734 0%, #1e5d47 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #154734;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .keyword-tag {
        background-color: #154734;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        margin: 0.25rem;
        display: inline-block;
        font-size: 0.875rem;
    }
    .stButton > button {
        background-color: #154734;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #1e5d47;
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

if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'ground_truth_data' not in st.session_state:
    st.session_state.ground_truth_data = None
if 'classified_data' not in st.session_state:
    st.session_state.classified_data = None

# Helper functions
def classify_text(text, keywords, case_sensitive=False, exact_match=False):
    """Classify text based on keyword matching"""
    if pd.isna(text) or not keywords:
        return 0
    
    text_to_search = text if case_sensitive else text.lower()
    keywords_to_search = keywords if case_sensitive else [k.lower() for k in keywords]
    
    for keyword in keywords_to_search:
        if exact_match:
            # Word boundary matching for exact words
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_to_search, re.IGNORECASE if not case_sensitive else 0):
                return 1
        else:
            # Partial matching
            if keyword in text_to_search:
                return 1
    
    return 0

def calculate_metrics(classified_data, ground_truth_data):
    """Calculate performance metrics"""
    # Merge data on ID
    merged_data = classified_data.merge(
        ground_truth_data, 
        on='ID', 
        how='inner'
    )
    
    if merged_data.empty:
        return None
    
    y_true = merged_data['Mode_Researcher'].astype(int)
    y_pred = merged_data['tactic_present'].astype(int)
    
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'total_samples': len(merged_data)
    }
    
    return metrics

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Dictionary Classifier</h1>
    <p>Cal Poly Pomona • Text Analysis Tool</p>
    <p>Classify text data using advanced keyword matching algorithms</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for settings
st.sidebar.header("⚙️ Configuration")

# File uploads
st.sidebar.subheader("📁 File Upload")

# Main CSV upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Data File (CSV)",
    type=['csv'],
    help="Expected columns: ID, Statement"
)

# Ground truth upload
ground_truth_file = st.sidebar.file_uploader(
    "Upload Ground Truth (Optional)",
    type=['csv'],
    help="Expected columns: ID, Mode_Researcher"
)

# Processing settings
st.sidebar.subheader("🔧 Processing Settings")
case_sensitive = st.sidebar.checkbox("Case Sensitive", value=False)
exact_match = st.sidebar.checkbox("Exact Word Match", value=False)

# Process uploaded files
if uploaded_file is not None:
    try:
        st.session_state.csv_data = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✅ Loaded {len(st.session_state.csv_data)} rows")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")

if ground_truth_file is not None:
    try:
        st.session_state.ground_truth_data = pd.read_csv(ground_truth_file)
        st.sidebar.success(f"✅ Loaded {len(st.session_state.ground_truth_data)} ground truth rows")
    except Exception as e:
        st.sidebar.error(f"Error loading ground truth file: {str(e)}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔤 Keywords Management")
    
    # Display current keywords
    st.subheader("Current Keywords")
    
    # Create a container for keywords with tags
    keywords_container = st.container()
    with keywords_container:
        keyword_html = ""
        for i, keyword in enumerate(st.session_state.keywords):
            keyword_html += f'<span class="keyword-tag">{keyword}</span>'
        st.markdown(keyword_html, unsafe_allow_html=True)
        st.write(f"**Total Keywords:** {len(st.session_state.keywords)}")
    
    # Add new keywords
    col_add, col_remove = st.columns([2, 1])
    
    with col_add:
        new_keywords = st.text_input(
            "Add Keywords (comma-separated)",
            placeholder="Enter keywords separated by commas..."
        )
        
        if st.button("➕ Add Keywords"):
            if new_keywords.strip():
                # Split by comma and clean up
                keywords_to_add = [
                    k.strip().lower() 
                    for k in new_keywords.split(',') 
                    if k.strip() and k.strip().lower() not in st.session_state.keywords
                ]
                
                if keywords_to_add:
                    st.session_state.keywords.extend(keywords_to_add)
                    st.success(f"Added {len(keywords_to_add)} new keywords!")
                    st.rerun()
                else:
                    st.warning("No new keywords to add (duplicates ignored)")
    
    with col_remove:
        if st.session_state.keywords:
            keyword_to_remove = st.selectbox(
                "Remove Keyword",
                options=["Select..."] + st.session_state.keywords
            )
            
            if st.button("🗑️ Remove") and keyword_to_remove != "Select...":
                st.session_state.keywords.remove(keyword_to_remove)
                st.success(f"Removed '{keyword_to_remove}'")
                st.rerun()
    
    # Reset to default
    if st.button("🔄 Reset to Default Keywords"):
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

with col2:
    st.header("📊 Quick Stats")
    
    if st.session_state.csv_data is not None:
        data = st.session_state.csv_data
        st.metric("Total Rows", len(data))
        
        if 'Statement' in data.columns:
            non_empty_statements = data['Statement'].notna().sum()
            st.metric("Non-empty Statements", non_empty_statements)
        
        if 'ID' in data.columns:
            unique_ids = data['ID'].nunique()
            st.metric("Unique IDs", unique_ids)
    else:
        st.info("Upload a CSV file to see statistics")

# Processing and Results
if st.session_state.csv_data is not None and len(st.session_state.keywords) > 0:
    st.header("🔄 Processing Results")
    
    # Process classification
    if st.button("🚀 Run Classification", type="primary"):
        with st.spinner("Processing classification..."):
            data = st.session_state.csv_data.copy()
            
            # Apply classification
            data['tactic_present'] = data['Statement'].apply(
                lambda x: classify_text(
                    x, 
                    st.session_state.keywords, 
                    case_sensitive, 
                    exact_match
                )
            )
            
            st.session_state.classified_data = data
            st.success("Classification completed!")
    
    # Display results if classification has been run
    if st.session_state.classified_data is not None:
        classified_data = st.session_state.classified_data
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_rows = len(classified_data)
        matches = (classified_data['tactic_present'] == 1).sum()
        no_matches = total_rows - matches
        match_percentage = (matches / total_rows * 100) if total_rows > 0 else 0
        
        with col1:
            st.metric("Total Rows", total_rows)
        with col2:
            st.metric("Matches", matches)
        with col3:
            st.metric("No Matches", no_matches)
        with col4:
            st.metric("Match %", f"{match_percentage:.1f}%")
        
        # Visualization
        st.subheader("📈 Distribution Chart")
        
        # Create pie chart
        fig = px.pie(
            values=[no_matches, matches],
            names=['No Match (0)', 'Match (1)'],
            color_discrete_sequence=['#dc2626', '#154734'],
            title="Classification Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics (if ground truth is available)
        if st.session_state.ground_truth_data is not None:
            st.subheader("📊 Performance Metrics")
            
            metrics = calculate_metrics(classified_data, st.session_state.ground_truth_data)
            
            if metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                with col2:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                with col3:
                    st.metric("F1 Score", f"{metrics['f1']:.3f}")
                with col4:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                
                st.info(f"Metrics calculated on {metrics['total_samples']} matched samples")
            else:
                st.warning("Unable to calculate metrics. Check that ID columns match between datasets.")
        
        # Data preview
        st.subheader("👀 Data Preview")
        
        # Show first 10 rows
        preview_data = classified_data.head(10)
        
        # Style the dataframe
        def highlight_matches(val):
            if val == 1:
                return 'background-color: #154734; color: white'
            else:
                return 'background-color: #dc2626; color: white'
        
        styled_df = preview_data.style.applymap(
            highlight_matches, 
            subset=['tactic_present']
        )
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Download button
        st.subheader("💾 Download Results")
        
        csv_buffer = StringIO()
        classified_data.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Classified Data (CSV)",
            data=csv_string,
            file_name="classified_data.csv",
            mime="text/csv"
        )

# Instructions
elif st.session_state.csv_data is None:
    st.header("🚀 Getting Started")
    
    st.markdown("""
    ### Welcome to the Cal Poly Pomona Dictionary Classifier
    
    This advanced text analysis tool helps you classify text data using keyword matching algorithms.
    
    **How to use:**
    
    1. **Upload your CSV file** using the sidebar (must contain 'ID' and 'Statement' columns)
    2. **Modify keywords** using the keywords management section above
    3. **Adjust settings** in the sidebar for case sensitivity and matching type
    4. **Run classification** to process your data
    5. **Review results** in the summary statistics and data preview
    6. **Optional:** Upload ground truth data to see performance metrics
    7. **Download** the classified data as a new CSV file
    
    **Expected CSV format:**
    - **ID**: Unique identifier for each row
    - **Statement**: Text content to be classified
    
    **Ground Truth format (optional):**
    - **ID**: Unique identifier matching your main data
    - **Mode_Researcher**: Binary classification (0 or 1)
    """)
    
    # Sample data format
    st.subheader("📋 Sample Data Format")
    
    sample_data = pd.DataFrame({
        'ID': [1, 2, 3],
        'Statement': [
            'This classic design represents timeless elegance',
            'Modern and trendy style for today',
            'Refined craftsmanship with luxurious materials'
        ]
    })
    
    st.dataframe(sample_data, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Dictionary Classifier • Cal Poly Pomona • Text Analysis Tool<br>
        Built with Streamlit 🚀
    </div>
    """, 
    unsafe_allow_html=True
)
