import streamlit as st
import pandas as pd
import io
import json
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Dictionary Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Green and Gold theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f4f7f0 0%, #e8f5e8 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #2d5016, #4a7c23);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(45, 80, 22, 0.2);
    }
    
    .main-header h1 {
        color: #ffd700;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #f0f8e8;
        text-align: center;
        margin: 1rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2d5016, #4a7c23);
    }
    
    /* Custom containers */
    .green-container {
        background: linear-gradient(135deg, #e8f5e8, #f0f8e8);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4a7c23;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(74, 124, 35, 0.1);
    }
    
    .gold-container {
        background: linear-gradient(135deg, #fffbf0, #fff8e1);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffd700;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(255, 215, 0, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #4a7c23, #2d5016);
        color: #ffd700;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(45, 80, 22, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #2d5016, #1a2e0d);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(45, 80, 22, 0.4);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(45deg, #ffd700, #daa520);
        color: #2d5016;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(45deg, #daa520, #b8860b);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.4);
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        background: linear-gradient(135deg, #f0f8e8, #e8f5e8);
        border: 2px dashed #4a7c23;
        border-radius: 10px;
    }
    
    /* Text area */
    .stTextArea > div > div > textarea {
        background: #f9fdf9;
        border: 2px solid #4a7c23;
        border-radius: 8px;
    }
    
    /* Success messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 1px solid #4a7c23;
        color: #2d5016;
    }
    
    /* Error messages */
    .stError {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 1px solid #dc3545;
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #fffbf0, #fff8e1);
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #ffd700;
        text-align: center;
        margin: 0.5rem;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #e8f5e8, #f0f8e8);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4a7c23;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Dictionary Classifier App</h1>
    <p>Upload your dataset, customize your classification dictionary, and get intelligent category predictions</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for instructions
with st.sidebar:
    st.markdown("### 📋 Instructions")
    st.markdown("""
    **Step 1:** Upload your CSV file  
    **Step 2:** Select the text column  
    **Step 3:** Customize your dictionary  
    **Step 4:** Run the analysis  
    **Step 5:** Download results  
    """)
    
    st.markdown("### 💡 Dictionary Format")
    st.markdown("""
    **JSON Format:**
    ```json
    {
        "Luxury": ["elegant", "premium"],
        "Sport": ["athletic", "fitness"]
    }
    ```
    
    **Line Format:**
    ```
    Luxury: elegant, premium, sophisticated
    Sport: athletic, fitness, performance
    ```
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="green-container">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file containing text data for classification"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = None
if 'result_df' not in st.session_state:
    st.session_state.result_df = None

# File processing
if uploaded_file is not None:
    try:
        st.session_state.df_loaded = pd.read_csv(uploaded_file)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("📊 Rows", len(st.session_state.df_loaded))
            st.metric("📋 Columns", len(st.session_state.df_loaded.columns))
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.success("✅ CSV uploaded successfully!")
        
        # Display data preview
        st.markdown('<div class="gold-container">', unsafe_allow_html=True)
        st.markdown("### 👀 Data Preview")
        st.dataframe(
            st.session_state.df_loaded.head(),
            use_container_width=True,
            hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Column selection and dictionary input
        col3, col4 = st.columns([1, 2])
        
        with col3:
            st.markdown('<div class="green-container">', unsafe_allow_html=True)
            st.markdown("### 🎯 Select Text Column")
            text_col = st.selectbox(
                "Choose the column containing text to classify:",
                st.session_state.df_loaded.columns,
                help="Select the column that contains the text you want to classify"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="green-container">', unsafe_allow_html=True)
            st.markdown("### 📚 Classification Dictionary")
            
            # Default dictionary options
            default_options = {
                "Luxury Products": "Luxury: elegant, timeless, refined, classic, sophisticated, luxury, polished, premium, exclusive, high-end",
                "Technology": "Technology: digital, software, tech, innovation, smart, automated, AI, machine learning, data, cloud",
                "Health & Fitness": "Health: wellness, fitness, healthy, nutrition, exercise, medical, healthcare, therapy, treatment, recovery",
                "Custom": ""
            }
            
            preset_choice = st.selectbox(
                "Choose a preset dictionary or create custom:",
                list(default_options.keys())
            )
            
            if preset_choice != "Custom":
                default_dict = default_options[preset_choice]
            else:
                default_dict = 'Luxury: elegant, timeless, refined, classic, sophisticated, luxury, polished'
            
            dict_input = st.text_area(
                "Dictionary (JSON or line format):",
                value=default_dict,
                height=150,
                help='Enter JSON format or lines like "Category: keyword1, keyword2, ..."'
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis button
        st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
        if st.button("🔍 Run Classification Analysis", use_container_width=True):
            with st.spinner("🔄 Processing your data..."):
                dict_text = dict_input.strip()
                categories = {}
                
                # Parse dictionary
                try:
                    # Try JSON first
                    parsed = json.loads(dict_text)
                    if isinstance(parsed, dict):
                        for k, v in parsed.items():
                            if isinstance(v, str):
                                v = [v]
                            categories[k] = [str(x).strip() for x in v]
                except:
                    # Fall back to line format
                    for line in dict_text.splitlines():
                        if ':' in line:
                            cat, terms = line.split(':', 1)
                            categories[cat.strip()] = [w.strip() for w in terms.split(',') if w.strip()]
                
                if not categories:
                    st.error("❌ Invalid dictionary format. Please check the sidebar for examples.")
                else:
                    # Perform classification
                    results = []
                    confidence_scores = []
                    
                    for txt in st.session_state.df_loaded[text_col].astype(str):
                        txt_l = txt.lower()
                        best_cat, max_count = None, 0
                        total_words = len(txt_l.split())
                        
                        for cat, words in categories.items():
                            count = sum(txt_l.count(w.lower()) for w in words if w)
                            if count > max_count:
                                best_cat, max_count = cat, count
                        
                        results.append(best_cat)
                        confidence_scores.append(max_count / max(total_words, 1) if total_words > 0 else 0)
                    
                    # Create results dataframe
                    st.session_state.result_df = st.session_state.df_loaded.copy()
                    st.session_state.result_df['Predicted_Category'] = results
                    st.session_state.result_df['Confidence_Score'] = [round(score, 3) for score in confidence_scores]
                    
                    st.success("✅ Classification complete!")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Results display
        if st.session_state.result_df is not None:
            st.markdown('<div class="gold-container">', unsafe_allow_html=True)
            st.markdown("### 📈 Classification Results")
            
            # Summary metrics
            col5, col6, col7, col8 = st.columns(4)
            
            category_counts = st.session_state.result_df['Predicted_Category'].value_counts()
            avg_confidence = st.session_state.result_df['Confidence_Score'].mean()
            unclassified = (st.session_state.result_df['Predicted_Category'].isna().sum())
            
            with col5:
                st.metric("📊 Total Classified", len(st.session_state.result_df) - unclassified)
            with col6:
                st.metric("❓ Unclassified", unclassified)
            with col7:
                st.metric("🎯 Categories Found", len(category_counts))
            with col8:
                st.metric("📏 Avg Confidence", f"{avg_confidence:.3f}")
            
            # Category breakdown
            if len(category_counts) > 0:
                st.markdown("#### 📋 Category Breakdown")
                breakdown_df = pd.DataFrame({
                    'Category': category_counts.index,
                    'Count': category_counts.values,
                    'Percentage': [f"{(count/len(st.session_state.result_df)*100):.1f}%" 
                                 for count in category_counts.values]
                })
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
            
            # Sample results
            st.markdown("#### 🔍 Sample Results")
            display_df = st.session_state.result_df[[text_col, 'Predicted_Category', 'Confidence_Score']].head(10)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Download buttons
            col_csv, col_excel = st.columns(2)
            
            with col_csv:
                csv = st.session_state.result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name='classification_results.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            
            with col_excel:
                # Create Excel file in memory
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    st.session_state.result_df.to_excel(writer, sheet_name='Classification Results', index=False)
                    
                    # Add summary sheet
                    summary_data = {
                        'Metric': ['Total Rows', 'Classified Rows', 'Unclassified Rows', 'Categories Found', 'Average Confidence'],
                        'Value': [
                            len(st.session_state.result_df),
                            len(st.session_state.result_df) - unclassified,
                            unclassified,
                            len(category_counts),
                            f"{avg_confidence:.3f}"
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Add category breakdown sheet
                    if len(category_counts) > 0:
                        breakdown_df.to_excel(writer, sheet_name='Category Breakdown', index=False)
                
                excel_buffer.seek(0)
                
                st.download_button(
                    label="📊 Download as Excel",
                    data=excel_buffer.getvalue(),
                    file_name='classification_results.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
else:
    # Instructions when no file is uploaded
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Upload your CSV file** using the file uploader above
    2. **Select the text column** you want to classify
    3. **Choose or customize your dictionary** with relevant keywords for each category
    4. **Run the analysis** to get predictions for each row
    5. **Download the results** with predicted categories and confidence scores
    
    The app uses keyword matching to classify text into categories based on your custom dictionary.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #4a7c23; font-size: 0.9rem;">
    🧠 Dictionary Classifier App | Built with Streamlit | Green & Gold Theme
</div>
""", unsafe_allow_html=True)
