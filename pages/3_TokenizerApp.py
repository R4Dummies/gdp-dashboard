import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import io
import sys
import os
from typing import Dict, Any, Optional

# Configure Streamlit page
st.set_page_config(
    page_title="Instagram Posts Processor",
    page_icon="📱",
    layout="wide"
)

@st.cache_resource
def setup_nltk():
    """Setup NLTK with proper error handling"""
    try:
        # Try to use punkt tokenizer
        sent_tokenize("Test sentence.")
        return True
    except LookupError:
        try:
            # Download if not available
            with st.spinner("Downloading language data..."):
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)  # For newer NLTK versions
            sent_tokenize("Test sentence.")
            return True
        except Exception as e:
            st.error(f"Failed to setup NLTK: {e}")
            return False

def safe_sent_tokenize(text: str) -> list:
    """Safely tokenize text with fallback"""
    if not text or pd.isna(text):
        return []
    
    try:
        return sent_tokenize(str(text))
    except Exception:
        # Fallback: simple sentence splitting
        text = str(text)
        sentences = []
        for delimiter in ['. ', '! ', '? ']:
            text = text.replace(delimiter, delimiter + '<SPLIT>')
        
        parts = text.split('<SPLIT>')
        for part in parts:
            clean_part = part.strip()
            if clean_part:
                sentences.append(clean_part)
        
        return sentences if sentences else [str(text)]

def process_posts(df: pd.DataFrame, id_col: str, caption_col: str, min_length: int = 3) -> pd.DataFrame:
    """Process posts with better error handling"""
    processed_data = []
    errors = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_rows = len(df)
    
    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            # Update progress
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processing row {idx + 1} of {total_rows}")
            
            post_id = row[id_col]
            caption = row[caption_col]
            
            # Skip empty captions
            if pd.isna(caption) or not str(caption).strip():
                continue
            
            # Convert to string and clean
            caption_str = str(caption).strip()
            
            # Tokenize into sentences
            sentences = safe_sent_tokenize(caption_str)
            
            # Process each sentence
            for sentence_id, sentence in enumerate(sentences, 1):
                sentence = sentence.strip()
                if len(sentence) >= min_length:
                    processed_data.append({
                        'ID': str(post_id),
                        'Sentence ID': sentence_id,
                        'Context': caption_str,
                        'Statement': sentence
                    })
                    
        except Exception as e:
            errors += 1
            if errors <= 5:  # Show first 5 errors
                st.warning(f"Error processing row {idx + 1}: {str(e)}")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    if errors > 5:
        st.warning(f"... and {errors - 5} more errors occurred during processing")
    
    return pd.DataFrame(processed_data)

def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate uploaded dataframe"""
    if df.empty:
        return False, "The uploaded file is empty"
    
    if len(df.columns) < 1:
        return False, "The file must have at least one column"
    
    return True, "Valid"

def main():
    st.title("📱 Instagram Posts Processor")
    st.markdown("Convert Instagram posts into individual sentences for analysis")
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # Setup NLTK
    if not setup_nltk():
        st.error("Failed to setup required language processing tools. Please refresh the page.")
        st.stop()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload a CSV file containing your posts data"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV with different encodings if needed
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                    st.info("File loaded with latin-1 encoding")
                except UnicodeDecodeError:
                    df = pd.read_csv(uploaded_file, encoding='cp1252')
                    st.info("File loaded with cp1252 encoding")
            
            # Validate dataframe
            is_valid, message = validate_dataframe(df)
            if not is_valid:
                st.error(f"Invalid file: {message}")
                return
            
            st.success(f"✅ File loaded successfully! Shape: {df.shape}")
            
            # Show data preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(), use_container_width=True)
                
                # Show column info
                st.write("**Column Information:**")
                col_info = []
                for col in df.columns:
                    non_null = df[col].notna().sum()
                    col_info.append({
                        'Column': col,
                        'Type': str(df[col].dtype),
                        'Non-null': f"{non_null}/{len(df)}",
                        'Sample': str(df[col].dropna().iloc[0]) if non_null > 0 else "No data"
                    })
                st.dataframe(pd.DataFrame(col_info), use_container_width=True)
            
            # Configuration section
            st.header("⚙️ Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Column selection
                available_columns = df.columns.tolist()
                
                # Smart defaults
                id_default = next((col for col in available_columns 
                                 if any(keyword in col.lower() for keyword in ['id', 'shortcode', 'post'])), 
                                available_columns[0])
                
                caption_default = next((col for col in available_columns 
                                      if any(keyword in col.lower() for keyword in ['caption', 'text', 'content', 'description'])), 
                                     available_columns[-1] if len(available_columns) > 1 else available_columns[0])
                
                id_column = st.selectbox(
                    "Select ID Column",
                    available_columns,
                    index=available_columns.index(id_default),
                    help="Column containing unique identifiers for each post"
                )
                
                caption_column = st.selectbox(
                    "Select Caption/Text Column",
                    available_columns,
                    index=available_columns.index(caption_default),
                    help="Column containing the text content to process"
                )
            
            with col2:
                # Processing options
                min_sentence_length = st.number_input(
                    "Minimum sentence length (characters)",
                    min_value=1,
                    max_value=100,
                    value=3,
                    help="Skip sentences shorter than this length"
                )
                
                # Show preview of selected columns
                if id_column != caption_column:
                    st.write("**Selected Data Preview:**")
                    preview_df = df[[id_column, caption_column]].head(3)
                    st.dataframe(preview_df, use_container_width=True)
                else:
                    st.warning("⚠️ ID and Caption columns cannot be the same")
            
            # Process button
            if st.button("🚀 Process Data", type="primary", disabled=(id_column == caption_column)):
                if id_column == caption_column:
                    st.error("Please select different columns for ID and Caption")
                    return
                
                with st.spinner("Processing posts..."):
                    try:
                        # Process the data
                        processed_df = process_posts(
                            df, 
                            id_column, 
                            caption_column, 
                            min_sentence_length
                        )
                        
                        if processed_df.empty:
                            st.warning("⚠️ No sentences were extracted. Check your data and settings.")
                            return
                        
                        # Store in session state
                        st.session_state.processed_data = processed_df
                        
                        st.success(f"✅ Processing complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        st.write("**Debug info:**")
                        st.write(f"- Selected ID column: {id_column}")
                        st.write(f"- Selected caption column: {caption_column}")
                        st.write(f"- Data shape: {df.shape}")
    
    # Display results if available
    if st.session_state.processed_data is not None:
        processed_df = st.session_state.processed_data
        
        st.header("📊 Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sentences", len(processed_df))
        with col2:
            st.metric("Unique Posts", processed_df['ID'].nunique())
        with col3:
            avg_sentences = len(processed_df) / processed_df['ID'].nunique() if processed_df['ID'].nunique() > 0 else 0
            st.metric("Avg Sentences/Post", f"{avg_sentences:.1f}")
        with col4:
            avg_length = processed_df['Statement'].str.len().mean()
            st.metric("Avg Sentence Length", f"{avg_length:.0f} chars")
        
        # Data display
        st.subheader("📋 Processed Data")
        st.dataframe(processed_df, use_container_width=True)
        
        # Download section
        st.subheader("💾 Download")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv_data = processed_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="processed_sentences.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel download (with error handling)
            try:
                excel_buffer = io.BytesIO()
                processed_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name="processed_sentences.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except ImportError:
                st.info("Install openpyxl for Excel export: `pip install openpyxl`")
    
    # Instructions
    if uploaded_file is None:
        st.markdown("""
        ## 📋 How to Use
        
        1. **Upload** a CSV file with your Instagram posts data
        2. **Select** the columns containing post IDs and captions
        3. **Configure** processing options (minimum sentence length)
        4. **Process** the data to extract individual sentences
        5. **Download** the results as CSV or Excel
        
        ### 📊 Expected Format
        Your CSV should have at least:
        - **ID column**: Unique identifier for each post
        - **Text column**: The caption or content to process
        
        ### 🔄 What it does
        - Splits each caption into individual sentences
        - Creates a new row for each sentence
        - Preserves original context
        - Assigns sentence numbers
        """)

if __name__ == "__main__":
    main()
