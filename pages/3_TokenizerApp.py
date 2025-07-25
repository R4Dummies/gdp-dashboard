import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import io
from typing import Dict, Any

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download NLTK data with caching to avoid repeated downloads"""
    try:
        nltk.download('punkt', quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

def process_instagram_posts(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Process Instagram posts by tokenizing captions into individual sentences.
    
    Args:
        df: Input DataFrame
        column_mapping: Dictionary mapping standard names to actual column names
    
    Returns:
        DataFrame with processed data
    """
    
    # Map columns based on user input
    id_col = column_mapping['id_column']
    caption_col = column_mapping['caption_column']
    
    # Initialize lists for processed data
    processed_data = []
    
    # Process each row
    for _, row in df.iterrows():
        post_id = row[id_col]
        caption = row[caption_col]
        
        # Skip if caption is empty or NaN
        if pd.isna(caption) or not str(caption).strip():
            continue
            
        # Tokenize caption into sentences
        sentences = sent_tokenize(str(caption))
        
        # Create entry for each sentence
        for sentence_id, sentence in enumerate(sentences, 1):
            sentence = sentence.strip()
            if sentence:  # Only add non-empty sentences
                processed_data.append({
                    'ID': post_id,
                    'Sentence ID': sentence_id,
                    'Context': caption,
                    'Statement': sentence
                })
    
    # Create DataFrame
    df_processed = pd.DataFrame(processed_data)
    
    return df_processed

def main():
    st.set_page_config(
        page_title="Instagram Posts Processor",
        page_icon="📱",
        layout="wide"
    )
    
    st.title("📱 Instagram Posts Processor")
    st.markdown("Convert Instagram posts into individual sentences for analysis")
    
    # Download NLTK data
    if not download_nltk_data():
        st.stop()
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload a CSV file containing Instagram posts data"
    )
    
    if uploaded_file is not None:
        try:
            # Load the data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Show preview of uploaded data
            with st.expander("📋 Preview uploaded data", expanded=True):
                st.dataframe(df.head(10))
            
            # Column mapping section
            st.sidebar.subheader("📊 Column Mapping")
            st.sidebar.markdown("Map your CSV columns to the required fields:")
            
            available_columns = df.columns.tolist()
            
            # Default mappings
            default_id = 'shortcode' if 'shortcode' in available_columns else available_columns[0]
            default_caption = 'caption' if 'caption' in available_columns else (
                available_columns[1] if len(available_columns) > 1 else available_columns[0]
            )
            
            id_column = st.sidebar.selectbox(
                "ID Column (Post identifier)",
                available_columns,
                index=available_columns.index(default_id),
                help="Column containing unique post identifiers"
            )
            
            caption_column = st.sidebar.selectbox(
                "Caption Column (Text content)",
                available_columns,
                index=available_columns.index(default_caption),
                help="Column containing the post captions/text"
            )
            
            # Processing options
            st.sidebar.subheader("🔧 Processing Options")
            
            min_sentence_length = st.sidebar.slider(
                "Minimum sentence length (characters)",
                min_value=1,
                max_value=50,
                value=3,
                help="Skip sentences shorter than this length"
            )
            
            skip_empty_captions = st.sidebar.checkbox(
                "Skip posts with empty captions",
                value=True,
                help="Exclude posts that have no caption text"
            )
            
            # Column mapping dictionary
            column_mapping = {
                'id_column': id_column,
                'caption_column': caption_column
            }
            
            # Process button
            if st.sidebar.button("🚀 Process Data", type="primary"):
                with st.spinner("Processing Instagram posts..."):
                    try:
                        # Create a copy for processing
                        df_to_process = df.copy()
                        
                        # Apply minimum sentence length filter in processing
                        df_processed = process_instagram_posts(df_to_process, column_mapping)
                        
                        # Apply minimum sentence length filter
                        if min_sentence_length > 1:
                            df_processed = df_processed[
                                df_processed['Statement'].str.len() >= min_sentence_length
                            ]
                        
                        if len(df_processed) == 0:
                            st.warning("⚠️ No sentences were extracted. Please check your data and settings.")
                        else:
                            # Display results
                            st.header("📊 Processing Results")
                            
                            # Summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Sentences", len(df_processed))
                            
                            with col2:
                                st.metric("Unique Posts", df_processed['ID'].nunique())
                            
                            with col3:
                                avg_sentences = len(df_processed) / df_processed['ID'].nunique()
                                st.metric("Avg Sentences/Post", f"{avg_sentences:.1f}")
                            
                            with col4:
                                st.metric("Original Posts", len(df))
                            
                            # Display processed data
                            st.subheader("📋 Processed Data")
                            st.dataframe(df_processed, use_container_width=True)
                            
                            # Download section
                            st.subheader("💾 Download Results")
                            
                            # Convert to CSV for download
                            csv_buffer = io.StringIO()
                            df_processed.to_csv(csv_buffer, index=False)
                            csv_data = csv_buffer.getvalue()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.download_button(
                                    label="📥 Download as CSV",
                                    data=csv_data,
                                    file_name="instagram_posts_processed.csv",
                                    mime="text/csv",
                                    type="primary"
                                )
                            
                            with col2:
                                # Convert to Excel for download
                                excel_buffer = io.BytesIO()
                                df_processed.to_excel(excel_buffer, index=False, engine='openpyxl')
                                excel_data = excel_buffer.getvalue()
                                
                                st.download_button(
                                    label="📥 Download as Excel",
                                    data=excel_data,
                                    file_name="instagram_posts_processed.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            
                            # Show sample of processed data
                            with st.expander("👀 Sample of processed data"):
                                st.write("First 10 processed sentences:")
                                sample_df = df_processed.head(10)[['ID', 'Sentence ID', 'Statement']]
                                st.dataframe(sample_df, use_container_width=True)
                                
                    except Exception as e:
                        st.error(f"❌ Error processing data: {str(e)}")
            
            # Show column info
            with st.expander("ℹ️ Column Information"):
                st.write("**Selected Columns:**")
                st.write(f"- **ID Column**: {id_column}")
                st.write(f"- **Caption Column**: {caption_column}")
                
                st.write("**Available Columns:**")
                for i, col in enumerate(available_columns):
                    st.write(f"{i+1}. {col}")
                    
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Please make sure your file is a valid CSV format.")
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a CSV file to get started")
        
        st.markdown("""
        ### 📋 Expected CSV Format
        
        Your CSV file should contain at least two columns:
        - **ID Column**: Unique identifier for each post (e.g., 'shortcode', 'post_id')
        - **Caption Column**: Text content of the post (e.g., 'caption', 'text', 'content')
        
        ### 🔄 What this app does:
        
        1. **Tokenizes** each Instagram caption into individual sentences
        2. **Creates** a new row for each sentence with context
        3. **Assigns** sentence IDs for tracking
        4. **Preserves** the original caption as context
        
        ### 📊 Output Format:
        
        | ID | Sentence ID | Context | Statement |
        |----|-----------:|---------|-----------|
        | post123 | 1 | Full caption text... | First sentence. |
        | post123 | 2 | Full caption text... | Second sentence. |
        
        ### 🛠️ Customization Options:
        
        - **Column Mapping**: Choose which columns contain your IDs and captions
        - **Minimum Length**: Filter out very short sentences
        - **Empty Captions**: Option to skip posts without text
        """)

if __name__ == "__main__":
    main()
