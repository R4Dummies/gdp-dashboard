"""
Text Transformation Streamlit App
Transform text data into sentence-level format (No external dependencies)
"""

import streamlit as st
import pandas as pd
import re
import io
from typing import List

def simple_sentence_tokenize(text):
    """Simple sentence tokenizer without external dependencies"""
    if not text:
        return []
    
    # Add period if text doesn't end with punctuation
    if text[-1] not in '.!?':
        text = text + '.'
    
    # Split on sentence endings, but be smart about abbreviations
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Clean up sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 2:  # Ignore very short fragments
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences

def extract_hashtags(text):
    """Extract hashtags from text and return them as a single string"""
    if not isinstance(text, str):
        return ""
    
    hashtags = re.findall(r'#\w+', text)
    return ' '.join(hashtags) if hashtags else ""

def clean_text(text):
    """Clean text by removing emojis, URLs, hashtags, and mentions"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Remove URLs, hashtags, mentions first
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    
    # Remove emojis and special characters (keep basic punctuation)
    text = re.sub(r'[^\w\s.,!?\'"()-]', '', text)
    
    # Replace newlines with spaces and clean up
    text = ' '.join(text.split())
    
    return text.strip()

def is_punctuation_only(text):
    """Check if text contains only punctuation marks and whitespace"""
    if not text:
        return True
    # Remove all whitespace and check if remaining characters are only punctuation
    cleaned = re.sub(r'\s+', '', text)
    if not cleaned:
        return True
    # Check if text contains only punctuation marks (no letters, numbers, or other meaningful characters)
    return bool(re.match(r'^[^\w#@]+$', cleaned))

def transform_data(df, id_column, context_column, include_hashtags=True):
    """Transform dataframe into sentence-level data"""
    transformed_rows = []
    
    for _, row in df.iterrows():
        row_id = row[id_column] if pd.notna(row[id_column]) else ""
        context = row[context_column] if pd.notna(row[context_column]) else ""
        
        if not context:
            continue
            
        # Clean the context text
        cleaned_context = clean_text(context)
        
        # Extract hashtags if requested
        hashtags = extract_hashtags(context) if include_hashtags else ""
        
        # Split into sentences
        sentences = simple_sentence_tokenize(cleaned_context)
        
        # Add hashtags as a separate sentence if they exist and contain actual content
        if hashtags and not is_punctuation_only(hashtags):
            sentences.append(hashtags)
        
        # Create rows for each sentence
        for sentence_id, sentence in enumerate(sentences, 1):
            transformed_rows.append({
                'ID': row_id,
                'Sentence ID': sentence_id,
                'Context': context,  # Original context
                'Statement': sentence
            })
    
    return pd.DataFrame(transformed_rows)

def main():
    st.set_page_config(
        page_title="Sentence Tokenizer App",
        page_icon="🔤",
        layout="wide"
    )
    
    st.title("🔤 Sentence Tokenizer App")
    st.markdown("Transform your text data into sentence-level analysis format")
    
    # Add info about no dependencies
    st.info("✅ This app works without any external NLP libraries!")
    
    # File upload in main area
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload a CSV file containing your text data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
            
            # Show preview of data
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Configuration section in main area
            st.markdown("---")
            st.subheader("⚙️ Configuration")
            
            # Single column layout for clean appearance
            st.markdown("**🎯 Column Mapping**")
            
            columns = list(df.columns)
            
            id_column = st.selectbox(
                "Select ID Column",
                options=columns,
                help="Choose the column that contains unique identifiers for your records"
            )
            
            context_column = st.selectbox(
                "Select Context Column", 
                options=columns,
                help="Choose the column that contains the text to be transformed into sentences"
            )
            
            st.markdown("**⚡ Options**")
            include_hashtags = st.checkbox(
                "Include hashtags as separate sentences",
                value=True,
                help="If checked, hashtags will be extracted and added as separate sentences"
            )
            
            # Transform button in main area
            st.markdown("---")
            if st.button("🚀 Transform Data", type="primary", use_container_width=True):
                if id_column == context_column:
                    st.error("ID and Context columns must be different!")
                else:
                    with st.spinner("Transforming data..."):
                        try:
                            transformed_df = transform_data(df, id_column, context_column, include_hashtags)
                            
                            if len(transformed_df) > 0:
                                st.balloons()  # 🎈 Original balloon effect!
                                st.success(f"Transformation complete! Generated {len(transformed_df)} sentences from {len(df)} records.")
                                
                                # Show results
                                st.markdown("---")
                                st.subheader("📈 Transformation Results")
                                st.dataframe(transformed_df.head(20), use_container_width=True)
                                
                                # Statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Original Records", len(df))
                                with col2:
                                    st.metric("Generated Sentences", len(transformed_df))
                                with col3:
                                    avg_sentences = len(transformed_df) / len(df) if len(df) > 0 else 0
                                    st.metric("Avg Sentences/Record", f"{avg_sentences:.1f}")
                                
                                # Download button
                                csv_buffer = io.StringIO()
                                transformed_df.to_csv(csv_buffer, index=False)
                                csv_data = csv_buffer.getvalue()
                                
                                st.download_button(
                                    label="📥 Download Transformed Data",
                                    data=csv_data,
                                    file_name="transformed_data.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                                
                            else:
                                st.warning("No data was generated. Please check your input data and column selections.")
                                
                        except Exception as e:
                            st.error(f"Error during transformation: {str(e)}")
            
            # Show column info
            st.markdown("---")
            st.subheader("📋 Column Information")
            col_info = []
            for col in df.columns:
                col_info.append({
                    "Column": col,
                    "Type": str(df[col].dtype),
                    "Non-null Count": df[col].count(),
                    "Sample Value": str(df[col].iloc[0]) if len(df) > 0 else "N/A"
                })
            
            st.dataframe(pd.DataFrame(col_info), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Please make sure your file is a valid CSV format.")
    
    else:
        # Instructions
        st.info("👆 Please upload a CSV file to get started")
        
        st.subheader("How to Use")
        st.markdown("""
        1. **Upload your CSV file** using the file uploader above
        2. **Select ID Column** - Choose the column that uniquely identifies each record
        3. **Select Context Column** - Choose the column containing the text to be transformed
        4. **Configure options** - Choose whether to include hashtags as separate sentences
        5. **Click Transform** - Process your data into sentence-level format
        6. **Download results** - Get your transformed data as a CSV file
        """)
        
        st.subheader("Output Format")
        st.markdown("""
        The transformed data will have the following columns:
        - **ID**: The identifier from your selected ID column
        - **Sentence ID**: Sequential number for each sentence within a record
        - **Context**: The original text from your Context column
        - **Statement**: Individual sentences extracted from the context
        """)
        
        # Example data
        st.subheader("Example")
        example_data = {
            "post_id": ["POST_001", "POST_002"],
            "caption": [
                "Great day at the beach! #summer #fun",
                "New product launch tomorrow. Very excited! #business #innovation"
            ]
        }
        example_df = pd.DataFrame(example_data)
        st.markdown("**Input Data:**")
        st.dataframe(example_df, use_container_width=True)
        
        example_output = {
            "ID": ["POST_001", "POST_001", "POST_002", "POST_002", "POST_002"],
            "Sentence ID": [1, 2, 1, 2, 3],
            "Context": [
                "Great day at the beach! #summer #fun",
                "Great day at the beach! #summer #fun", 
                "New product launch tomorrow. Very excited! #business #innovation",
                "New product launch tomorrow. Very excited! #business #innovation",
                "New product launch tomorrow. Very excited! #business #innovation"
            ],
            "Statement": [
                "Great day at the beach!",
                "#summer #fun",
                "New product launch tomorrow.",
                "Very excited!",
                "#business #innovation"
            ]
        }
        example_output_df = pd.DataFrame(example_output)
        st.markdown("**Output Data:**")
        st.dataframe(example_output_df, use_container_width=True)

if __name__ == "__main__":
    main()
