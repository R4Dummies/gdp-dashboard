"""
Instagram Caption Transformation Streamlit App
Based on original by Dr. Yufan (Frank) Lin
Modified to be a web app with configurable preprocessing options
"""

import streamlit as st
import pandas as pd
import nltk
import re
import logging
import emoji
import csv
from typing import List, Dict, Optional
from nltk.tokenize import sent_tokenize
from unidecode import unidecode
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Class to handle text preprocessing operations for Instagram captions"""

    def __init__(self, config: Dict):
        self.config = config
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.hashtag_pattern = re.compile(r'#[\w\d]+')
        self.mention_pattern = re.compile(r'@[\w\d]+')

    def remove_emoji(self, text: str) -> str:
        """Remove emoji characters from text"""
        if self.config.get('remove_emojis', True):
            return emoji.replace_emoji(text, replace='')
        return text

    def clean_text(self, text: str) -> str:
        """Clean and normalize Instagram caption text"""
        if not isinstance(text, str):
            return self.config.get('empty_text_replacement', "[PAD]")
        
        try:
            # Remove emojis if configured
            if self.config.get('remove_emojis', True):
                text = self.remove_emoji(text)

            # Normalize text with unidecode if configured
            if self.config.get('normalize_unicode', True):
                text = unidecode(text)

            # Remove URLs if configured
            if self.config.get('remove_urls', True):
                text = self.url_pattern.sub('', text)
            
            # Remove email addresses if configured
            if self.config.get('remove_emails', True):
                text = self.email_pattern.sub('', text)
            
            # Remove hashtags if configured
            if self.config.get('remove_hashtags', True):
                text = self.hashtag_pattern.sub('', text)
            
            # Remove mentions if configured
            if self.config.get('remove_mentions', True):
                text = self.mention_pattern.sub('', text)

            # Replace newlines with spaces if configured
            if self.config.get('replace_newlines', True):
                text = text.replace('\n', ' ')

            # Remove extra spaces and strip
            if self.config.get('normalize_whitespace', True):
                text = ' '.join(text.split())

            return text if text.strip() else self.config.get('empty_text_replacement', "[PAD]")
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return self.config.get('empty_text_replacement', "[PAD]")

    def split_sentences(self, caption: str) -> List[str]:
        """Split caption text into sentences using NLTK"""
        try:
            # Handle Instagram captions that might not end with proper punctuation
            if self.config.get('add_period_if_missing', True) and caption and not caption[-1] in '.!?':
                caption = caption + '.'

            sentences = sent_tokenize(caption)
            # Filter out empty sentences
            return [sent.strip() for sent in sentences if sent.strip()]
        except Exception as e:
            logger.error(f"Error splitting sentences: {str(e)}")
            return []

    def transform_caption_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform Instagram caption dataframe into sentence-level data with specific columns"""
        transformed_rows = []

        for _, row in df.iterrows():
            # Skip if caption is missing
            if pd.isna(row['caption']):
                continue

            sentences = self.split_sentences(row['cleaned_caption'])
            for turn, sentence in enumerate(sentences, 1):
                transformed_row = {
                    'shortcode': row.get('shortcode', row.get('post_id', '')),
                    'turn': turn,
                    'caption': row['caption'],
                    'transcript': sentence,
                    'post_url': row.get('post_url', '')
                }
                transformed_rows.append(transformed_row)

        return pd.DataFrame(transformed_rows)

def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        with st.spinner('Downloading NLTK data...'):
            nltk.download('punkt', quiet=True)

def verify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Verify and prepare the dataframe for processing"""
    # Check for required caption column
    if 'caption' not in df.columns:
        st.error("The uploaded file must contain a 'caption' column.")
        st.stop()
    
    # Check for columns needed for output
    if 'shortcode' not in df.columns and 'post_id' not in df.columns:
        st.warning("Neither 'shortcode' nor 'post_id' found. Creating empty 'shortcode' column.")
        df['shortcode'] = ''
    
    if 'post_url' not in df.columns:
        st.info("Creating 'post_url' column from available data.")
        if 'shortcode' in df.columns:
            df['post_url'] = df['shortcode'].apply(lambda x: f"https://www.instagram.com/p/{x}/" if x else '')
        elif 'post_id' in df.columns:
            df['post_url'] = df['post_id'].apply(lambda x: f"https://www.instagram.com/p/{x}/" if x else '')
        else:
            df['post_url'] = ''
    
    return df

def main():
    st.set_page_config(
        page_title="Instagram Caption Transformation",
        page_icon="📸",
        layout="wide"
    )
    
    st.title("📸 Instagram Caption Transformation Tool")
    st.markdown("Transform Instagram captions into sentence-level data with configurable preprocessing options.")
    
    # Download NLTK data
    download_nltk_data()
    
    # Sidebar for configuration
    st.sidebar.header("🔧 Preprocessing Configuration")
    
    # Default configuration dictionary
    default_config = {
        'remove_emojis': True,
        'normalize_unicode': True,
        'remove_urls': True,
        'remove_emails': True,
        'remove_hashtags': True,
        'remove_mentions': True,
        'replace_newlines': True,
        'normalize_whitespace': True,
        'add_period_if_missing': True,
        'empty_text_replacement': '[PAD]'
    }
    
    # Create configuration interface
    config = {}
    
    st.sidebar.subheader("Text Cleaning Options")
    config['remove_emojis'] = st.sidebar.checkbox("Remove Emojis", default_config['remove_emojis'])
    config['normalize_unicode'] = st.sidebar.checkbox("Normalize Unicode Characters", default_config['normalize_unicode'])
    config['remove_urls'] = st.sidebar.checkbox("Remove URLs", default_config['remove_urls'])
    config['remove_emails'] = st.sidebar.checkbox("Remove Email Addresses", default_config['remove_emails'])
    config['remove_hashtags'] = st.sidebar.checkbox("Remove Hashtags", default_config['remove_hashtags'])
    config['remove_mentions'] = st.sidebar.checkbox("Remove Mentions (@username)", default_config['remove_mentions'])
    
    st.sidebar.subheader("Text Formatting Options")
    config['replace_newlines'] = st.sidebar.checkbox("Replace Newlines with Spaces", default_config['replace_newlines'])
    config['normalize_whitespace'] = st.sidebar.checkbox("Normalize Whitespace", default_config['normalize_whitespace'])
    config['add_period_if_missing'] = st.sidebar.checkbox("Add Period if Missing", default_config['add_period_if_missing'])
    
    st.sidebar.subheader("Other Options")
    config['empty_text_replacement'] = st.sidebar.text_input(
        "Empty Text Replacement", 
        default_config['empty_text_replacement']
    )
    
    # Reset to defaults button
    if st.sidebar.button("🔄 Reset to Defaults"):
        st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📂 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file containing Instagram captions. Must have a 'caption' column."
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Successfully loaded {len(df)} rows")
                
                # Show basic info about the dataset
                st.subheader("📊 Dataset Information")
                st.write(f"**Number of rows:** {len(df)}")
                st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
                
                # Show sample data
                st.subheader("🔍 Sample Data")
                st.dataframe(df.head(), use_container_width=True)
                
                # Verify and prepare dataframe
                df = verify_dataframe(df)
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.stop()
    
    with col2:
        st.header("⚙️ Current Configuration")
        
        # Display current configuration
        config_df = pd.DataFrame([
            {"Setting": key.replace('_', ' ').title(), "Value": str(value)}
            for key, value in config.items()
        ])
        st.dataframe(config_df, use_container_width=True, hide_index=True)
    
    # Processing section
    if uploaded_file is not None:
        st.header("🚀 Process Data")
        
        if st.button("Transform Captions", type="primary", use_container_width=True):
            with st.spinner("Processing captions..."):
                try:
                    # Initialize preprocessor with current config
                    preprocessor = TextPreprocessor(config)
                    
                    # Apply cleaning to captions
                    df['cleaned_caption'] = df['caption'].apply(
                        lambda x: preprocessor.clean_text(x)
                    )
                    
                    # Transform to sentence-level data
                    transformed_df = preprocessor.transform_caption_data(df)
                    
                    # Ensure correct column order
                    output_columns = ['shortcode', 'turn', 'caption', 'transcript', 'post_url']
                    transformed_df = transformed_df[output_columns]
                    
                    st.success(f"✅ Successfully created {len(transformed_df)} sentence-level records from {len(df)} captions")
                    
                    # Display results
                    st.subheader("📋 Transformation Results")
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Posts", len(df))
                    with col2:
                        st.metric("Generated Sentences", len(transformed_df))
                    with col3:
                        avg_sentences = len(transformed_df) / len(df) if len(df) > 0 else 0
                        st.metric("Avg Sentences per Post", f"{avg_sentences:.2f}")
                    
                    # Show preview of transformed data
                    st.subheader("👀 Preview Transformed Data")
                    st.dataframe(transformed_df.head(10), use_container_width=True)
                    
                    # Download section
                    st.subheader("💾 Download Results")
                    
                    # Convert to CSV for download
                    csv_buffer = io.StringIO()
                    transformed_df.to_csv(
                        csv_buffer,
                        index=False,
                        quoting=csv.QUOTE_NONNUMERIC,
                        quotechar='"',
                        escapechar='\\',
                        encoding='utf-8'
                    )
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Transformed Data as CSV",
                        data=csv_data,
                        file_name="ig_posts_transformed.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Show some examples of transformations
                    st.subheader("🔍 Transformation Examples")
                    
                    for i in range(min(3, len(df))):
                        if not pd.isna(df.iloc[i]['caption']):
                            with st.expander(f"Example {i+1}"):
                                st.write("**Original Caption:**")
                                st.write(df.iloc[i]['caption'][:200] + "..." if len(df.iloc[i]['caption']) > 200 else df.iloc[i]['caption'])
                                
                                st.write("**Cleaned Caption:**")
                                st.write(df.iloc[i]['cleaned_caption'])
                                
                                # Show corresponding sentences
                                post_sentences = transformed_df[
                                    transformed_df['shortcode'] == df.iloc[i].get('shortcode', df.iloc[i].get('post_id', ''))
                                ]['transcript'].tolist()
                                
                                st.write("**Generated Sentences:**")
                                for j, sentence in enumerate(post_sentences, 1):
                                    st.write(f"{j}. {sentence}")
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    logger.error(f"Processing error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Instagram Caption Transformation Tool | Based on original work by Dr. Yufan (Frank) Lin</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
