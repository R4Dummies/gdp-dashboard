import streamlit as st
import pandas as pd
import nltk
import re
import logging
import emoji
import csv
from typing import List, Dict
from unidecode import unidecode
from nltk.tokenize import sent_tokenize
import io
import zipfile

# Download required NLTK data (only once)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Class to handle text preprocessing operations for Instagram captions"""

    def __init__(self, custom_patterns=None):
        # Default patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.hashtag_pattern = re.compile(r'#[\w\d]+')
        self.mention_pattern = re.compile(r'@[\w\d]+')
        
        # Custom patterns from user input
        if custom_patterns:
            self.custom_patterns = {name: re.compile(pattern) for name, pattern in custom_patterns.items()}
        else:
            self.custom_patterns = {}

    def remove_emoji(self, text: str) -> str:
        """Remove emoji characters from text"""
        return emoji.replace_emoji(text, replace='')

    def clean_text(self, text: str, cleaning_options: Dict) -> str:
        """Clean and normalize Instagram caption text"""
        if not isinstance(text, str):
            return "[PAD]"
        try:
            # Remove emojis if option is enabled
            if cleaning_options.get('remove_emojis', True):
                text = self.remove_emoji(text)

            # Normalize text with unidecode if option is enabled
            if cleaning_options.get('normalize_unicode', True):
                text = unidecode(text)

            # Remove URLs if option is enabled
            if cleaning_options.get('remove_urls', True):
                text = self.url_pattern.sub('', text)
            
            # Remove email addresses if option is enabled
            if cleaning_options.get('remove_emails', True):
                text = self.email_pattern.sub('', text)
            
            # Remove hashtags if option is enabled
            if cleaning_options.get('remove_hashtags', True):
                text = self.hashtag_pattern.sub('', text)
            
            # Remove mentions if option is enabled
            if cleaning_options.get('remove_mentions', True):
                text = self.mention_pattern.sub('', text)

            # Apply custom patterns
            for name, pattern in self.custom_patterns.items():
                if cleaning_options.get(f'custom_{name}', False):
                    text = pattern.sub('', text)

            # Replace newlines with spaces
            text = text.replace('\n', ' ')

            # Remove extra spaces and strip
            text = ' '.join(text.split())

            return text if text.strip() else "[PAD]"
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return "[PAD]"

    def split_sentences(self, caption: str) -> List[str]:
        """Split caption text into sentences using NLTK"""
        try:
            # Handle Instagram captions that might not end with proper punctuation
            if caption and not caption[-1] in '.!?':
                caption = caption + '.'

            sentences = sent_tokenize(caption)
            # Filter out empty sentences
            return [sent.strip() for sent in sentences if sent.strip()]
        except Exception as e:
            logger.error(f"Error splitting sentences: {str(e)}")
            return []

    def transform_caption_data(self, df: pd.DataFrame, column_mapping: Dict) -> pd.DataFrame:
        """Transform Instagram caption dataframe into sentence-level data with specific columns"""
        transformed_rows = []

        for _, row in df.iterrows():
            # Skip if caption is missing
            caption_col = column_mapping.get('caption', 'caption')
            if pd.isna(row[caption_col]):
                continue

            sentences = self.split_sentences(row['cleaned_caption'])
            for turn, sentence in enumerate(sentences, 1):
                transformed_row = {
                    'shortcode': row.get(column_mapping.get('shortcode', 'shortcode'), 
                                       row.get(column_mapping.get('post_id', 'post_id'), '')),
                    'turn': turn,
                    'caption': row[caption_col],  # Keep original caption
                    'transcript': sentence,     # Renamed from 'sentence'
                    'post_url': row.get(column_mapping.get('post_url', 'post_url'), '')
                }
                transformed_rows.append(transformed_row)

        return pd.DataFrame(transformed_rows)

def main():
    st.set_page_config(
        page_title="Instagram Caption Transformation Tool",
        page_icon="📱",
        layout="wide"
    )
    
    st.title("📱 Instagram Caption Transformation Tool")
    st.markdown("Transform Instagram caption data into sentence-level records with customizable cleaning options.")
    
    # Download NLTK data
    download_nltk_data()
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload a CSV file containing Instagram caption data"
    )
    
    if uploaded_file is not None:
        try:
            # Try to read the CSV with different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    st.success(f"✅ Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                st.error("❌ Could not read CSV with any supported encoding")
                return
            
            # Show data preview
            st.subheader("📊 Data Preview")
            st.write(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            st.dataframe(df.head())
            
            # Column mapping section
            st.subheader("🔗 Column Mapping")
            st.write("Map your CSV columns to the required fields:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                caption_col = st.selectbox(
                    "Caption Column*",
                    options=df.columns.tolist(),
                    index=0 if 'caption' not in df.columns else df.columns.tolist().index('caption'),
                    help="Select the column containing Instagram captions"
                )
            
            with col2:
                shortcode_col = st.selectbox(
                    "Shortcode/Post ID Column",
                    options=['None'] + df.columns.tolist(),
                    index=0 if 'shortcode' not in df.columns else df.columns.tolist().index('shortcode') + 1,
                    help="Select the column containing post shortcodes or IDs"
                )
            
            with col3:
                post_url_col = st.selectbox(
                    "Post URL Column",
                    options=['None'] + df.columns.tolist(),
                    index=0 if 'post_url' not in df.columns else df.columns.tolist().index('post_url') + 1,
                    help="Select the column containing post URLs (optional)"
                )
            
            # Cleaning options
            st.sidebar.subheader("🧹 Text Cleaning Options")
            
            cleaning_options = {
                'remove_emojis': st.sidebar.checkbox("Remove Emojis", value=True),
                'normalize_unicode': st.sidebar.checkbox("Normalize Unicode", value=True),
                'remove_urls': st.sidebar.checkbox("Remove URLs", value=True),
                'remove_emails': st.sidebar.checkbox("Remove Email Addresses", value=True),
                'remove_hashtags': st.sidebar.checkbox("Remove Hashtags", value=True),
                'remove_mentions': st.sidebar.checkbox("Remove Mentions (@username)", value=True)
            }
            
            # Custom patterns section
            st.sidebar.subheader("🔧 Custom Patterns")
            st.sidebar.write("Add custom regex patterns to remove specific text:")
            
            custom_patterns = {}
            num_patterns = st.sidebar.number_input("Number of custom patterns", min_value=0, max_value=5, value=0)
            
            for i in range(num_patterns):
                pattern_name = st.sidebar.text_input(f"Pattern {i+1} Name", key=f"pattern_name_{i}")
                pattern_regex = st.sidebar.text_input(f"Pattern {i+1} Regex", key=f"pattern_regex_{i}")
                
                if pattern_name and pattern_regex:
                    try:
                        re.compile(pattern_regex)  # Test if regex is valid
                        custom_patterns[pattern_name] = pattern_regex
                        cleaning_options[f'custom_{pattern_name}'] = st.sidebar.checkbox(
                            f"Apply {pattern_name}", key=f"apply_{pattern_name}"
                        )
                    except re.error:
                        st.sidebar.error(f"Invalid regex pattern for {pattern_name}")
            
            # Processing button
            if st.button("🚀 Transform Data", type="primary"):
                with st.spinner("Processing data..."):
                    try:
                        # Create column mapping
                        column_mapping = {
                            'caption': caption_col,
                            'shortcode': shortcode_col if shortcode_col != 'None' else None,
                            'post_url': post_url_col if post_url_col != 'None' else None
                        }
                        
                        # Initialize preprocessor
                        preprocessor = TextPreprocessor(custom_patterns)
                        
                        # Clean captions
                        df['cleaned_caption'] = df[caption_col].apply(
                            lambda x: preprocessor.clean_text(x, cleaning_options)
                        )
                        
                        # Handle missing columns
                        if shortcode_col == 'None':
                            df['shortcode'] = ''
                        
                        if post_url_col == 'None':
                            if shortcode_col != 'None':
                                df['post_url'] = df[shortcode_col].apply(
                                    lambda x: f"https://www.instagram.com/p/{x}/" if x else ''
                                )
                            else:
                                df['post_url'] = ''
                        
                        # Transform data
                        transformed_df = preprocessor.transform_caption_data(df, column_mapping)
                        
                        # Display results
                        st.success(f"✅ Successfully transformed {len(df)} captions into {len(transformed_df)} sentence-level records")
                        
                        # Show transformation statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Records", len(df))
                        with col2:
                            st.metric("Transformed Records", len(transformed_df))
                        with col3:
                            avg_sentences = len(transformed_df) / len(df) if len(df) > 0 else 0
                            st.metric("Avg Sentences/Caption", f"{avg_sentences:.2f}")
                        
                        # Preview transformed data
                        st.subheader("📋 Transformed Data Preview")
                        st.dataframe(transformed_df.head(10))
                        
                        # Download options
                        st.subheader("⬇️ Download Results")
                        
                        # Prepare CSV for download
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
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.download_button(
                                label="📄 Download CSV",
                                data=csv_data,
                                file_name="ig_posts_transformed.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Create Excel file
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                transformed_df.to_excel(writer, index=False, sheet_name='Transformed_Data')
                                df[['cleaned_caption'] + [caption_col]].to_excel(
                                    writer, index=False, sheet_name='Cleaning_Preview'
                                )
                            
                            st.download_button(
                                label="📊 Download Excel",
                                data=excel_buffer.getvalue(),
                                file_name="ig_posts_transformed.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        # Show cleaning examples
                        if st.checkbox("Show cleaning examples"):
                            st.subheader("🔍 Text Cleaning Examples")
                            example_df = df[[caption_col, 'cleaned_caption']].head(5)
                            example_df.columns = ['Original Caption', 'Cleaned Caption']
                            st.dataframe(example_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        logger.error(f"Processing error: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a CSV file to get started")
        
        st.subheader("📋 Expected CSV Format")
        st.write("Your CSV file should contain at least one column with Instagram captions. Optional columns include:")
        
        example_data = pd.DataFrame({
            'caption': [
                'Amazing sunset today! 🌅 #beautiful #nature',
                'Check out our new product @instagram 💪',
                'Love this place! Visit https://example.com for more info'
            ],
            'shortcode': ['ABC123', 'DEF456', 'GHI789'],
            'post_url': [
                'https://www.instagram.com/p/ABC123/',
                'https://www.instagram.com/p/DEF456/',
                'https://www.instagram.com/p/GHI789/'
            ]
        })
        
        st.dataframe(example_data)
        
        st.subheader("✨ Features")
        features = [
            "🧹 **Text Cleaning**: Remove emojis, URLs, hashtags, mentions, and more",
            "🔧 **Custom Patterns**: Add your own regex patterns for specific cleaning needs",
            "📊 **Column Mapping**: Flexible mapping of your CSV columns to required fields",
            "📱 **Sentence Splitting**: Automatically split captions into individual sentences",
            "⬇️ **Multiple Formats**: Download results as CSV or Excel",
            "📈 **Statistics**: View transformation metrics and cleaning examples"
        ]
        
        for feature in features:
            st.markdown(feature)

if __name__ == "__main__":
    main()
