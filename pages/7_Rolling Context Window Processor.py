import streamlit as st
import pandas as pd
import re
import io

def clean_text(text, emoji_removal=True, whitespace_normalization=True):
    """Clean text by removing emojis and normalizing whitespace"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    if emoji_removal:
        # Remove emojis (basic Unicode ranges)
        emoji_pattern = re.compile("["
                                 u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                 u"\U00002702-\U000027B0"
                                 u"\U000024C2-\U0001F251"
                                 "]+", flags=re.UNICODE)
        text = emoji_pattern.sub('', text)
    
    if whitespace_normalization:
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_sentences(text, sentence_endings='.!?', preserve_hashtags=True):
    """Simple sentence tokenization with customizable parameters"""
    if pd.isna(text) or text == "":
        return []
    
    text = str(text)
    
    if preserve_hashtags:
        # First, separate hashtags at the end
        hashtag_match = re.search(r'(\s+#\w+(?:\s+#\w+)*)\s*$', text)
        hashtags = hashtag_match.group(1).strip() if hashtag_match else ""
        
        # Remove hashtags from main text for sentence splitting
        main_text = re.sub(r'\s+#\w+(?:\s+#\w+)*\s*$', '', text)
    else:
        main_text = text
        hashtags = ""
    
    # Create pattern from sentence endings
    pattern = '[' + re.escape(sentence_endings) + ']+'
    
    # Split sentences on sentence endings
    sentences = re.split(pattern, main_text)
    
    # Clean and filter sentences
    result = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:  # Only add non-empty sentences
            result.append(sentence + '.')
    
    # Add hashtags as final sentence if they exist
    if hashtags and preserve_hashtags:
        result.append(hashtags)
    
    return result

def preprocess_data(df, id_column, speaker_column, turn_column, transcript_column, settings):
    """Transform raw conversation data to sentence-tokenized format"""
    
    # Initialize output data
    output_data = []
    
    for _, row in df.iterrows():
        conversation_id = row[id_column]
        speaker = row[speaker_column]
        turn = row[turn_column]
        transcript = row[transcript_column]
        
        # Clean the transcript
        cleaned_transcript = clean_text(
            transcript, 
            emoji_removal=settings['remove_emojis'],
            whitespace_normalization=settings['normalize_whitespace']
        )
        
        # Tokenize into sentences
        sentences = tokenize_sentences(
            cleaned_transcript,
            sentence_endings=settings['sentence_endings'],
            preserve_hashtags=settings['preserve_hashtags']
        )
        
        # Create rows for each sentence
        for i, sentence in enumerate(sentences, 1):
            output_data.append({
                'Call_ID': conversation_id,
                'Speaker': speaker,
                'Turn': turn,
                'Sentence_ID': i,
                'Original_Transcript': transcript,  # Keep original transcript as context
                'Processed_Statement': sentence
            })
    
    # Create output dataframe
    output_df = pd.DataFrame(output_data)
    
    return output_df

# Set page config
st.set_page_config(page_title="Conversation Data Preprocessor", page_icon="💬", layout="wide")

# Main app
st.title("💬 Conversation Data Preprocessor")
st.markdown("Transform your conversation/transcript dataset into sentence-tokenized format")

# Sidebar for settings
st.sidebar.header("⚙️ Processing Settings")

# Text cleaning settings
st.sidebar.subheader("Text Cleaning")
remove_emojis = st.sidebar.checkbox("Remove emojis", value=True)
normalize_whitespace = st.sidebar.checkbox("Normalize whitespace", value=True)

# Sentence tokenization settings
st.sidebar.subheader("Sentence Tokenization")
sentence_endings = st.sidebar.text_input(
    "Sentence endings", 
    value=".!?",
    help="Characters that mark the end of a sentence"
)
preserve_hashtags = st.sidebar.checkbox("Preserve hashtags as separate sentences", value=True)

# Collect settings
settings = {
    'remove_emojis': remove_emojis,
    'normalize_whitespace': normalize_whitespace,
    'sentence_endings': sentence_endings,
    'preserve_hashtags': preserve_hashtags
}

# File upload
st.header("📁 Upload Your Dataset")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload your Instagram posts dataset in CSV format"
)

if uploaded_file is not None:
    try:
        # Load the dataset
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Dataset loaded successfully! ({len(df)} rows, {len(df.columns)} columns)")
        
        # Show dataset preview
        with st.expander("📊 Dataset Preview", expanded=True):
            st.dataframe(df.head())
        
        # Column selection
        st.header("🎯 Column Mapping")
        st.markdown("Map your columns to the required fields (auto-detected defaults shown):")
        
        col1, col2 = st.columns(2)
        
        with col1:
            id_column = st.selectbox(
                "ID Column (Call/Conversation ID)",
                options=df.columns.tolist(),
                help="Column containing unique conversation/call identifiers"
            )
            
            turn_column = st.selectbox(
                "Turn Column",
                options=df.columns.tolist(),
                help="Column indicating the turn number or sequence"
            )
        
        with col2:
            speaker_column = st.selectbox(
                "Speaker Column",
                options=df.columns.tolist(),
                help="Column containing speaker information"
            )
            
            transcript_column = st.selectbox(
                "Statement/Transcript Column",
                options=df.columns.tolist(),
                help="Column containing the text content/transcript"
            )
        
        # Process button
        if st.button("🚀 Process Data", type="primary"):
            with st.spinner("Processing your dataset..."):
                try:
                    # Process the data
                    result_df = preprocess_data(df, id_column, speaker_column, turn_column, transcript_column, settings)
                    
                    # Display results
                    st.success(f"✅ Processing complete! Transformed {len(df)} records into {len(result_df)} sentences")
                    
                    # Show processed data preview
                    st.header("📋 Processed Data Preview")
                    st.dataframe(result_df.head(10))
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Original Records", len(df))
                    with col2:
                        st.metric("Total Sentences", len(result_df))
                    with col3:
                        avg_sentences = len(result_df) / len(df) if len(df) > 0 else 0
                        st.metric("Avg Sentences/Record", f"{avg_sentences:.2f}")
                    with col4:
                        unique_conversations = result_df['Call_ID'].nunique()
                        st.metric("Unique Conversations", unique_conversations)
                    
                    # Download button
                    st.header("💾 Download Results")
                    
                    # Convert dataframe to CSV
                    csv_buffer = io.StringIO()
                    result_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Processed Data as CSV",
                        data=csv_data,
                        file_name="processed_conversation_data.csv",
                        mime="text/csv"
                    )
                    
                    # Show sample transformations
                    st.header("🔍 Sample Transformations")
                    
                    # Get a few examples
                    sample_records = df.head(3)
                    
                    for idx, row in sample_records.iterrows():
                        with st.expander(f"Record {idx + 1}: {row[id_column]} - {row[speaker_column]}"):
                            st.subheader("Original Transcript:")
                            st.text(row[transcript_column])
                            
                            st.subheader("Processed Sentences:")
                            record_sentences = result_df[
                                (result_df['Call_ID'] == row[id_column]) & 
                                (result_df['Speaker'] == row[speaker_column]) &
                                (result_df['Turn'] == row[turn_column])
                            ]
                            for _, sentence_row in record_sentences.iterrows():
                                st.write(f"**Sentence {sentence_row['Sentence_ID']}:** {sentence_row['Processed_Statement']}")
                    
                except Exception as e:
                    st.error(f"❌ Error processing dataset: {str(e)}")
                    st.info("Please check your column selections and data format.")
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.info("Please make sure you've uploaded a valid CSV file.")

else:
    # Show instructions when no file is uploaded
    st.info("👆 Please upload a CSV file to get started")
    
    st.header("📖 How to Use")
    st.markdown("""
    1. **Upload your dataset**: Choose a CSV file containing conversation/transcript data
    2. **Configure settings**: Use the sidebar to customize text cleaning and tokenization
    3. **Map columns**: Select the appropriate columns for Call ID, Speaker, Turn, and Transcript
    4. **Process**: Click the process button to transform your data
    5. **Download**: Get your processed dataset as a CSV file
    
    ### Expected Input Format
    Your CSV should have at least four columns:
    - **ID column**: Unique identifier for each conversation/call
    - **Speaker column**: Who is speaking (e.g., 'Agent', 'Customer', 'Speaker1')
    - **Turn column**: Turn number or sequence in the conversation
    - **Transcript column**: The actual text/speech content
    
    ### Output Format
    The processed dataset will have these columns:
    - **Call_ID**: Original conversation identifier
    - **Speaker**: Speaker information
    - **Turn**: Turn number/sequence
    - **Sentence_ID**: Sequential number for each sentence within a turn
    - **Original_Transcript**: Original transcript text (for reference)
    - **Processed_Statement**: Individual sentence extracted from the transcript
    """)
