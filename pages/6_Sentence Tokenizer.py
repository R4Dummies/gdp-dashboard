"""
🎉 Enhanced Text Transformation Streamlit App
Transform your text data into sentence-level analysis format with style!
"""

import streamlit as st
import pandas as pd
import nltk
import re
import io
import plotly.express as px
import plotly.graph_objects as go
from typing import List
import time

# Set page config with fun theme
st.set_page_config(
    page_title="✨ Text Transformer Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fun styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .upload-box {
        border: 2px dashed #4ECDC4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: rgba(78, 205, 196, 0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Download required NLTK data and setup sentence tokenizer
@st.cache_resource
def setup_sentence_tokenizer():
    try:
        import nltk
        # Try to download NLTK data
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize
        # Test if it works
        sent_tokenize("Test sentence.")
        return sent_tokenize
    except:
        # Fallback sentence tokenizer if NLTK fails
        def sent_tokenize(text):
            # Simple sentence splitting based on punctuation
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
        return sent_tokenize

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

def split_into_sentences(text, sent_tokenize):
    """Split text into sentences using NLTK or fallback tokenizer"""
    if not text:
        return []
    
    # Add period if text doesn't end with punctuation
    if text[-1] not in '.!?':
        text = text + '.'
    
    sentences = sent_tokenize(text)
    # Filter out empty sentences and punctuation-only sentences
    return [sent.strip() for sent in sentences if sent.strip() and not is_punctuation_only(sent.strip())]

def transform_data(df, id_column, context_column, include_hashtags=True, progress_bar=None):
    """Transform dataframe into sentence-level data"""
    sent_tokenize = setup_sentence_tokenizer()
    transformed_rows = []
    total_rows = len(df)
    
    for idx, (_, row) in enumerate(df.iterrows()):
        if progress_bar:
            progress_bar.progress((idx + 1) / total_rows)
        
        row_id = row[id_column] if pd.notna(row[id_column]) else ""
        context = row[context_column] if pd.notna(row[context_column]) else ""
        
        if not context:
            continue
            
        # Clean the context text
        cleaned_context = clean_text(context)
        
        # Extract hashtags if requested
        hashtags = extract_hashtags(context) if include_hashtags else ""
        
        # Split into sentences
        sentences = split_into_sentences(cleaned_context, sent_tokenize)
        
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

def create_analytics_charts(original_df, transformed_df):
    """Create fun analytics charts"""
    # Chart 1: Sentences per record distribution
    sentences_per_record = transformed_df.groupby('ID').size().reset_index(name='sentence_count')
    
    fig1 = px.histogram(
        sentences_per_record, 
        x='sentence_count',
        title="📊 Distribution of Sentences per Record",
        color_discrete_sequence=['#FF6B6B']
    )
    fig1.update_layout(
        xaxis_title="Number of Sentences",
        yaxis_title="Number of Records",
        showlegend=False
    )
    
    # Chart 2: Top records by sentence count
    top_records = sentences_per_record.nlargest(10, 'sentence_count')
    
    fig2 = px.bar(
        top_records,
        x='ID',
        y='sentence_count',
        title="🏆 Top 10 Records by Sentence Count",
        color='sentence_count',
        color_continuous_scale='Viridis'
    )
    fig2.update_layout(
        xaxis_title="Record ID",
        yaxis_title="Number of Sentences"
    )
    
    return fig1, fig2

def show_animated_success():
    """Show animated success message"""
    success_placeholder = st.empty()
    
    messages = [
        "🎉 Transformation starting...",
        "🔄 Processing your data...",
        "✨ Creating magic...",
        "🚀 Almost there...",
        "🎊 Transformation complete!"
    ]
    
    for msg in messages:
        success_placeholder.success(msg)
        time.sleep(0.5)
    
    success_placeholder.empty()

def main():
    # Fun animated header
    st.markdown('<h1 class="main-header">🚀 Text Transformer Pro ✨</h1>', unsafe_allow_html=True)
    
    # Subtitle with emojis
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">
        Transform your boring text data into exciting sentence-level insights! 📝➡️📊
    </div>
    """, unsafe_allow_html=True)
    
    # Fun stats counter
    if 'transformation_count' not in st.session_state:
        st.session_state.transformation_count = 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Transformations Today", st.session_state.transformation_count)
    with col2:
        st.metric("🔥 App Version", "2.0 Pro")
    with col3:
        st.metric("⭐ User Rating", "5.0/5.0")
    
    # Sidebar with fun styling
    st.sidebar.markdown("## 🎮 Control Panel")
    st.sidebar.markdown("---")
    
    # File upload with custom styling
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "🎯 Drop your CSV file here!",
        type=['csv'],
        help="Upload a CSV file containing your text data"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Animated success message
            st.balloons()
            st.success(f"🎉 File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
            
            # Show preview with fun styling
            st.markdown("## 👀 Data Preview")
            st.dataframe(
                df.head(), 
                use_container_width=True,
                height=200
            )
            
            # Sidebar configuration
            st.sidebar.markdown("### 🔧 Column Mapping")
            
            columns = list(df.columns)
            
            id_column = st.sidebar.selectbox(
                "🆔 Select ID Column",
                options=columns,
                help="Choose the column that contains unique identifiers"
            )
            
            context_column = st.sidebar.selectbox(
                "📝 Select Context Column", 
                options=columns,
                help="Choose the column containing text to transform"
            )
            
            # Fun options section
            st.sidebar.markdown("### 🎛️ Transformation Options")
            include_hashtags = st.sidebar.checkbox(
                "🏷️ Include hashtags as sentences",
                value=True,
                help="Extract hashtags and add them as separate sentences"
            )
            
            # Advanced options in expander
            with st.sidebar.expander("🔬 Advanced Options"):
                show_analytics = st.checkbox("📊 Show analytics charts", value=True)
                animate_progress = st.checkbox("🎭 Animate transformation", value=True)
            
            # Transform button with custom styling
            st.sidebar.markdown("---")
            transform_clicked = st.sidebar.button(
                "🚀 Transform My Data!", 
                type="primary",
                use_container_width=True
            )
            
            if transform_clicked:
                if id_column == context_column:
                    st.error("❌ ID and Context columns must be different!")
                else:
                    # Show animated progress
                    if animate_progress:
                        show_animated_success()
                    
                    progress_bar = st.progress(0) if animate_progress else None
                    status_text = st.empty()
                    
                    with st.spinner("🔄 Transforming your data into pure awesomeness..."):
                        try:
                            transformed_df = transform_data(
                                df, id_column, context_column, 
                                include_hashtags, progress_bar
                            )
                            
                            if progress_bar:
                                progress_bar.empty()
                            status_text.empty()
                            
                            if len(transformed_df) > 0:
                                # Update transformation counter
                                st.session_state.transformation_count += 1
                                
                                # Success celebration
                                st.success(f"🎊 Transformation complete! Generated {len(transformed_df)} sentences from {len(df)} records.")
                                st.balloons()
                                
                                # Results section
                                st.markdown("## 🎯 Transformation Results")
                                
                                # Fun metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("📄 Original Records", len(df), delta=None)
                                with col2:
                                    st.metric("✨ Generated Sentences", len(transformed_df), delta=f"+{len(transformed_df)}")
                                with col3:
                                    avg_sentences = len(transformed_df) / len(df) if len(df) > 0 else 0
                                    st.metric("📊 Avg Sentences/Record", f"{avg_sentences:.1f}")
                                with col4:
                                    efficiency = (len(transformed_df) / len(df) - 1) * 100 if len(df) > 0 else 0
                                    st.metric("🚀 Data Expansion", f"+{efficiency:.0f}%")
                                
                                # Show results table
                                st.dataframe(
                                    transformed_df.head(20), 
                                    use_container_width=True,
                                    height=400
                                )
                                
                                # Analytics charts
                                if show_analytics and len(transformed_df) > 0:
                                    st.markdown("## 📊 Fun Analytics")
                                    fig1, fig2 = create_analytics_charts(df, transformed_df)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.plotly_chart(fig1, use_container_width=True)
                                    with col2:
                                        st.plotly_chart(fig2, use_container_width=True)
                                
                                # Download section with fun styling
                                st.markdown("## 💾 Download Your Masterpiece")
                                
                                csv_buffer = io.StringIO()
                                transformed_df.to_csv(csv_buffer, index=False)
                                csv_data = csv_buffer.getvalue()
                                
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.download_button(
                                        label="📥 Download Transformed Data",
                                        data=csv_data,
                                        file_name=f"transformed_data_{st.session_state.transformation_count}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                with col2:
                                    if st.button("🎉 Celebrate!", use_container_width=True):
                                        st.balloons()
                                        st.success("🎊 You're awesome!")
                                
                            else:
                                st.warning("🤔 No data was generated. Please check your input data and column selections.")
                                
                        except Exception as e:
                            st.error(f"💥 Oops! Something went wrong: {str(e)}")
                            st.info("💡 Try checking your data format or column selections!")
            
            # Column information with fun styling
            with st.expander("🔍 Column Information", expanded=False):
                col_info = []
                for col in df.columns:
                    col_info.append({
                        "Column": f"📋 {col}",
                        "Type": str(df[col].dtype),
                        "Non-null Count": f"{df[col].count():,}",
                        "Sample Value": str(df[col].iloc[0])[:50] + "..." if len(str(df[col].iloc[0])) > 50 else str(df[col].iloc[0])
                    })
                
                st.dataframe(pd.DataFrame(col_info), use_container_width=True)
                
        except Exception as e:
            st.error(f"💥 Error reading file: {str(e)}")
            st.info("💡 Please make sure your file is a valid CSV format!")
    
    else:
        # Welcome section with fun instructions
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     border-radius: 15px; color: white; margin: 2rem 0;">
            <h2>🎯 Ready to Transform Your Data?</h2>
            <p style="font-size: 1.1rem;">Upload your CSV file above to get started with the magic! ✨</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Fun how-to section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎮 How to Use")
            st.markdown("""
            1. 📤 **Upload your CSV file** using the uploader above
            2. 🆔 **Select ID Column** - Your unique record identifier  
            3. 📝 **Select Context Column** - Text to be transformed
            4. 🎛️ **Configure options** - Customize your transformation
            5. 🚀 **Click Transform** - Watch the magic happen!
            6. 📥 **Download results** - Get your awesome new dataset
            """)
        
        with col2:
            st.markdown("### 📊 Output Format")
            st.markdown("""
            Your transformed data will include:
            - 🆔 **ID**: Original record identifier
            - 🔢 **Sentence ID**: Sequential sentence numbers
            - 📝 **Context**: Original text content
            - ✨ **Statement**: Individual extracted sentences
            """)
        
        # Fun example section
        st.markdown("### 🎭 See It In Action!")
        
        tab1, tab2 = st.tabs(["📥 Input Example", "📤 Output Example"])
        
        with tab1:
            example_data = {
                "post_id": ["POST_001", "POST_002", "POST_003"],
                "caption": [
                    "Amazing sunset at the beach! Perfect end to a great day. #sunset #beach #happiness",
                    "New product launch tomorrow. Very excited! Can't wait to share it with everyone. #business #innovation #excited",
                    "Coffee and coding session. Productivity at its finest! #coffee #coding #productivity"
                ]
            }
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df, use_container_width=True)
        
        with tab2:
            example_output = {
                "ID": ["POST_001", "POST_001", "POST_001", "POST_001", "POST_002", "POST_002", "POST_002", "POST_002"],
                "Sentence ID": [1, 2, 3, 4, 1, 2, 3, 4],
                "Context": [
                    "Amazing sunset at the beach! Perfect end to a great day. #sunset #beach #happiness",
                    "Amazing sunset at the beach! Perfect end to a great day. #sunset #beach #happiness",
                    "Amazing sunset at the beach! Perfect end to a great day. #sunset #beach #happiness",
                    "Amazing sunset at the beach! Perfect end to a great day. #sunset #beach #happiness",
                    "New product launch tomorrow. Very excited! Can't wait to share it with everyone. #business #innovation #excited",
                    "New product launch tomorrow. Very excited! Can't wait to share it with everyone. #business #innovation #excited",
                    "New product launch tomorrow. Very excited! Can't wait to share it with everyone. #business #innovation #excited",
                    "New product launch tomorrow. Very excited! Can't wait to share it with everyone. #business #innovation #excited"
                ],
                "Statement": [
                    "Amazing sunset at the beach!",
                    "Perfect end to a great day.",
                    "#sunset #beach #happiness",
                    "",
                    "New product launch tomorrow.",
                    "Very excited!",
                    "Can't wait to share it with everyone.",
                    "#business #innovation #excited"
                ]
            }
            example_output_df = pd.DataFrame(example_output)
            example_output_df = example_output_df[example_output_df['Statement'] != ""]  # Remove empty rows
            st.dataframe(example_output_df, use_container_width=True)
        
        # Footer with fun facts
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>🎉 Made with ❤️ using Streamlit | 🚀 Transform your data, transform your insights!</p>
            <p>💡 Pro tip: The more creative your text, the more fun the transformation becomes!</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
