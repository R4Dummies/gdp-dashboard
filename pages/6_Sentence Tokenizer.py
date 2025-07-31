"""
NLTK-Free Text Transformation Streamlit App
No external NLP dependencies required
"""

import streamlit as st
import pandas as pd
import re
import io
import json
from typing import List, Dict
from datetime import datetime

# Custom CSS for styling
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    
    .fireworks {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 9999;
    }
    
    .firework {
        position: absolute;
        width: 4px;
        height: 4px;
        border-radius: 50%;
        animation: firework 1.5s ease-out forwards;
    }
    
    @keyframes firework {
        0% {
            transform: scale(0);
            opacity: 1;
        }
        15% {
            transform: scale(1);
        }
        100% {
            transform: scale(20);
            opacity: 0;
        }
    }
    
    .spark {
        position: absolute;
        width: 2px;
        height: 2px;
        border-radius: 50%;
        animation: spark 2s ease-out forwards;
    }
    
    @keyframes spark {
        0% {
            transform: translate(0, 0) scale(1);
            opacity: 1;
        }
        100% {
            transform: translate(var(--dx), var(--dy)) scale(0);
            opacity: 0;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def create_fireworks_effect():
    """Create a spectacular fireworks animation"""
    return """
    <div class="fireworks" id="fireworks-container"></div>
    <script>
    function createFirework(x, y) {
        const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff', '#fd79a8'];
        const container = document.getElementById('fireworks-container');
        
        // Main firework burst
        const firework = document.createElement('div');
        firework.className = 'firework';
        firework.style.left = x + 'px';
        firework.style.top = y + 'px';
        firework.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
        firework.style.boxShadow = `0 0 20px ${colors[Math.floor(Math.random() * colors.length)]}`;
        container.appendChild(firework);
        
        // Create beautiful sparks
        for (let i = 0; i < 15; i++) {
            const spark = document.createElement('div');
            spark.className = 'spark';
            spark.style.left = x + 'px';
            spark.style.top = y + 'px';
            spark.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
            spark.style.boxShadow = `0 0 10px ${colors[Math.floor(Math.random() * colors.length)]}`;
            
            const angle = (i * 24) * Math.PI / 180;
            const distance = 80 + Math.random() * 60;
            const dx = Math.cos(angle) * distance;
            const dy = Math.sin(angle) * distance;
            
            spark.style.setProperty('--dx', dx + 'px');
            spark.style.setProperty('--dy', dy + 'px');
            
            container.appendChild(spark);
            
            setTimeout(() => {
                if (spark.parentNode) spark.parentNode.removeChild(spark);
            }, 2000);
        }
        
        setTimeout(() => {
            if (firework.parentNode) firework.parentNode.removeChild(firework);
        }, 1500);
    }
    
    function launchFireworks() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        // Launch multiple fireworks with timing
        for (let i = 0; i < 6; i++) {
            setTimeout(() => {
                const x = Math.random() * (width - 200) + 100;
                const y = Math.random() * (height * 0.5) + 50;
                createFirework(x, y);
            }, i * 300);
        }
        
        // Second wave
        setTimeout(() => {
            for (let i = 0; i < 4; i++) {
                setTimeout(() => {
                    const x = Math.random() * (width - 200) + 100;
                    const y = Math.random() * (height * 0.6) + 50;
                    createFirework(x, y);
                }, i * 200);
            }
        }, 1000);
        
        // Clean up
        setTimeout(() => {
            const container = document.getElementById('fireworks-container');
            if (container) container.innerHTML = '';
        }, 5000);
    }
    
    launchFireworks();
    </script>
    """

def advanced_sentence_tokenize(text: str) -> List[str]:
    """
    Advanced sentence tokenizer without external dependencies
    Handles common abbreviations and edge cases
    """
    if not text or not text.strip():
        return []
    
    # Common abbreviations that shouldn't trigger sentence breaks
    abbreviations = {
        'dr', 'mr', 'mrs', 'ms', 'prof', 'vs', 'etc', 'inc', 'ltd', 'co',
        'corp', 'dept', 'govt', 'univ', 'assn', 'bros', 'rep', 'sen'
    }
    
    # Protect abbreviations by temporarily replacing periods
    protected_text = text
    for abbr in abbreviations:
        pattern = rf'\b{re.escape(abbr)}\.(?=\s+[a-z])'
        protected_text = re.sub(pattern, f'{abbr}<!PERIOD!>', protected_text, flags=re.IGNORECASE)
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected_text)
    
    # Clean up and restore periods
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.replace('<!PERIOD!>', '.')
        sentence = sentence.strip()
        
        # Filter out very short or empty sentences
        if sentence and len(sentence.split()) >= 2:
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences

def extract_hashtags(text: str) -> str:
    """Extract hashtags from text"""
    if not isinstance(text, str):
        return ""
    hashtags = re.findall(r'#\w+', text)
    return ' '.join(hashtags) if hashtags else ""

def clean_text(text: str, remove_urls: bool = True, remove_mentions: bool = True) -> str:
    """Clean text with configurable options"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    
    # Remove mentions
    if remove_mentions:
        text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (they'll be added separately if needed)
    text = re.sub(r'#\w+', '', text)
    
    # Remove emojis and special characters (keep basic punctuation)
    text = re.sub(r'[^\w\s.,!?\'"()-]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def is_meaningful_text(text: str) -> bool:
    """Check if text contains meaningful content"""
    if not text:
        return False
    
    # Remove punctuation and whitespace
    cleaned = re.sub(r'[^\w]', '', text)
    
    # Must have at least some letters
    return bool(cleaned and re.search(r'[a-zA-Z]', cleaned))

def transform_data(df: pd.DataFrame, id_column: str, context_column: str, 
                  include_hashtags: bool = True) -> pd.DataFrame:
    """Transform dataframe into sentence-level data"""
    transformed_rows = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_rows = len(df)
    
    for idx, (_, row) in enumerate(df.iterrows()):
        # Update progress
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f'Processing row {idx + 1} of {total_rows}...')
        
        row_id = row[id_column] if pd.notna(row[id_column]) else f"ROW_{idx+1}"
        context = row[context_column] if pd.notna(row[context_column]) else ""
        
        if not context:
            continue
        
        # Extract hashtags before cleaning
        hashtags = extract_hashtags(context) if include_hashtags else ""
        
        # Clean the text
        cleaned_text = clean_text(context)
        
        # Split into sentences
        sentences = advanced_sentence_tokenize(cleaned_text)
        
        # Add hashtags as separate sentence if they exist
        if hashtags and is_meaningful_text(hashtags):
            sentences.append(hashtags)
        
        # Create rows for each sentence
        for sentence_id, sentence in enumerate(sentences, 1):
            if is_meaningful_text(sentence):
                transformed_rows.append({
                    'ID': row_id,
                    'Sentence_ID': sentence_id,
                    'Context': context,
                    'Statement': sentence,
                    'Character_Count': len(sentence),
                    'Word_Count': len(sentence.split())
                })
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(transformed_rows)

def main():
    st.set_page_config(
        page_title="Text Transformation App",
        page_icon="🚀",
        layout="wide"
    )
    
    load_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Text Transformation App</h1>
        <p>Transform your text data into sentence-level format (No external dependencies)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        uploaded_file = st.file_uploader(
            "📁 Upload CSV file",
            type=['csv'],
            help="Upload your CSV file containing text data"
        )
        
        if uploaded_file:
            file_size = len(uploaded_file.getvalue()) / 1024
            st.info(f"📄 **{uploaded_file.name}**\n💾 Size: {file_size:.1f} KB")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Rows", f"{len(df):,}")
            with col2:
                st.metric("📋 Columns", len(df.columns))
            with col3:
                st.metric("💾 Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Data preview
            with st.expander("👀 Data Preview", expanded=True):
                st.dataframe(df.head(), use_container_width=True)
            
            # Column selection
            with st.sidebar:
                st.markdown("---")
                st.subheader("🎯 Column Selection")
                
                columns = list(df.columns)
                
                id_column = st.selectbox(
                    "🆔 ID Column",
                    options=columns,
                    help="Column with unique identifiers"
                )
                
                context_column = st.selectbox(
                    "📝 Text Column",
                    options=columns,
                    help="Column containing text to process"
                )
                
                st.markdown("---")
                st.subheader("⚙️ Options")
                
                include_hashtags = st.checkbox(
                    "Include hashtags separately",
                    value=True,
                    help="Extract hashtags as separate sentences"
                )
            
            # Validation and transformation
            if id_column and context_column:
                if id_column == context_column:
                    st.error("❌ ID and Text columns must be different!")
                else:
                    # Show selected columns
                    st.markdown("### 🔍 Selected Columns")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**🆔 ID: `{id_column}`**")
                        st.write(df[id_column].head(3))
                    
                    with col2:
                        st.markdown(f"**📝 Text: `{context_column}`**")
                        st.write(df[context_column].head(3))
                    
                    # Transform button
                    if st.button("🚀 Transform Data", type="primary", use_container_width=True):
                        start_time = datetime.now()
                        
                        try:
                            transformed_df = transform_data(df, id_column, context_column, include_hashtags)
                            
                            end_time = datetime.now()
                            processing_time = (end_time - start_time).total_seconds()
                            
                            if len(transformed_df) > 0:
                                # 🎆 SPECTACULAR FIREWORKS! 🎆
                                st.components.v1.html(create_fireworks_effect(), height=0)
                                st.success(f"✅ Completed in {processing_time:.2f} seconds!")
                                
                                # Results metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("📊 Original", len(df))
                                with col2:
                                    st.metric("📝 Sentences", len(transformed_df))
                                with col3:
                                    avg_sentences = len(transformed_df) / len(df)
                                    st.metric("📈 Avg/Record", f"{avg_sentences:.1f}")
                                with col4:
                                    avg_words = transformed_df['Word_Count'].mean()
                                    st.metric("📏 Avg Words", f"{avg_words:.1f}")
                                
                                # Results display
                                st.markdown("### 📋 Results")
                                st.dataframe(transformed_df.head(15), use_container_width=True)
                                
                                # Download
                                csv_buffer = io.StringIO()
                                transformed_df.to_csv(csv_buffer, index=False)
                                
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv_buffer.getvalue(),
                                    file_name=f"transformed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            else:
                                st.warning("⚠️ No sentences generated. Check your data.")
                        
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
        except Exception as e:
            st.error(f"❌ File error: {str(e)}")
            st.info("💡 Ensure your file is valid CSV format")
    
    else:
        # Instructions
        st.markdown("""
        <div class="feature-card">
            <h3>👋 Welcome!</h3>
            <p>This app transforms text data into sentence-level format without requiring external NLP libraries.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **📁 Upload** your CSV file using the sidebar
        2. **🎯 Select** ID and text columns from dropdowns  
        3. **⚙️ Configure** processing options
        4. **🚀 Transform** your data with one click
        5. **📥 Download** results as CSV
        """)
        
        # Example
        st.markdown("### 💡 Example")
        example_input = pd.DataFrame({
            "id": ["1", "2"],
            "text": [
                "Great day! #happy #sunshine",
                "New project starting. Excited! #work"
            ]
        })
        
        example_output = pd.DataFrame({
            "ID": ["1", "1", "2", "2", "2"],
            "Sentence_ID": [1, 2, 1, 2, 3],
            "Context": [
                "Great day! #happy #sunshine",
                "Great day! #happy #sunshine",
                "New project starting. Excited! #work",
                "New project starting. Excited! #work",
                "New project starting. Excited! #work"
            ],
            "Statement": [
                "Great day!",
                "#happy #sunshine", 
                "New project starting.",
                "Excited!",
                "#work"
            ]
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Input:**")
            st.dataframe(example_input, use_container_width=True)
        with col2:
            st.markdown("**Output:**")
            st.dataframe(example_output, use_container_width=True)

if __name__ == "__main__":
    main()
