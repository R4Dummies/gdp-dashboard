"""
Enhanced Text Transformation Streamlit App
Transform your text data into sentence-level analysis format with improved UI/UX
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re
import io
import json
from typing import List, Dict, Any
from datetime import datetime

# Try to import nltk, but don't fail if it's not available
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    st.warning("⚠️ NLTK not available. Using fallback sentence tokenizer.")

# Custom CSS for better styling
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
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
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
    
    .sidebar .stSelectbox > div > div {
        background-color: #f1f3f6;
    }
    
    .uploadedfile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Setup sentence tokenizer with improved fallback
@st.cache_resource
def setup_sentence_tokenizer():
    """Setup sentence tokenizer with robust fallback"""
    if NLTK_AVAILABLE:
        try:
            nltk.download('punkt_tab', quiet=True)
            nltk.download('punkt', quiet=True)
            from nltk.tokenize import sent_tokenize
            # Test if it works
            sent_tokenize("Test sentence.")
            return sent_tokenize, "NLTK"
        except Exception as e:
            st.info(f"NLTK setup failed: {str(e)}")
    
    # Improved fallback tokenizer
    def advanced_sent_tokenize(text):
        """Advanced sentence tokenizer without NLTK"""
        if not text:
            return []
        
        # Split on sentence endings, but be smart about abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 2:  # Ignore very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    return advanced_sent_tokenize, "Advanced Fallback"

# Default cleaning patterns - can be customized by users
DEFAULT_PATTERNS = {
    "urls": r'http[s]?://\S+',
    "hashtags": r'#\w+',
    "mentions": r'@\w+',
    "emojis": r'[^\w\s.,!?\'"()-]',
    "extra_whitespace": r'\s+'
}

def extract_hashtags(text: str) -> str:
    """Extract hashtags from text and return them as a single string"""
    if not isinstance(text, str):
        return ""
    
    hashtags = re.findall(r'#\w+', text)
    return ' '.join(hashtags) if hashtags else ""

def clean_text(text: str, patterns: Dict[str, str] = None) -> str:
    """Clean text using customizable patterns"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    if patterns is None:
        patterns = DEFAULT_PATTERNS
    
    # Apply cleaning patterns
    for pattern_name, pattern in patterns.items():
        if pattern_name == "extra_whitespace":
            text = re.sub(pattern, ' ', text)
        else:
            text = re.sub(pattern, '', text)
    
    return text.strip()

def is_punctuation_only(text: str) -> bool:
    """Check if text contains only punctuation marks and whitespace"""
    if not text:
        return True
    cleaned = re.sub(r'\s+', '', text)
    if not cleaned:
        return True
    return bool(re.match(r'^[^\w#@]+$', cleaned))

def split_into_sentences(text: str, sent_tokenize) -> List[str]:
    """Split text into sentences using tokenizer"""
    if not text:
        return []
    
    if text[-1] not in '.!?':
        text = text + '.'
    
    sentences = sent_tokenize(text)
    return [sent.strip() for sent in sentences if sent.strip() and not is_punctuation_only(sent.strip())]

def transform_data(df: pd.DataFrame, id_column: str, context_column: str, 
                  include_hashtags: bool = True, custom_patterns: Dict[str, str] = None) -> pd.DataFrame:
    """Transform dataframe into sentence-level data with progress tracking"""
    sent_tokenize, tokenizer_type = setup_sentence_tokenizer()
    transformed_rows = []
    
    # Progress bar
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
            
        # Clean the context text
        cleaned_context = clean_text(context, custom_patterns)
        
        # Extract hashtags if requested
        hashtags = extract_hashtags(context) if include_hashtags else ""
        
        # Split into sentences
        sentences = split_into_sentences(cleaned_context, sent_tokenize)
        
        # Add hashtags as a separate sentence if they exist
        if hashtags and not is_punctuation_only(hashtags):
            sentences.append(hashtags)
        
        # Create rows for each sentence
        for sentence_id, sentence in enumerate(sentences, 1):
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

def create_fireworks_effect():
    """Create a fireworks animation effect"""
    return """
    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 9999;" id="fireworks-container"></div>
    <script>
    function createFirework(x, y) {
        const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff'];
        const container = document.getElementById('fireworks-container');
        if (!container) return;
        
        // Main firework explosion
        const firework = document.createElement('div');
        firework.style.position = 'absolute';
        firework.style.left = x + 'px';
        firework.style.top = y + 'px';
        firework.style.width = '4px';
        firework.style.height = '4px';
        firework.style.borderRadius = '50%';
        firework.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
        firework.style.boxShadow = '0 0 20px currentColor';
        
        container.appendChild(firework);
        
        // Animate main firework
        let scale = 0;
        let opacity = 1;
        const animateFirework = () => {
            scale += 0.8;
            opacity -= 0.05;
            firework.style.transform = `scale(${scale})`;
            firework.style.opacity = opacity;
            
            if (opacity > 0) {
                requestAnimationFrame(animateFirework);
            } else {
                if (firework.parentNode) firework.parentNode.removeChild(firework);
            }
        };
        requestAnimationFrame(animateFirework);
        
        // Create sparks
        for (let i = 0; i < 12; i++) {
            const spark = document.createElement('div');
            spark.style.position = 'absolute';
            spark.style.left = x + 'px';
            spark.style.top = y + 'px';
            spark.style.width = '2px';
            spark.style.height = '2px';
            spark.style.borderRadius = '50%';
            spark.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
            spark.style.boxShadow = '0 0 10px currentColor';
            
            container.appendChild(spark);
            
            const angle = (i * 30) * Math.PI / 180;
            const distance = 100 + Math.random() * 50;
            const dx = Math.cos(angle) * distance;
            const dy = Math.sin(angle) * distance;
            
            let sparkX = 0;
            let sparkY = 0;
            let sparkOpacity = 1;
            let sparkScale = 1;
            
            const animateSpark = () => {
                sparkX += dx * 0.02;
                sparkY += dy * 0.02;
                sparkOpacity -= 0.015;
                sparkScale -= 0.02;
                
                spark.style.transform = `translate(${sparkX}px, ${sparkY}px) scale(${sparkScale})`;
                spark.style.opacity = sparkOpacity;
                
                if (sparkOpacity > 0) {
                    requestAnimationFrame(animateSpark);
                } else {
                    if (spark.parentNode) spark.parentNode.removeChild(spark);
                }
            };
            requestAnimationFrame(animateSpark);
        }
    }
    
    function launchFireworks() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        for (let i = 0; i < 5; i++) {
            setTimeout(() => {
                const x = Math.random() * (width - 200) + 100;
                const y = Math.random() * (height * 0.6) + 50;
                createFirework(x, y);
            }, i * 300);
        }
        
        // Clean up container after animation
        setTimeout(() => {
            const container = document.getElementById('fireworks-container');
            if (container) {
                container.innerHTML = '';
            }
        }, 4000);
    }
    
    // Launch immediately
    setTimeout(launchFireworks, 100);
    </script>
    """
    """Display a nice summary of the dataframe"""
    st.markdown(f"### {title}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Rows", f"{len(df):,}")
    with col2:
        st.metric("📋 Columns", len(df.columns))
    with col3:
        st.metric("💾 Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    with col4:
        null_count = df.isnull().sum().sum()
        st.metric("❌ Null Values", f"{null_count:,}")

def display_data_summary(df: pd.DataFrame, title: str):
    """Create an expandable pattern editor for text cleaning"""
    with st.expander("🔧 Advanced: Customize Text Cleaning Patterns", expanded=False):
        st.markdown("**Modify the regular expressions used for text cleaning:**")
        
        patterns = {}
        col1, col2 = st.columns(2)
        
        with col1:
            patterns["urls"] = st.text_input(
                "URLs Pattern", 
                value=DEFAULT_PATTERNS["urls"],
                help="Pattern to remove URLs"
            )
            patterns["hashtags"] = st.text_input(
                "Hashtags Pattern", 
                value=DEFAULT_PATTERNS["hashtags"],
                help="Pattern to remove hashtags (when not including them separately)"
            )
            patterns["mentions"] = st.text_input(
                "Mentions Pattern", 
                value=DEFAULT_PATTERNS["mentions"],
                help="Pattern to remove @mentions"
            )
        
        with col2:
            patterns["emojis"] = st.text_input(
                "Emojis/Special Chars", 
                value=DEFAULT_PATTERNS["emojis"],
                help="Pattern to remove emojis and special characters"
            )
            patterns["extra_whitespace"] = st.text_input(
                "Extra Whitespace", 
                value=DEFAULT_PATTERNS["extra_whitespace"],
                help="Pattern to normalize whitespace"
            )
        
        # Test pattern functionality
        test_text = st.text_area(
            "Test your patterns:", 
            value="Check out this link http://example.com @user #hashtag 🎉 Multiple   spaces!",
            help="Enter text to see how your patterns will clean it"
        )
        
        if st.button("Test Patterns"):
            try:
                cleaned = clean_text(test_text, patterns)
                st.success(f"**Original:** {test_text}")
                st.success(f"**Cleaned:** {cleaned}")
            except Exception as e:
                st.error(f"Pattern error: {str(e)}")
        
        return patterns

def main():
    st.set_page_config(
        page_title="Enhanced Text Transformation App",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Enhanced Text Transformation App</h1>
        <p>Transform your text data into sentence-level analysis format with advanced features</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload with custom styling
        uploaded_file = st.file_uploader(
            "📁 Upload your CSV file",
            type=['csv'],
            help="Supported formats: CSV files only"
        )
        
        # Show file info if uploaded
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / 1024
            st.info(f"📄 **{uploaded_file.name}**\n\n💾 Size: {file_size:.1f} KB")
    
    if uploaded_file is not None:
        try:
            # Read and display file info
            df = pd.read_csv(uploaded_file)
            
            # Data summary
            display_data_summary(df, "📊 Uploaded Data Summary")
            
            # Show data preview with better formatting
            with st.expander("👀 Data Preview", expanded=True):
                st.dataframe(
                    df.head(10), 
                    use_container_width=True,
                    height=300
                )
            
            # Sidebar configuration continues
            with st.sidebar:
                st.markdown("---")
                st.subheader("🎯 Column Mapping")
                
                columns = list(df.columns)
                
                id_column = st.selectbox(
                    "🆔 Select ID Column",
                    options=columns,
                    help="Column containing unique identifiers"
                )
                
                context_column = st.selectbox(
                    "📝 Select Context Column", 
                    options=columns,
                    help="Column containing text to transform"
                )
                
                st.markdown("---")
                st.subheader("⚡ Processing Options")
                
                include_hashtags = st.checkbox(
                    "Include hashtags as separate sentences",
                    value=True,
                    help="Extract hashtags and create separate sentence entries"
                )
                
                use_custom_patterns = st.checkbox(
                    "Use custom cleaning patterns",
                    value=False,
                    help="Enable advanced pattern customization"
                )
            
            # Pattern editor (main area)
            custom_patterns = DEFAULT_PATTERNS
            if use_custom_patterns:
                custom_patterns = create_pattern_editor()
            
            # Column validation and preview
            if id_column and context_column:
                if id_column == context_column:
                    st.error("❌ ID and Context columns must be different!")
                else:
                    # Show sample of selected columns
                    st.markdown("### 🔍 Selected Columns Preview")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**🆔 ID Column: `{id_column}`**")
                        st.write(df[id_column].head())
                    
                    with col2:
                        st.markdown(f"**📝 Context Column: `{context_column}`**")
                        st.write(df[context_column].head())
                    
                    # Transform button
                    if st.button("🚀 Transform Data", type="primary", use_container_width=True):
                        start_time = datetime.now()
                        
                        try:
                            with st.spinner("🔄 Transforming your data..."):
                                transformed_df = transform_data(
                                    df, id_column, context_column, 
                                    include_hashtags, custom_patterns
                                )
                            
                            end_time = datetime.now()
                            processing_time = (end_time - start_time).total_seconds()
                            
                            if len(transformed_df) > 0:
                                # 🎆 FIREWORKS EFFECT! 🎆
                                components.html(create_fireworks_effect(), height=200)
                                st.success(f"✅ Transformation completed in {processing_time:.2f} seconds!")
                                
                                # Results summary
                                display_data_summary(transformed_df, "📈 Transformation Results")
                                
                                # Additional metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("📊 Original Records", len(df))
                                with col2:
                                    st.metric("📝 Generated Sentences", len(transformed_df))
                                with col3:
                                    avg_sentences = len(transformed_df) / len(df) if len(df) > 0 else 0
                                    st.metric("📈 Avg Sentences/Record", f"{avg_sentences:.1f}")
                                with col4:
                                    avg_words = transformed_df['Word_Count'].mean() if 'Word_Count' in transformed_df.columns else 0
                                    st.metric("📏 Avg Words/Sentence", f"{avg_words:.1f}")
                                
                                # Show results with tabs
                                tab1, tab2, tab3 = st.tabs(["📋 Preview", "📊 Statistics", "📁 Column Info"])
                                
                                with tab1:
                                    st.dataframe(
                                        transformed_df.head(20), 
                                        use_container_width=True,
                                        height=400
                                    )
                                
                                with tab2:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Character Count Distribution**")
                                        st.bar_chart(transformed_df['Character_Count'].value_counts().head(10))
                                    with col2:
                                        st.markdown("**Word Count Distribution**")
                                        st.bar_chart(transformed_df['Word_Count'].value_counts().head(10))
                                
                                with tab3:
                                    column_info = []
                                    for col in transformed_df.columns:
                                        column_info.append({
                                            "Column": col,
                                            "Type": str(transformed_df[col].dtype),
                                            "Non-null": f"{transformed_df[col].count():,}",
                                            "Unique Values": f"{transformed_df[col].nunique():,}",
                                            "Sample": str(transformed_df[col].iloc[0]) if len(transformed_df) > 0 else "N/A"
                                        })
                                    st.dataframe(pd.DataFrame(column_info), use_container_width=True)
                                
                                # Download section
                                st.markdown("---")
                                st.markdown("### 💾 Download Results")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # CSV download
                                    csv_buffer = io.StringIO()
                                    transformed_df.to_csv(csv_buffer, index=False)
                                    csv_data = csv_buffer.getvalue()
                                    
                                    st.download_button(
                                        label="📥 Download as CSV",
                                        data=csv_data,
                                        file_name=f"transformed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                
                                with col2:
                                    # JSON download
                                    json_data = transformed_df.to_json(orient='records', indent=2)
                                    st.download_button(
                                        label="📥 Download as JSON",
                                        data=json_data,
                                        file_name=f"transformed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        mime="application/json",
                                        use_container_width=True
                                    )
                                
                            else:
                                st.warning("⚠️ No sentences were generated. Please check your data and settings.")
                                
                        except Exception as e:
                            st.error(f"❌ Error during transformation: {str(e)}")
                            st.info("💡 Try checking your column selections and data format.")
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("💡 Please ensure your file is in valid CSV format with proper encoding.")
    
    else:
        # Welcome section with instructions
        st.markdown("""
        <div class="feature-card">
            <h3>👋 Welcome to the Enhanced Text Transformation App!</h3>
            <p>Upload your CSV file to get started with advanced text processing capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🎯 Smart Processing</h4>
                <ul>
                    <li>Intelligent sentence splitting</li>
                    <li>Hashtag extraction</li>
                    <li>Customizable text cleaning</li>
                    <li>Progress tracking</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>📊 Rich Analytics</h4>
                <ul>
                    <li>Data quality metrics</li>
                    <li>Processing statistics</li>
                    <li>Word/character counts</li>
                    <li>Distribution charts</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h4>💾 Multiple Exports</h4>
                <ul>
                    <li>CSV format</li>
                    <li>JSON format</li>
                    <li>Timestamped filenames</li>
                    <li>Detailed column info</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Instructions
        st.markdown("### 📋 How to Use")
        
        instructions = [
            "📁 **Upload**: Select your CSV file using the sidebar uploader",
            "🎯 **Configure**: Choose your ID and Context columns from the dropdown menus", 
            "⚙️ **Customize**: Optionally modify text cleaning patterns and processing options",
            "🚀 **Transform**: Click the transform button to process your data",
            "📊 **Analyze**: Review the results, statistics, and data quality metrics",
            "💾 **Download**: Export your transformed data in CSV or JSON format"
        ]
        
        for i, instruction in enumerate(instructions, 1):
            st.markdown(f"{i}. {instruction}")
        
        # Example section
        st.markdown("---")
        st.markdown("### 💡 Example Transformation")
        
        example_input = pd.DataFrame({
            "post_id": ["POST_001", "POST_002"],
            "caption": [
                "Amazing sunset at the beach! 🌅 #summer #vacation #blessed",
                "New product launch today. Very excited about this! #startup #innovation"
            ]
        })
        
        example_output = pd.DataFrame({
            "ID": ["POST_001", "POST_001", "POST_002", "POST_002", "POST_002"],
            "Sentence_ID": [1, 2, 1, 2, 3],
            "Context": [
                "Amazing sunset at the beach! 🌅 #summer #vacation #blessed",
                "Amazing sunset at the beach! 🌅 #summer #vacation #blessed",
                "New product launch today. Very excited about this! #startup #innovation",
                "New product launch today. Very excited about this! #startup #innovation", 
                "New product launch today. Very excited about this! #startup #innovation"
            ],
            "Statement": [
                "Amazing sunset at the beach!",
                "#summer #vacation #blessed",
                "New product launch today.",
                "Very excited about this!",
                "#startup #innovation"
            ],
            "Character_Count": [29, 27, 26, 25, 20],
            "Word_Count": [5, 3, 4, 4, 2]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📥 Input Data:**")
            st.dataframe(example_input, use_container_width=True)
        
        with col2:
            st.markdown("**📤 Output Data:**")
            st.dataframe(example_output, use_container_width=True)

if __name__ == "__main__":
    main()
