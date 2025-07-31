import streamlit as st
import pandas as pd
import io

def find_best_match(columns, keywords):
    for keyword in keywords:
        for col in columns:
            if keyword.lower() in col.lower():
                return col
    return columns[0] if columns else None

def get_default_columns(df):
    columns = df.columns.tolist()
    
    id_keywords = ['id', 'chatid', 'call_id', 'callid', 'conversation_id', 'conversationid']
    turn_keywords = ['turn', 'step', 'sequence', 'order', 'number']
    speaker_keywords = ['speaker', 'role', 'person', 'user', 'participant']
    transcript_keywords = ['transcript', 'text', 'message', 'statement', 'content', 'utterance']
    
    return {
        'id': find_best_match(columns, id_keywords),
        'turn': find_best_match(columns, turn_keywords),
        'speaker': find_best_match(columns, speaker_keywords),
        'transcript': find_best_match(columns, transcript_keywords)
    }

def create_rolling_context(df, id_col, turn_col, speaker_col, transcript_col):
    result = []
    
    for call_id in df[id_col].unique():
        call_data = df[df[id_col] == call_id].sort_values(turn_col)
        
        for i, row in call_data.iterrows():
            if row[speaker_col] == 'salesperson':
                context_lines = []
                for j, prev_row in call_data[call_data[turn_col] < row[turn_col]].iterrows():
                    context_lines.append(f"Turn {prev_row[turn_col]} ({prev_row[speaker_col]}): {prev_row[transcript_col]}")
                
                context_lines.append(f"Turn {row[turn_col]} ({row[speaker_col]}): {row[transcript_col]}")
                context = '\n '.join(context_lines)
                
                result.append({
                    'ID': call_id,
                    'Turn': row[turn_col],
                    'Speaker': row[speaker_col],
                    'Context': context,
                    'Statement': row[transcript_col]
                })
    
    return pd.DataFrame(result)

# 🎨 Page Configuration
st.set_page_config(
    page_title="Rolling Context Processor",
    page_icon="🔄",
    layout="wide"
)

# 🎯 Header Section
st.title("🔄 Rolling Context Window Processor")
st.markdown("### 📊 Transform your conversation data with intelligent context windows")

# 📝 Info Section
with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
    **This tool processes conversation datasets by:**
    
    🎯 **Auto-detecting** column mappings for your data  
    🔄 **Creating rolling context** windows for each conversation  
    👤 **Filtering salesperson** statements with full conversation history  
    📥 **Exporting processed** data as downloadable CSV  
    
    Simply upload your CSV file and let the magic happen! ✨
    """)

# 📁 File Upload Section
st.markdown("---")
st.markdown("## 📁 Upload Your Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file 📄", 
    type="csv",
    help="Upload a CSV file containing conversation data with columns for ID, turn, speaker, and transcript"
)

if uploaded_file is not None:
    # 📊 Loading and Preview Section
    with st.spinner("🔍 Loading your data..."):
        df = pd.read_csv(uploaded_file)
    
    st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns!")
    
    st.markdown("## 👀 Data Preview")
    
    # 📈 Data Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Rows", len(df))
    with col2:
        st.metric("📋 Columns", len(df.columns))
    with col3:
        unique_ids = df.iloc[:, 0].nunique() if len(df.columns) > 0 else 0
        st.metric("🆔 Unique IDs", unique_ids)
    with col4:
        st.metric("💾 File Size", f"{uploaded_file.size / 1024:.1f} KB")
    
    # 📊 Data Preview Table
    st.dataframe(df.head(10), use_container_width=True)
    
    # ⚙️ Column Mapping Section
    st.markdown("---")
    st.markdown("## ⚙️ Column Mapping Configuration")
    
    st.info("🎯 We've auto-detected the best column matches. You can adjust them below if needed!")
    
    defaults = get_default_columns(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Identification & Structure")
        id_col = st.selectbox(
            "🆔 ID Column (Call/Conversation ID)", 
            df.columns, 
            index=df.columns.tolist().index(defaults['id']) if defaults['id'] else 0,
            help="Select the column that uniquely identifies each conversation or call"
        )
        turn_col = st.selectbox(
            "🔄 Turn Column",
            df.columns,
            index=df.columns.tolist().index(defaults['turn']) if defaults['turn'] else 0,
            help="Select the column that indicates the sequence/order of statements"
        )
    
    with col2:
        st.markdown("### 👥 Content & Speakers")
        speaker_col = st.selectbox(
            "👤 Speaker Column", 
            df.columns,
            index=df.columns.tolist().index(defaults['speaker']) if defaults['speaker'] else 0,
            help="Select the column that identifies who is speaking (should include 'salesperson')"
        )
        transcript_col = st.selectbox(
            "💬 Statement/Transcript Column", 
            df.columns,
            index=df.columns.tolist().index(defaults['transcript']) if defaults['transcript'] else 0,
            help="Select the column containing the actual conversation text"
        )
    
    # 🔧 Processing Section
    st.markdown("---")
    
    # Preview selected mapping
    with st.expander("🔍 Preview Column Mapping", expanded=False):
        preview_data = {
            "Field": ["🆔 ID", "🔄 Turn", "👤 Speaker", "💬 Transcript"],
            "Selected Column": [id_col, turn_col, speaker_col, transcript_col],
            "Sample Data": [
                str(df[id_col].iloc[0]) if len(df) > 0 else "N/A",
                str(df[turn_col].iloc[0]) if len(df) > 0 else "N/A", 
                str(df[speaker_col].iloc[0]) if len(df) > 0 else "N/A",
                str(df[transcript_col].iloc[0])[:50] + "..." if len(df) > 0 and len(str(df[transcript_col].iloc[0])) > 50 else str(df[transcript_col].iloc[0]) if len(df) > 0 else "N/A"
            ]
        }
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
    
    # 🚀 Process Button
    if st.button("🚀 Process Data", type="primary", use_container_width=True):
        with st.spinner("⚡ Processing your data... This may take a moment!"):
            try:
                processed_df = create_rolling_context(df, id_col, turn_col, speaker_col, transcript_col)
                
                if len(processed_df) > 0:
                    st.success(f"🎉 Successfully processed {len(processed_df)} salesperson statements!")
                    
                    # 📊 Results Section
                    st.markdown("---")
                    st.markdown("## 📊 Processing Results")
                    
                    # Results metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("✅ Processed Statements", len(processed_df))
                    with col2:
                        avg_context_length = processed_df['Context'].str.len().mean()
                        st.metric("📏 Avg Context Length", f"{avg_context_length:.0f} chars")
                    with col3:
                        unique_conversations = processed_df['ID'].nunique()
                        st.metric("💬 Conversations", unique_conversations)
                    
                    # 👀 Results Preview
                    st.markdown("### 👀 Processed Data Preview")
                    st.dataframe(processed_df.head(), use_container_width=True)
                    
                    # 📥 Download Section
                    st.markdown("### 📥 Download Your Results")
                    
                    csv_buffer = io.StringIO()
                    processed_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.download_button(
                            label="📥 Download Processed CSV",
                            data=csv_data,
                            file_name="processed_rolling_window.csv",
                            mime="text/csv",
                            type="primary",
                            use_container_width=True
                        )
                    with col2:
                        st.info("💡 **Tip:** The processed file contains rolling context windows for all salesperson statements!")
                
                else:
                    st.warning("⚠️ No salesperson statements found in your data. Please check your speaker column values.")
                    
            except Exception as e:
                st.error(f"❌ An error occurred while processing: {str(e)}")
                st.info("💡 Please check your column mappings and data format.")

else:
    # 🎨 Welcome Section
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h3>🎯 Ready to get started?</h3>
            <p style='font-size: 1.1em; color: #666;'>
                Upload your conversation CSV file above to begin processing! 
                <br><br>
                📄 <strong>Expected format:</strong> CSV with columns for conversation ID, turn number, speaker, and transcript text.
            </p>
        </div>
        """, unsafe_allow_html=True)

# 🔗 Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9em;'>
    🔄 Rolling Context Window Processor | Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
