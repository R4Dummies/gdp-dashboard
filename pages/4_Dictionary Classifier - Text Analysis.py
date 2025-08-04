with col3:
            # Download filtered results
            if show_category != "All" or show_rows != "All":
                # Apply the same filters to create download data
                download_df = st.session_state.result_df.copy()
                
                if show_category != "All":
                    if show_category == "Unclassified":
                        download_df = download_df[download_df['Predicted_Category'].isna()]
                    else:
                        download_df = download_df[download_df['Predicted_Category'] == show_category]
                
                if show_rows != "All":
                    download_df = download_df.head(show_rows)
                
                if len(download_df) > 0:
                    filtered_csv = export_results_csv(download_df)
                    st.download_button(
                        label="🔍 Download Filtered",
                        data=filtered_csv,
                        file_name=f'filtered_results_{timestamp}.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                else:
                    st.button("🔍 No Filtered Data", disabled=True, use_container_width=True)
            else:
                st.write("*Apply filters to enable filtered download*")import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime

def parse_dictionary(dict_input):
    """Parse dictionary input from JSON or text format"""
    dict_text = dict_input.strip()
    categories = {}
    
    # Try parsing as JSON first
    try:
        parsed = json.loads(dict_text)
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                if isinstance(v, str):
                    v = [v]
                categories[k] = [str(x).strip() for x in v]
            return categories, None
    except json.JSONDecodeError:
        pass
    
    # Parse as text format
    try:
        for line in dict_text.splitlines():
            if ':' in line:
                cat, terms = line.split(':', 1)
                categories[cat.strip()] = [w.strip() for w in terms.split(',') if w.strip()]
        return categories, None
    except Exception as e:
        return {}, f"Error parsing dictionary: {str(e)}"

def classify_text(text, categories):
    """Classify a single text based on keyword matches"""
    if pd.isna(text):
        return None
        
    text_lower = str(text).lower()
    best_category, max_count = None, 0
    
    for category, words in categories.items():
        count = sum(text_lower.count(word.lower()) for word in words if word)
        if count > max_count:
            best_category, max_count = category, count
    
    return best_category

def export_results_csv(df):
    """Export results as CSV"""
    return df.to_csv(index=False).encode('utf-8')

def export_results_excel(df):
    """Export results as Excel"""
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Classification Results')
        return output.getvalue()
    except ImportError:
        # If openpyxl is not available, return None
        return None

def export_dictionary_json(categories):
    """Export dictionary as JSON"""
    return json.dumps(categories, indent=2).encode('utf-8')

def main():
    st.set_page_config(
        page_title="Dictionary Classifier",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Dictionary Classifier App")
    st.markdown("""
    Upload a CSV, select the column with text, and input your dictionary as either JSON or lines like:
    `Luxury: elegant, timeless, classic`.
    Then run the analysis to get category predictions based on keyword matches.
    """)
    
    # Initialize session state
    if 'df_loaded' not in st.session_state:
        st.session_state.df_loaded = None
    if 'result_df' not in st.session_state:
        st.session_state.result_df = None
    if 'categories' not in st.session_state:
        st.session_state.categories = {}
    
    # Sidebar for additional options
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Export options
        if st.session_state.result_df is not None:
            st.subheader("📥 Export Options")
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # CSV Download
            csv_data = export_results_csv(st.session_state.result_df)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f'classification_results_{timestamp}.csv',
                mime='text/csv',
                use_container_width=True
            )
            
            # Excel Download (only if openpyxl is available)
            try:
                excel_data = export_results_excel(st.session_state.result_df)
                if excel_data is not None:
                    st.download_button(
                        label="📊 Download Excel",
                        data=excel_data,
                        file_name=f'classification_results_{timestamp}.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        use_container_width=True
                    )
                else:
                    st.info("📊 Excel export unavailable (install openpyxl)")
            except Exception:
                st.info("📊 Excel export unavailable (install openpyxl)")
        
        # Dictionary export
        if st.session_state.categories:
            st.subheader("📚 Export Dictionary")
            dict_json = export_dictionary_json(st.session_state.categories)
            st.download_button(
                label="💾 Download Dictionary",
                data=dict_json,
                file_name='classification_dictionary.json',
                mime='application/json',
                use_container_width=True
            )
        
        # Advanced options
        st.subheader("🔧 Advanced Options")
        case_sensitive = st.checkbox("Case Sensitive Matching", value=False)
        show_confidence = st.checkbox("Show Match Count", value=False)
        min_matches = st.number_input("Minimum Matches Required", min_value=0, value=1)
    
    # File upload section
    st.header("📁 Upload Your Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="Upload a CSV file containing the text data you want to classify"
        )
    
    with col2:
        if uploaded_file is not None:
            st.info(f"**File:** {uploaded_file.name}\n**Size:** {uploaded_file.size} bytes")
    
    if uploaded_file is not None:
        try:
            df_loaded = pd.read_csv(uploaded_file)
            st.session_state.df_loaded = df_loaded
            st.success(f"✅ CSV uploaded successfully! Shape: {df_loaded.shape}")
            
            # Show data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df_loaded.shape[0])
            with col2:
                st.metric("Columns", df_loaded.shape[1])
            with col3:
                st.metric("Memory Usage", f"{df_loaded.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Show preview of the data
            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df_loaded.head(10))
                
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            return
    
    # Configuration section
    if st.session_state.df_loaded is not None:
        st.header("⚙️ Configuration")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            text_col = st.selectbox(
                "📝 Select Text Column:",
                st.session_state.df_loaded.columns,
                help="Choose the column containing the text you want to classify"
            )
            
            # Show column statistics
            if text_col:
                col_data = st.session_state.df_loaded[text_col]
                st.write("**Column Statistics:**")
                st.write(f"- Non-null values: {col_data.notna().sum()}")
                st.write(f"- Null values: {col_data.isna().sum()}")
                st.write(f"- Unique values: {col_data.nunique()}")
        
        with col2:
            st.write("📋 **Sample text from selected column:**")
            if text_col:
                # Show multiple samples
                sample_texts = st.session_state.df_loaded[text_col].dropna().head(3)
                for i, text in enumerate(sample_texts, 1):
                    sample_text = str(text)[:150] + ("..." if len(str(text)) > 150 else "")
                    st.text_area(f"Sample {i}:", value=sample_text, height=60, disabled=True)
        
        # Dictionary input section
        st.subheader("📚 Dictionary Configuration")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            dict_format = st.radio(
                "Choose input format:",
                ["Text Format", "JSON Format"],
                help="Text format: 'Category: word1, word2, word3'\nJSON format: {'Category': ['word1', 'word2', 'word3']}"
            )
            
            # Dictionary templates
            st.write("**Quick Templates:**")
            if st.button("📱 Tech & Business"):
                if dict_format == "JSON Format":
                    template = """{
    "Technology": ["digital", "tech", "software", "AI", "automated", "cloud", "data"],
    "Business": ["revenue", "profit", "sales", "market", "customer", "strategy"],
    "Marketing": ["brand", "campaign", "advertising", "social", "engagement"]
}"""
                else:
                    template = """Technology: digital, tech, software, AI, automated, cloud, data
Business: revenue, profit, sales, market, customer, strategy
Marketing: brand, campaign, advertising, social, engagement"""
                st.session_state.dict_template = template
            
            if st.button("😊 Sentiment Analysis"):
                if dict_format == "JSON Format":
                    template = """{
    "Positive": ["good", "great", "excellent", "amazing", "love", "best", "perfect"],
    "Negative": ["bad", "terrible", "awful", "hate", "worst", "horrible", "disappointing"],
    "Neutral": ["okay", "fine", "average", "normal", "standard"]
}"""
                else:
                    template = """Positive: good, great, excellent, amazing, love, best, perfect
Negative: bad, terrible, awful, hate, worst, horrible, disappointing
Neutral: okay, fine, average, normal, standard"""
                st.session_state.dict_template = template
        
        with col2:
            # Use template if selected, otherwise use default
            if dict_format == "Text Format":
                default_dict = getattr(st.session_state, 'dict_template', """Luxury: elegant, timeless, refined, classic, sophisticated, luxury, polished
Technology: digital, innovation, tech, software, AI, automated, smart
Nature: organic, natural, eco, green, sustainable, environment, earth""")
            else:
                default_dict = getattr(st.session_state, 'dict_template', """{
    "Luxury": ["elegant", "timeless", "refined", "classic", "sophisticated", "luxury", "polished"],
    "Technology": ["digital", "innovation", "tech", "software", "AI", "automated", "smart"],
    "Nature": ["organic", "natural", "eco", "green", "sustainable", "environment", "earth"]
}""")
            
            dict_input = st.text_area(
                "Dictionary:",
                value=default_dict,
                height=250,
                help='Enter your dictionary in the selected format'
            )
        
        # Parse and validate dictionary
        categories, error = parse_dictionary(dict_input)
        
        if error:
            st.error(f"❌ {error}")
        elif categories:
            st.session_state.categories = categories
            st.success(f"✅ Dictionary parsed successfully! Categories: {list(categories.keys())}")
            
            # Show dictionary preview with statistics
            with st.expander("🔍 Dictionary Preview & Statistics"):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Categories & Keywords:**")
                    for cat, words in categories.items():
                        st.write(f"**{cat}:** {', '.join(words)}")
                
                with col2:
                    st.write("**Statistics:**")
                    total_keywords = sum(len(words) for words in categories.values())
                    st.metric("Total Categories", len(categories))
                    st.metric("Total Keywords", total_keywords)
                    st.metric("Avg Keywords/Category", f"{total_keywords/len(categories):.1f}")
        
        # Analysis section
        if categories and not error:
            st.header("🔍 Run Analysis")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("🚀 Run Classification", type="primary", use_container_width=True):
                    with st.spinner("Classifying texts..."):
                        results = []
                        match_counts = [] if show_confidence else None
                        progress_bar = st.progress(0)
                        
                        for i, text in enumerate(st.session_state.df_loaded[text_col]):
                            if pd.isna(text):
                                results.append(None)
                                if show_confidence:
                                    match_counts.append(0)
                            else:
                                text_lower = str(text).lower() if not case_sensitive else str(text)
                                best_category, max_count = None, 0
                                
                                for category, words in categories.items():
                                    if case_sensitive:
                                        count = sum(text.count(word) for word in words if word)
                                    else:
                                        count = sum(text_lower.count(word.lower()) for word in words if word)
                                    
                                    if count >= min_matches and count > max_count:
                                        best_category, max_count = category, count
                                
                                results.append(best_category)
                                if show_confidence:
                                    match_counts.append(max_count)
                            
                            progress_bar.progress((i + 1) / len(st.session_state.df_loaded))
                        
                        # Create result dataframe
                        result_df = st.session_state.df_loaded.copy()
                        result_df['Predicted_Category'] = results
                        if show_confidence:
                            result_df['Match_Count'] = match_counts
                        
                        st.session_state.result_df = result_df
                        
                        st.success("✅ Analysis complete!")
            
            with col2:
                if st.button("🔄 Reset Results", use_container_width=True):
                    st.session_state.result_df = None
                    st.success("Results cleared!")
            
            with col3:
                if st.session_state.result_df is not None:
                    st.write(f"**Results ready!** {len(st.session_state.result_df)} rows processed")
    
    # Results section
    if st.session_state.result_df is not None:
        st.header("📊 Results")
        
        # Results summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_rows = len(st.session_state.result_df)
            st.metric("Total Rows", total_rows)
        
        with col2:
            classified_rows = len(st.session_state.result_df[st.session_state.result_df['Predicted_Category'].notna()])
            st.metric("Classified", classified_rows)
        
        with col3:
            unclassified_rows = total_rows - classified_rows
            st.metric("Unclassified", unclassified_rows)
        
        with col4:
            classification_rate = (classified_rows / total_rows) * 100 if total_rows > 0 else 0
            st.metric("Classification Rate", f"{classification_rate:.1f}%")
        
        # Category distribution
        if 'Predicted_Category' in st.session_state.result_df.columns:
            category_counts = st.session_state.result_df['Predicted_Category'].value_counts()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📈 Category Distribution")
                st.bar_chart(category_counts)
            
            with col2:
                st.subheader("📋 Category Counts")
                counts_df = category_counts.reset_index().rename(columns={'index': 'Category', 'Predicted_Category': 'Count'})
                # Fix percentage calculation with proper error handling
                try:
                    counts_df['Percentage'] = (counts_df['Count'].astype(float) / total_rows * 100).round(1)
                except:
                    counts_df['Percentage'] = 0.0
                st.dataframe(counts_df, use_container_width=True)
        
        # Download and Filter section
        st.subheader("📥 Download & Filter Results")
        
        # Download buttons row
        col1, col2, col3, col4 = st.columns(4)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with col1:
            # Download All Results as CSV
            csv_data = export_results_csv(st.session_state.result_df)
            st.download_button(
                label="📄 Download All (CSV)",
                data=csv_data,
                file_name=f'all_results_{timestamp}.csv',
                mime='text/csv',
                use_container_width=True,
                type="primary"
            )
        
        with col2:
            # Download All Results as Excel (with error handling)
            try:
                excel_data = export_results_excel(st.session_state.result_df)
                if excel_data is not None:
                    st.download_button(
                        label="📊 Download All (Excel)",
                        data=excel_data,
                        file_name=f'all_results_{timestamp}.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        use_container_width=True
                    )
                else:
                    st.button("📊 Excel Unavailable", disabled=True, use_container_width=True, 
                             help="Install openpyxl: pip install openpyxl")
            except Exception:
                st.button("📊 Excel Unavailable", disabled=True, use_container_width=True,
                         help="Install openpyxl: pip install openpyxl")
        
        with col3:
            # Download only classified results
            classified_df = st.session_state.result_df[st.session_state.result_df['Predicted_Category'].notna()]
            if len(classified_df) > 0:
                classified_csv = export_results_csv(classified_df)
                st.download_button(
                    label="✅ Download Classified",
                    data=classified_csv,
                    file_name=f'classified_results_{timestamp}.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            else:
                st.button("✅ No Classified Data", disabled=True, use_container_width=True)
        
        with col4:
            # Download only unclassified results
            unclassified_df = st.session_state.result_df[st.session_state.result_df['Predicted_Category'].isna()]
            if len(unclassified_df) > 0:
                unclassified_csv = export_results_csv(unclassified_df)
                st.download_button(
                    label="❌ Download Unclassified",
                    data=unclassified_csv,
                    file_name=f'unclassified_results_{timestamp}.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            else:
                st.button("❌ No Unclassified Data", disabled=True, use_container_width=True)
        
        # Filter options
        st.write("**Filter Display Options:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_category = st.selectbox(
                "Show Category:",
                ["All"] + list(st.session_state.result_df['Predicted_Category'].dropna().unique()) + ["Unclassified"]
            )
        
        with col2:
            show_rows = st.selectbox("Rows to Display:", [10, 25, 50, 100, "All"])
        
        # Apply filters
        filtered_df = st.session_state.result_df.copy()
        
        if show_category != "All":
            if show_category == "Unclassified":
                filtered_df = filtered_df[filtered_df['Predicted_Category'].isna()]
            else:
                filtered_df = filtered_df[filtered_df['Predicted_Category'] == show_category]
        
        if show_rows != "All":
            filtered_df = filtered_df.head(show_rows)
        
        # Display filtered results
        st.subheader(f"📄 Results ({len(filtered_df)} rows)")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Full results (expandable)
        with st.expander("📊 View All Results"):
            st.dataframe(st.session_state.result_df, use_container_width=True)

if __name__ == "__main__":
    main()
