import streamlit as st
import pandas as pd
import json

st.title("🧠 Dictionary Classifier App")
st.markdown("""
Upload a CSV, select the column with text, and input your dictionary as either JSON or lines like:
`Luxury: elegant, timeless, classic`.
Then run the analysis to get category predictions based on keyword matches.
""")

# Add example dictionary formats in sidebar
with st.sidebar:
    st.header("Dictionary Format Examples")
    st.markdown("""
    **Line Format:**
    ```
    Luxury: elegant, timeless, classic
    Budget: cheap, affordable, value
    Sport: athletic, performance, active
    ```
    
    **JSON Format:**
    ```json
    {
        "Luxury": ["elegant", "timeless", "classic"],
        "Budget": ["cheap", "affordable", "value"]
    }
    ```
    """)

uploaded_file = st.file_uploader("📁 Upload CSV", type="csv")

if uploaded_file:
    try:
        df_loaded = pd.read_csv(uploaded_file)
        st.success(f"CSV uploaded successfully! ({len(df_loaded)} rows, {len(df_loaded.columns)} columns)")
        
        # Show data preview
        with st.expander("Preview Data"):
            st.dataframe(df_loaded.head())
        
        text_col = st.selectbox("Text Column:", df_loaded.columns)
        
        # Show sample from selected column
        if text_col:
            st.write("**Sample texts:**")
            sample_texts = df_loaded[text_col].dropna().head(3)
            for i, text in enumerate(sample_texts, 1):
                st.text(f"{i}. {str(text)[:150]}{'...' if len(str(text)) > 150 else ''}")
        
        dict_input = st.text_area(
            "Dictionary:",
            value='Luxury: elegant, timeless, refined, classic, sophisticated, luxury, polished\nBudget: cheap, affordable, economical, budget, value, inexpensive',
            height=100,
            help='Enter JSON or lines like "Category: keyword1, keyword2, ..."'
        )
        
        # Preview parsed dictionary
        if dict_input.strip():
            dict_text = dict_input.strip()
            categories = {}
            
            # Try JSON first
            try:
                parsed = json.loads(dict_text)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        if isinstance(v, str):
                            v = [v]
                        categories[k] = [str(x).strip() for x in v]
            except:
                # Try line format
                for line in dict_text.splitlines():
                    if ':' in line:
                        cat, terms = line.split(':', 1)
                        categories[cat.strip()] = [w.strip() for w in terms.split(',') if w.strip()]
            
            if categories:
                st.write("**Dictionary Preview:**")
                for cat, words in categories.items():
                    st.write(f"• **{cat}**: {', '.join(words[:8])}{'...' if len(words) > 8 else ''}")
            else:
                st.warning("⚠️ Dictionary format not recognized. Check the sidebar for examples.")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            run_button = st.button("🔍 Run Analysis", type="primary")
        
        if run_button:
            if not categories:
                st.error("❌ Invalid dictionary format.")
            else:
                # Add progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                texts = df_loaded[text_col].astype(str)
                total = len(texts)
                
                for i, txt in enumerate(texts):
                    txt_l = txt.lower()
                    best_cat, max_count = None, 0
                    
                    for cat, words in categories.items():
                        count = sum(txt_l.count(w.lower()) for w in words if w)
                        if count > max_count:
                            best_cat, max_count = cat, count
                    
                    results.append(best_cat)
                    
                    # Update progress
                    if i % max(1, total // 100) == 0 or i == total - 1:
                        progress_bar.progress((i + 1) / total)
                        status_text.text(f"Processing {i + 1}/{total} texts...")
                
                result_df = df_loaded.copy()
                result_df['Predicted_Category'] = results
                
                progress_bar.empty()
                status_text.empty()
                
                # Show results summary
                classified_count = result_df['Predicted_Category'].notna().sum()
                st.success(f"✅ Analysis complete! {classified_count}/{len(result_df)} texts classified")
                
                # Category distribution
                if classified_count > 0:
                    category_counts = result_df['Predicted_Category'].value_counts()
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write("**Category Distribution:**")
                        for cat, count in category_counts.items():
                            percentage = (count / classified_count) * 100
                            st.write(f"• {cat}: {count} ({percentage:.1f}%)")
                    
                    with col2:
                        st.bar_chart(category_counts)
                
                # Sample results
                st.write("**Sample Results:**")
                sample_df = result_df[[text_col, 'Predicted_Category']].head(10)
                st.dataframe(sample_df, use_container_width=True)
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download All Results",
                        data=csv,
                        file_name='classification_results.csv',
                        mime='text/csv'
                    )
                
                with col2:
                    # Only classified results
                    classified_df = result_df[result_df['Predicted_Category'].notna()]
                    if len(classified_df) > 0:
                        csv_classified = classified_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="✅ Download Classified Only",
                            data=csv_classified,
                            file_name='classification_results_classified.csv',
                            mime='text/csv'
                        )
                
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        st.info("Please ensure your CSV file is properly formatted and try uploading again.")
