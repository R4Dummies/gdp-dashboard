import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime

def parse_dictionary(dict_input):
    dict_text = dict_input.strip()
    categories = {}
    
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
    
    try:
        for line in dict_text.splitlines():
            if ':' in line:
                cat, terms = line.split(':', 1)
                categories[cat.strip()] = [w.strip() for w in terms.split(',') if w.strip()]
        return categories, None
    except Exception as e:
        return {}, f"Error parsing dictionary: {str(e)}"

def classify_text(text, categories):
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
    return df.to_csv(index=False).encode('utf-8')

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
    
    if 'df_loaded' not in st.session_state:
        st.session_state.df_loaded = None
    if 'result_df' not in st.session_state:
        st.session_state.result_df = None
    if 'categories' not in st.session_state:
        st.session_state.categories = {}
    
    st.header("📁 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_loaded = pd.read_csv(uploaded_file)
            st.session_state.df_loaded = df_loaded
            st.success(f"✅ CSV uploaded successfully! Shape: {df_loaded.shape}")
            
            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df_loaded.head(10))
                
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            return
    
    if st.session_state.df_loaded is not None:
        st.header("⚙️ Configuration")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            text_col = st.selectbox(
                "📝 Select Text Column:",
                st.session_state.df_loaded.columns,
                help="Choose the column containing the text you want to classify"
            )
        
        with col2:
            st.write("📋 **Sample text from selected column:**")
            if text_col:
                sample_text = str(st.session_state.df_loaded[text_col].iloc[0])[:200] + "..."
                st.text_area("Sample:", value=sample_text, height=100, disabled=True)
        
        st.subheader("📚 Dictionary Configuration")
        
        dict_format = st.radio(
            "Choose input format:",
            ["Text Format", "JSON Format"],
            help="Text format: 'Category: word1, word2, word3'\nJSON format: {'Category': ['word1', 'word2', 'word3']}"
        )
        
        if dict_format == "Text Format":
            default_dict = """Luxury: elegant, timeless, refined, classic, sophisticated, luxury, polished
Technology: digital, innovation, tech, software, AI, automated, smart
Nature: organic, natural, eco, green, sustainable, environment, earth"""
        else:
            default_dict = """{
    "Luxury": ["elegant", "timeless", "refined", "classic", "sophisticated", "luxury", "polished"],
    "Technology": ["digital", "innovation", "tech", "software", "AI", "automated", "smart"],
    "Nature": ["organic", "natural", "eco", "green", "sustainable", "environment", "earth"]
}"""
        
        dict_input = st.text_area(
            "Dictionary:",
            value=default_dict,
            height=200,
            help='Enter your dictionary in the selected format'
        )
        
        categories, error = parse_dictionary(dict_input)
        
        if error:
            st.error(f"❌ {error}")
        elif categories:
            st.session_state.categories = categories
            st.success(f"✅ Dictionary parsed successfully! Categories: {list(categories.keys())}")
            
            with st.expander("🔍 Dictionary Preview"):
                for cat, words in categories.items():
                    st.write(f"**{cat}:** {', '.join(words)}")
        
        if categories and not error:
            st.header("🔍 Run Analysis")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🚀 Run Classification", type="primary"):
                    with st.spinner("Classifying texts..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, text in enumerate(st.session_state.df_loaded[text_col]):
                            predicted_category = classify_text(text, categories)
                            results.append(predicted_category)
                            progress_bar.progress((i + 1) / len(st.session_state.df_loaded))
                        
                        result_df = st.session_state.df_loaded.copy()
                        result_df['Predicted_Category'] = results
                        st.session_state.result_df = result_df
                        
                        st.success("✅ Analysis complete!")
            
            with col2:
                if st.session_state.result_df is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_data = export_results_csv(st.session_state.result_df)
                    st.download_button(
                        label="💾 Download Results",
                        data=csv_data,
                        file_name=f'classification_results_{timestamp}.csv',
                        mime='text/csv',
                        type="secondary"
                    )
    
    if st.session_state.result_df is not None:
        st.header("📊 Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_rows = len(st.session_state.result_df)
            st.metric("Total Rows", total_rows)
        
        with col2:
            classified_rows = len(st.session_state.result_df[st.session_state.result_df['Predicted_Category'].notna()])
            st.metric("Classified", classified_rows)
        
        with col3:
            unclassified_rows = total_rows - classified_rows
            st.metric("Unclassified", unclassified_rows)
        
        if 'Predicted_Category' in st.session_state.result_df.columns:
            category_counts = st.session_state.result_df['Predicted_Category'].value_counts()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📈 Category Distribution")
                st.bar_chart(category_counts)
            
            with col2:
                st.subheader("📋 Category Counts")
                st.dataframe(category_counts.reset_index().rename(columns={'index': 'Category', 'Predicted_Category': 'Count'}))
        
        st.subheader("📥 Download Options")
        
        col1, col2, col3, col4 = st.columns(4)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with col1:
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
        
        with col3:
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
        
        with col4:
            show_category = st.selectbox(
                "Filter by Category:",
                ["All"] + list(st.session_state.result_df['Predicted_Category'].dropna().unique()) + ["Unclassified"]
            )
        
        filtered_df = st.session_state.result_df.copy()
        
        if show_category != "All":
            if show_category == "Unclassified":
                filtered_df = filtered_df[filtered_df['Predicted_Category'].isna()]
            else:
                filtered_df = filtered_df[filtered_df['Predicted_Category'] == show_category]
        
        st.subheader(f"📄 Sample Results ({len(filtered_df)} rows)")
        st.dataframe(filtered_df.head(10))
        
        with st.expander("📊 View All Results"):
            st.dataframe(st.session_state.result_df)

if __name__ == "__main__":
    main()
