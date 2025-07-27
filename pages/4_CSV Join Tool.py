import streamlit as st
import pandas as pd
import re
import io
from typing import Dict, List, Tuple, Optional
import xlsxwriter
from difflib import SequenceMatcher

# Configure page
st.set_page_config(
    page_title="CSV Join Tool",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default dictionaries for data cleaning and standardization
DEFAULT_CLEANING_DICT = {
    "company_suffixes": {
        "inc": "Inc",
        "corp": "Corp",
        "llc": "LLC", 
        "ltd": "Ltd",
        "co": "Co",
        "company": "Company",
        "corporation": "Corporation"
    },
    "states": {
        "ca": "California",
        "ny": "New York", 
        "tx": "Texas",
        "fl": "Florida",
        "il": "Illinois"
    },
    "common_replacements": {
        "&": "and",
        "@": "at",
        "#": "number",
        "%": "percent"
    }
}

def clean_text(text: str, use_fuzzy: bool = False, cleaning_dict: Dict = None) -> str:
    """Clean and standardize text based on user preferences"""
    if not text or pd.isna(text):
        return ''
    
    text = str(text).strip()
    
    if use_fuzzy:
        # Basic fuzzy cleaning
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Apply user-defined cleaning dictionary
        if cleaning_dict:
            for category, replacements in cleaning_dict.items():
                for old_val, new_val in replacements.items():
                    text = text.replace(old_val.lower(), new_val.lower())
    
    return text.strip()

def fuzzy_similarity(a: str, b: str, threshold: float = 0.8) -> bool:
    """Check if two strings are similar based on fuzzy matching"""
    if not a or not b:
        return False
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

def generate_row_key(row: pd.Series, columns: List[str], use_fuzzy: bool = False, 
                    cleaning_dict: Dict = None) -> str:
    """Generate a key for row matching"""
    values = []
    for col in columns:
        val = row.get(col, '')
        if use_fuzzy:
            val = clean_text(val, use_fuzzy, cleaning_dict)
        else:
            val = str(val).lower() if val else ''
        values.append(val)
    return '|'.join(values)

def deduplicate_data(df: pd.DataFrame, columns: List[str], use_fuzzy: bool = False,
                    cleaning_dict: Dict = None) -> Tuple[pd.DataFrame, Dict]:
    """Remove duplicate rows based on specified columns"""
    if not columns or df.empty:
        return df, {"original": len(df), "final": len(df), "removed": 0}
    
    original_count = len(df)
    
    # Create keys for deduplication
    df_copy = df.copy()
    df_copy['_dedup_key'] = df_copy.apply(
        lambda row: generate_row_key(row, columns, use_fuzzy, cleaning_dict), 
        axis=1
    )
    
    # Remove duplicates
    df_deduplicated = df_copy.drop_duplicates(subset=['_dedup_key']).drop('_dedup_key', axis=1)
    
    final_count = len(df_deduplicated)
    removed_count = original_count - final_count
    
    stats = {
        "original": original_count,
        "final": final_count, 
        "removed": removed_count
    }
    
    return df_deduplicated, stats

def perform_join(df1: pd.DataFrame, df2: pd.DataFrame, 
                join_cols1: List[str], join_cols2: List[str],
                join_type: str, use_fuzzy: bool = False,
                cleaning_dict: Dict = None, fuzzy_threshold: float = 0.8) -> pd.DataFrame:
    """Perform join operation between two dataframes"""
    
    if use_fuzzy:
        # Create temporary keys for fuzzy matching
        df1_temp = df1.copy()
        df2_temp = df2.copy()
        
        df1_temp['_join_key'] = df1_temp.apply(
            lambda row: generate_row_key(row, join_cols1, use_fuzzy, cleaning_dict), 
            axis=1
        )
        df2_temp['_join_key'] = df2_temp.apply(
            lambda row: generate_row_key(row, join_cols2, use_fuzzy, cleaning_dict),
            axis=1
        )
        
        # Perform join on the temporary keys
        if join_type == 'inner':
            result = df1_temp.merge(df2_temp, on='_join_key', how='inner', suffixes=('', '_right'))
        elif join_type == 'left':
            result = df1_temp.merge(df2_temp, on='_join_key', how='left', suffixes=('', '_right'))
        elif join_type == 'right':
            result = df1_temp.merge(df2_temp, on='_join_key', how='right', suffixes=('', '_right'))
        elif join_type == 'outer':
            result = df1_temp.merge(df2_temp, on='_join_key', how='outer', suffixes=('', '_right'))
        
        # Remove temporary key
        if '_join_key' in result.columns:
            result = result.drop('_join_key', axis=1)
            
    else:
        # Standard join
        # Create multi-column keys for joining
        if len(join_cols1) == 1 and len(join_cols2) == 1:
            left_on = join_cols1[0]
            right_on = join_cols2[0]
        else:
            left_on = join_cols1
            right_on = join_cols2
        
        if join_type == 'inner':
            result = df1.merge(df2, left_on=left_on, right_on=right_on, how='inner', suffixes=('', '_right'))
        elif join_type == 'left':
            result = df1.merge(df2, left_on=left_on, right_on=right_on, how='left', suffixes=('', '_right'))
        elif join_type == 'right':
            result = df1.merge(df2, left_on=left_on, right_on=right_on, how='right', suffixes=('', '_right'))
        elif join_type == 'outer':
            result = df1.merge(df2, left_on=left_on, right_on=right_on, how='outer', suffixes=('', '_right'))
    
    return result

def create_excel_download(df: pd.DataFrame, filename: str = "joined_data.xlsx") -> bytes:
    """Create Excel file for download"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Joined Data', index=False)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Joined Data']
        
        # Add formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#154734',  # Cal Poly Green
            'font_color': 'white',
            'border': 1
        })
        
        # Apply header formatting
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            # Auto-adjust column width
            worksheet.set_column(col_num, col_num, min(len(str(value)) + 5, 50))
    
    output.seek(0)
    return output.getvalue()

def main():
    st.title("🔗 CSV Join Tool")
    st.markdown("**Cal Poly Pomona** - Advanced CSV file merging with customizable data cleaning")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Join type selection
        join_type = st.selectbox(
            "Join Type",
            options=['inner', 'left', 'right', 'outer'],
            help="Select how to join the datasets"
        )
        
        # Advanced options
        st.subheader("Advanced Options")
        use_fuzzy = st.checkbox(
            "Enable Fuzzy Matching",
            help="Clean text and use approximate matching"
        )
        
        if use_fuzzy:
            fuzzy_threshold = st.slider(
                "Fuzzy Matching Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Higher values require closer matches"
            )
        
        use_dedupe = st.checkbox(
            "Remove Duplicates Before Join",
            help="Remove duplicate rows based on join columns"
        )
        
        # Dictionary customization
        st.subheader("📚 Data Cleaning Dictionary")
        
        if st.checkbox("Customize Cleaning Dictionary"):
            st.markdown("**Default dictionaries available:**")
            
            # Allow users to modify the cleaning dictionary
            cleaning_dict = st.session_state.get('cleaning_dict', DEFAULT_CLEANING_DICT.copy())
            
            # Company suffixes
            st.markdown("*Company Suffixes:*")
            company_suffixes = st.text_area(
                "Company Suffixes (format: old:new, one per line)",
                value="\n".join([f"{k}:{v}" for k, v in cleaning_dict.get('company_suffixes', {}).items()]),
                height=100
            )
            
            # Parse company suffixes
            new_suffixes = {}
            for line in company_suffixes.split('\n'):
                if ':' in line:
                    old, new = line.split(':', 1)
                    new_suffixes[old.strip()] = new.strip()
            cleaning_dict['company_suffixes'] = new_suffixes
            
            # States
            st.markdown("*State Abbreviations:*")
            states = st.text_area(
                "State Abbreviations (format: abbrev:full, one per line)",
                value="\n".join([f"{k}:{v}" for k, v in cleaning_dict.get('states', {}).items()]),
                height=100
            )
            
            # Parse states
            new_states = {}
            for line in states.split('\n'):
                if ':' in line:
                    old, new = line.split(':', 1)
                    new_states[old.strip()] = new.strip()
            cleaning_dict['states'] = new_states
            
            # Common replacements
            st.markdown("*Common Replacements:*")
            replacements = st.text_area(
                "Common Replacements (format: old:new, one per line)",
                value="\n".join([f"{k}:{v}" for k, v in cleaning_dict.get('common_replacements', {}).items()]),
                height=100
            )
            
            # Parse replacements
            new_replacements = {}
            for line in replacements.split('\n'):
                if ':' in line:
                    old, new = line.split(':', 1)
                    new_replacements[old.strip()] = new.strip()
            cleaning_dict['common_replacements'] = new_replacements
            
            st.session_state.cleaning_dict = cleaning_dict
        else:
            st.session_state.cleaning_dict = DEFAULT_CLEANING_DICT.copy()
    
    # Main content area
    col1, col2 = st.columns(2)
    
    # File uploads
    with col1:
        st.subheader("📁 Upload First CSV File")
        file1 = st.file_uploader(
            "Choose first CSV file",
            type=['csv'],
            key="file1"
        )
        
        if file1:
            try:
                df1 = pd.read_csv(file1)
                st.success(f"✅ Loaded {len(df1)} rows, {len(df1.columns)} columns")
                st.write("**Preview:**")
                st.dataframe(df1.head(), use_container_width=True)
                
                # Column selection for joining
                st.subheader("Select Join Columns (File 1)")
                join_cols1 = st.multiselect(
                    "Join columns from first file",
                    options=df1.columns.tolist(),
                    key="join_cols1",
                    max_selections=3
                )
                
            except Exception as e:
                st.error(f"Error reading file 1: {str(e)}")
                df1 = None
                join_cols1 = []
        else:
            df1 = None
            join_cols1 = []
    
    with col2:
        st.subheader("📁 Upload Second CSV File") 
        file2 = st.file_uploader(
            "Choose second CSV file",
            type=['csv'],
            key="file2"
        )
        
        if file2:
            try:
                df2 = pd.read_csv(file2)
                st.success(f"✅ Loaded {len(df2)} rows, {len(df2.columns)} columns")
                st.write("**Preview:**")
                st.dataframe(df2.head(), use_container_width=True)
                
                # Column selection for joining
                st.subheader("Select Join Columns (File 2)")
                join_cols2 = st.multiselect(
                    "Join columns from second file",
                    options=df2.columns.tolist(),
                    key="join_cols2",
                    max_selections=3
                )
                
            except Exception as e:
                st.error(f"Error reading file 2: {str(e)}")
                df2 = None
                join_cols2 = []
        else:
            df2 = None
            join_cols2 = []
    
    # Join operation
    if st.button("🔗 Perform Join", type="primary", use_container_width=True):
        if df1 is None or df2 is None:
            st.error("Please upload both CSV files")
        elif not join_cols1 or not join_cols2:
            st.error("Please select join columns for both files")
        elif len(join_cols1) != len(join_cols2):
            st.error("Please select the same number of join columns for both files")
        else:
            try:
                with st.spinner("Processing join operation..."):
                    # Get cleaning dictionary
                    cleaning_dict = st.session_state.get('cleaning_dict', DEFAULT_CLEANING_DICT)
                    
                    # Store original data
                    original_df1, original_df2 = df1.copy(), df2.copy()
                    
                    # Apply deduplication if requested
                    dedupe_stats = {}
                    if use_dedupe:
                        df1, stats1 = deduplicate_data(df1, join_cols1, use_fuzzy, cleaning_dict)
                        df2, stats2 = deduplicate_data(df2, join_cols2, use_fuzzy, cleaning_dict)
                        dedupe_stats = {'file1': stats1, 'file2': stats2}
                    
                    # Perform the join
                    if use_fuzzy:
                        result_df = perform_join(
                            df1, df2, join_cols1, join_cols2, join_type, 
                            use_fuzzy, cleaning_dict, fuzzy_threshold
                        )
                    else:
                        result_df = perform_join(
                            df1, df2, join_cols1, join_cols2, join_type
                        )
                    
                    # Store results in session state
                    st.session_state.result_df = result_df
                    st.session_state.dedupe_stats = dedupe_stats
                    
                st.success(f"✅ Join completed! Result contains {len(result_df)} rows")
                
            except Exception as e:
                st.error(f"Error performing join: {str(e)}")
    
    # Display results
    if 'result_df' in st.session_state and not st.session_state.result_df.empty:
        result_df = st.session_state.result_df
        dedupe_stats = st.session_state.get('dedupe_stats', {})
        
        st.header("📊 Join Results")
        
        # Show deduplication stats if applicable
        if dedupe_stats:
            st.subheader("🔄 Deduplication Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'file1' in dedupe_stats:
                    stats = dedupe_stats['file1']
                    st.info(f"**File 1:** {stats['original']} → {stats['final']} rows ({stats['removed']} removed)")
            
            with col2:
                if 'file2' in dedupe_stats:
                    stats = dedupe_stats['file2']
                    st.info(f"**File 2:** {stats['original']} → {stats['final']} rows ({stats['removed']} removed)")
        
        # Display result summary
        st.metric("Total Rows", len(result_df))
        st.metric("Total Columns", len(result_df.columns))
        
        # Show preview
        st.subheader("📋 Data Preview")
        st.dataframe(result_df.head(100), use_container_width=True)
        
        if len(result_df) > 100:
            st.info(f"Showing first 100 rows. Download Excel file to see all {len(result_df)} rows.")
        
        # Download section
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel download
            excel_data = create_excel_download(result_df)
            st.download_button(
                label="📊 Download as Excel",
                data=excel_data,
                file_name="joined_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # CSV download
            csv_data = result_df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name="joined_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
