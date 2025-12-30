"""
COVID Earnings Manipulation Analysis
MID-LEVEL APP - WITH BUG FIXES
"""

import streamlit as st
import pandas as pd
import numpy as np  # ← ADDED THIS LINE
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="COVID Earnings Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

# Title
st.title("📊 COVID-Era Earnings Analysis")
st.markdown("Difference-in-Differences Analysis of Financial Reporting")

# ============================================================================
# SIDEBAR - DATA UPLOAD
# ============================================================================
with st.sidebar:
    st.header("📁 Data Upload")
    
    # Upload CSV
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Upload your financial data CSV"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check required columns
            required = ['company_id', 'year', 'treatment_group', 'post_covid']
            missing = [col for col in required if col not in df.columns]
            
            if missing:
                st.error(f"❌ Missing: {', '.join(missing)}")
            else:
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df)} rows")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Sample data option
    st.markdown("---")
    st.header("🎲 Sample Data")
    
    if st.button("Generate Sample Data", use_container_width=True):
        # Simple sample data
        np.random.seed(42)  # ← NOW THIS WILL WORK
        data = []
        for i in range(200):
            treatment = 1 if i < 100 else 0
            for year in [2019, 2020]:
                base = 1000
                if year == 2020 and treatment == 1:
                    profit = base * np.random.uniform(0.9, 1.3)
                else:
                    profit = base * np.random.uniform(0.8, 1.1)
                
                data.append({
                    'company_id': f'C{i:03d}',
                    'year': year,
                    'treatment_group': treatment,
                    'post_covid': 1 if year == 2020 else 0,
                    'net_profit': round(profit, 2),
                    'revenue': round(profit * np.random.uniform(5, 10), 2)
                })
        
        df = pd.DataFrame(data)
        st.session_state.df = df
        st.success(f"✅ Generated {len(df)} rows")
        st.rerun()
    
    # Template
    st.markdown("---")
    st.header("📥 Template")
    
    template_data = {
        'company_id': ['C001', 'C001', 'C002', 'C002'],
        'year': [2019, 2020, 2019, 2020],
        'treatment_group': [1, 1, 0, 0],
        'post_covid': [0, 1, 0, 1],
        'net_profit': [100, 120, 200, 210],
        'revenue': [1000, 1100, 2000, 2050]
    }
    
    template_df = pd.DataFrame(template_data)
    csv = template_df.to_csv(index=False)
    
    st.download_button(
        "Download Template",
        data=csv,
        file_name="template.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================================
# MAIN CONTENT
# ============================================================================
if st.session_state.df is None:
    # Welcome screen
    st.info("👈 Upload a CSV file or generate sample data to begin analysis")
    st.markdown("---")
    st.markdown("""
    ### Required CSV Format:
    
    | Column | Description | Example |
    |--------|-------------|---------|
    | company_id | Unique ID | C001 |
    | year | Year | 2020 |
    | treatment_group | 1=treatment, 0=control | 1 |
    | post_covid | 1=2020+, 0=pre-2020 | 1 |
    | net_profit | Net profit amount | 100.50 |
    
    ### Example CSV:
    """)
    
    example_df = pd.DataFrame({
        'company_id': ['C001', 'C001', 'C002', 'C002'],
        'year': [2019, 2020, 2019, 2020],
        'treatment_group': [1, 1, 0, 0],
        'post_covid': [0, 1, 0, 1],
        'net_profit': [100.0, 120.0, 200.0, 210.0]
    })
    st.dataframe(example_df)
    
    st.stop()

# Data is loaded
df = st.session_state.df

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Analysis", "📉 Charts"])

with tab1:
    st.header("Data Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Companies", df['company_id'].nunique())
    with col2:
        st.metric("Rows", len(df))
    with col3:
        st.metric("Years", f"{df['year'].min()} to {df['year'].max()}")
    
    st.subheader("Data Preview")
    st.dataframe(df.head(50))
    
    st.subheader("Summary Statistics")
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe())

with tab2:
    st.header("DiD Analysis")
    
    # Select outcome variable
    numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
                   if col not in ['year', 'treatment_group', 'post_covid']]
    
    if not numeric_cols:
        st.warning("No numeric columns found for analysis")
        st.stop()
    
    outcome_var = st.selectbox("Select variable to analyze:", numeric_cols)
    
    # Calculate DiD
    st.subheader("Difference-in-Differences Calculation")
    
    try:
        # Prepare data
        df_analysis = df.copy()
        df_analysis['period'] = df_analysis['year'].apply(
            lambda x: 'Pre-COVID' if x < 2020 else 'Post-COVID'
        )
        
        # Calculate group means
        means = df_analysis.groupby(['treatment_group', 'period'])[outcome_var].mean().unstack()
        
        # Calculate changes
        control_pre = means.loc[0, 'Pre-COVID'] if 'Pre-COVID' in means.columns else 0
        control_post = means.loc[0, 'Post-COVID'] if 'Post-COVID' in means.columns else 0
        treatment_pre = means.loc[1, 'Pre-COVID'] if 'Pre-COVID' in means.columns else 0
        treatment_post = means.loc[1, 'Post-COVID'] if 'Post-COVID' in means.columns else 0
        
        control_change = control_post - control_pre
        treatment_change = treatment_post - treatment_pre
        did_effect = treatment_change - control_change
        
        # Display results
        results_data = {
            'Group': ['Control (0)', 'Treatment (1)'],
            'Pre-COVID': [control_pre, treatment_pre],
            'Post-COVID': [control_post, treatment_post],
            'Change': [control_change, treatment_change]
        }
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df.style.format("{:.2f}"))
        
        # DiD result
        st.metric("DiD Effect", f"{did_effect:.2f}")
        
        # Interpretation
        if did_effect > 0:
            st.success(f"**Interpretation:** Treatment group shows {did_effect:.2f} higher change. Suggests potential earnings manipulation.")
        else:
            st.info(f"**Interpretation:** Treatment group shows {did_effect:.2f} change. No strong evidence of earnings manipulation.")
            
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")

with tab3:
    st.header("Visualizations")
    
    # Select variable
    numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
                   if col not in ['year', 'treatment_group', 'post_covid']]
    
    if not numeric_cols:
        st.warning("No numeric columns for charts")
        st.stop()
    
    chart_var = st.selectbox("Select variable for charts:", numeric_cols, key='chart')
    
    # Time trend chart
    st.subheader("Time Trends")
    
    try:
        trend_data = df.groupby(['year', 'treatment_group'])[chart_var].mean().reset_index()
        trend_data['Group'] = trend_data['treatment_group'].apply(lambda x: 'Treatment' if x == 1 else 'Control')
        
        fig = px.line(
            trend_data,
            x='year',
            y=chart_var,
            color='Group',
            markers=True,
            title=f"{chart_var} Over Time"
        )
        
        if 2020 in df['year'].values:
            fig.add_vrect(x0=2019.5, x1=2020.5, fillcolor="red", opacity=0.1, annotation_text="COVID")
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Chart error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("COVID Earnings Analysis Tool | Upload CSV for DiD analysis")
