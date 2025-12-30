"""
COVID Earnings Manipulation Analysis
Mid-Level Streamlit App with CSV Upload and DiD Analysis
"""

import streamlit as st
import pandas as pd
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
    st.session_state.analysis_done = False

# Custom CSS for better look
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1E3A8A;
        padding: 1rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 0.5rem 0;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">📊 COVID-Era Earnings Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Difference-in-Differences Analysis of Financial Reporting</p>', unsafe_allow_html=True)

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
                st.error(f"❌ Missing: {missing}")
            else:
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df)} rows")
                
                # Show quick stats
                st.info(f"""
                **Quick Stats:**
                - Companies: {df['company_id'].nunique()}
                - Years: {df['year'].min()} to {df['year'].max()}
                - Treatment: {df[df['treatment_group']==1]['company_id'].nunique()}
                - Control: {df[df['treatment_group']==0]['company_id'].nunique()}
                """)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Sample data option
    st.markdown("---")
    st.header("🎲 Sample Data")
    
    if st.button("Generate Sample Data", use_container_width=True):
        # Generate realistic sample data
        data = []
        for i in range(200):
            treatment = 1 if i < 100 else 0  # First 100 treatment, rest control
            industry = 'Aviation' if treatment == 1 else 'IT'
            
            for year in [2018, 2019, 2020, 2021]:
                base = 1000 + i * 10
                
                # COVID impact for treatment group
                if year >= 2020 and treatment == 1:
                    revenue = base * (1 + 0.1 * (year - 2018)) * 0.7  # 30% drop
                    profit = revenue * 0.15  # Higher margin (possible manipulation)
                else:
                    revenue = base * (1 + 0.1 * (year - 2018))
                    profit = revenue * 0.10  # Normal margin
                
                data.append({
                    'company_id': f'C{i:04d}',
                    'industry': industry,
                    'year': year,
                    'treatment_group': treatment,
                    'post_covid': 1 if year >= 2020 else 0,
                    'revenue': round(revenue, 2),
                    'net_profit': round(profit, 2),
                    'cfo': round(profit * 0.9, 2),
                    'total_assets': round(revenue * 2, 2)
                })
        
        df = pd.DataFrame(data)
        st.session_state.df = df
        st.success(f"✅ Generated {len(df)} rows")
        st.rerun()
    
    # Template download
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
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col2:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown("## 👈 Upload Your Data")
        st.markdown("""
        ### Required CSV Format:
        
        | Column | Required | Example |
        |--------|----------|---------|
        | company_id | Yes | C001 |
        | year | Yes | 2020 |
        | treatment_group | Yes | 1 or 0 |
        | post_covid | Yes | 1 or 0 |
        | net_profit | Recommended | 100.50 |
        | revenue | Optional | 1000.00 |
        
        **Treatment Group:** 1 = COVID-affected, 0 = Less affected
        **Post-COVID:** 1 = 2020+, 0 = Before 2020
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Project info
    st.markdown("---")
    st.markdown("## 📋 Project Information")
    
    st.markdown("""
    ### What this app does:
    
    1. **CSV Upload** - Upload your company financial data
    2. **DiD Analysis** - Calculate Difference-in-Differences effect
    3. **Visualizations** - Interactive charts and graphs
    4. **Results** - Export analysis results
    
    ### Methodology:
    
    - **Treatment Group:** Aviation, Hospitality, Real Estate
    - **Control Group:** IT, FMCG, Pharma
    - **Pre-Period:** Before 2020
    - **Post-Period:** 2020 and later
    - **Analysis:** Difference-in-Differences (DiD)
    """)
    
    st.stop()

# Data is loaded - show analysis
df = st.session_state.df

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "📈 Analysis", "📉 Charts", "📋 Report"])

with tab1:
    st.header("Data Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Companies", df['company_id'].nunique())
    
    with col2:
        treatment = df[df['treatment_group'] == 1]['company_id'].nunique()
        st.metric("Treatment Group", treatment)
    
    with col3:
        control = df[df['treatment_group'] == 0]['company_id'].nunique()
        st.metric("Control Group", control)
    
    with col4:
        st.metric("Years", df['year'].nunique())
    
    # Data preview
    st.subheader("Data Preview")
    
    show_cols = st.multiselect(
        "Select columns to show:",
        df.columns.tolist(),
        default=['company_id', 'year', 'treatment_group', 'post_covid', 'net_profit'][:5]
    )
    
    if show_cols:
        st.dataframe(df[show_cols].head(50), use_container_width=True)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)

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
    
    # Prepare data
    df_analysis = df.copy()
    df_analysis['period'] = df_analysis['year'].apply(
        lambda x: 'Pre-COVID' if x < 2020 else 'Post-COVID'
    )
    
    # Calculate group means
    try:
        means = df_analysis.groupby(['treatment_group', 'period'])[outcome_var].mean()
        
        # Extract values
        control_pre = means.get((0, 'Pre-COVID'), 0)
        control_post = means.get((0, 'Post-COVID'), 0)
        treatment_pre = means.get((1, 'Pre-COVID'), 0)
        treatment_post = means.get((1, 'Post-COVID'), 0)
        
        # Calculate changes
        control_change = control_post - control_pre
        treatment_change = treatment_post - treatment_pre
        did_effect = treatment_change - control_change
        
        # Display results in a nice table
        results_data = {
            'Group': ['Control (0)', 'Treatment (1)'],
            'Pre-COVID Mean': [control_pre, treatment_pre],
            'Post-COVID Mean': [control_post, treatment_post],
            'Change': [control_change, treatment_change]
        }
        
        results_df = pd.DataFrame(results_data)
        
        st.dataframe(
            results_df.style.format("{:.2f}"),
            use_container_width=True
        )
        
        # DiD result
        st.markdown(f"""
        <div class="metric-card">
        <h4>🎯 DiD Result</h4>
        <p><strong>Control Group Change:</strong> {control_change:.2f}</p>
        <p><strong>Treatment Group Change:</strong> {treatment_change:.2f}</p>
        <p><strong>DiD Effect (Treatment - Control):</strong> {did_effect:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Interpretation
        if did_effect > 0:
            st.success(f"""
            **Interpretation:** The treatment group shows {did_effect:.2f} higher change in {outcome_var}.
            
            This suggests **potential earnings manipulation** in COVID-affected industries.
            """)
        else:
            st.info(f"""
            **Interpretation:** The treatment group shows {did_effect:.2f} change in {outcome_var}.
            
            This suggests **no significant evidence of earnings manipulation**.
            """)
            
    except Exception as e:
        st.error(f"Error in calculation: {str(e)}")

with tab3:
    st.header("Visualizations")
    
    # Select variable for charts
    numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
                   if col not in ['year', 'treatment_group', 'post_covid']]
    
    if not numeric_cols:
        st.warning("No numeric columns for charts")
        st.stop()
    
    chart_var = st.selectbox("Select variable for charts:", numeric_cols, key='chart_var')
    
    # Time trend chart
    st.subheader("Time Trends")
    
    # Calculate average by year and group
    trend_data = df.groupby(['year', 'treatment_group'])[chart_var].mean().reset_index()
    trend_data['Group'] = trend_data['treatment_group'].map({0: 'Control', 1: 'Treatment'})
    
    fig1 = px.line(
        trend_data,
        x='year',
        y=chart_var,
        color='Group',
        markers=True,
        title=f"{chart_var.replace('_', ' ').title()} Over Time"
    )
    
    # Add COVID period if 2020 exists
    if 2020 in df['year'].values:
        fig1.add_vrect(
            x0=2019.5, x1=2020.5,
            fillcolor="red",
            opacity=0.1,
            line_width=0,
            annotation_text="COVID"
        )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Bar chart comparison
    st.subheader("Pre vs Post Comparison")
    
    df_chart = df.copy()
    df_chart['Period'] = df_chart['year'].apply(lambda x: 'Post-COVID' if x >= 2020 else 'Pre-COVID')
    df_chart['Group'] = df_chart['treatment_group'].map({0: 'Control', 1: 'Treatment'})
    
    # Calculate averages
    bar_data = df_chart.groupby(['Group', 'Period'])[chart_var].mean().reset_index()
    
    fig2 = px.bar(
        bar_data,
        x='Period',
        y=chart_var,
        color='Group',
        barmode='group',
        title=f"Average {chart_var.replace('_', ' ').title()} by Period and Group"
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Distribution chart
    st.subheader("Distribution by Group")
    
    fig3 = px.box(
        df,
        x='treatment_group',
        y=chart_var,
        color='treatment_group',
        points=False,
        title=f"Distribution of {chart_var.replace('_', ' ').title()}",
        labels={'treatment_group': 'Group (0=Control, 1=Treatment)'}
    )
    
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.header("Analysis Report")
    
    # Generate report
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_companies = df['company_id'].nunique()
    treatment_count = df[df['treatment_group'] == 1]['company_id'].nunique()
    control_count = df[df['treatment_group'] == 0]['company_id'].nunique()
    
    # Try to get DiD result
    did_result = ""
    try:
        df_analysis = df.copy()
        df_analysis['period'] = df_analysis['year'].apply(lambda x: 'pre' if x < 2020 else 'post')
        means = df_analysis.groupby(['treatment_group', 'period'])['net_profit'].mean().unstack()
        did_result = (means.loc[1, 'post'] - means.loc[1, 'pre']) - (means.loc[0, 'post'] - means.loc[0, 'pre'])
    except:
        pass
    
    report_content = f"""
# COVID Earnings Analysis Report

**Generated:** {report_date}
**Dataset:** {total_companies} companies, {len(df)} observations

## Executive Summary

This report presents a Difference-in-Differences (DiD) analysis of potential earnings manipulation during the COVID-19 pandemic.

### Key Findings:
1. **Dataset:** Analyzed {total_companies} companies ({treatment_count} treatment, {control_count} control)
2. **Period:** {df['year'].min()} to {df['year'].max()}
3. **DiD Effect:** {did_result:.2f if did_result != '' else 'Run analysis in Analysis tab'}
4. **Interpretation:** {'Potential earnings manipulation detected' if did_result > 0 else 'No strong evidence found' if did_result != '' else 'Analysis not completed'}

## Methodology

### Research Design:
- **Treatment Group:** COVID-affected industries
- **Control Group:** Less affected industries  
- **Pre-Period:** Before 2020
- **Post-Period:** 2020 and later
- **Statistical Method:** Difference-in-Differences (DiD)

## Recommendations

1. **Data Verification:** Ensure data accuracy
2. **Extended Analysis:** Include more financial metrics
3. **Industry Analysis:** Break down by specific sectors
4. **Time Extension:** Add more years of data

---
*Report generated by COVID Earnings Analysis Tool*
"""
    
    st.markdown(report_content)
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export report
        st.download_button(
            "📥 Download Report",
            data=report_content,
            file_name="covid_report.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        # Export data
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📊 Download Data",
            data=csv_data,
            file_name="analysis_data.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>COVID Earnings Analysis Tool | Upload CSV data for DiD analysis</p>
</div>
""", unsafe_allow_html=True)
