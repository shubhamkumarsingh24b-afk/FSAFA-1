import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="COVID-Era Earnings Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1E3A8A;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 24px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for data persistence
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# Advanced financial calculations
class FinancialAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.pre_covid_years = [2018, 2019]
        self.covid_years = [2020, 2021]
        
    def calculate_advanced_metrics(self):
        """Calculate advanced financial metrics"""
        # Basic metrics
        self.df['accruals'] = self.df['net_profit'] - self.df['operating_cash_flow']
        self.df['current_ratio'] = (self.df['receivables'] + self.df.get('inventory', 0)) / self.df['current_liabilities']
        self.df['debt_to_equity'] = self.df['total_debt'] / self.df['total_equity']
        self.df['profit_margin'] = self.df['net_profit'] / self.df['revenue']
        self.df['asset_turnover'] = self.df['revenue'] / self.df['total_assets']
        self.df['roa'] = self.df['net_profit'] / self.df['total_assets']
        self.df['roe'] = self.df['net_profit'] / self.df['total_equity']
        
        # Advanced metrics
        self.df['abnormal_accruals'] = self.calculate_abnormal_accruals()
        self.df['beneish_m_score'] = self.calculate_beneish_m_score()
        self.df['financial_distress'] = self.calculate_altman_z_score()
        self.df['earnings_quality'] = self.df['operating_cash_flow'] / abs(self.df['net_profit'])
        
        # Manipulation indicators
        self.df['manipulation_risk'] = self.calculate_manipulation_risk()
        
        return self.df
    
    def calculate_abnormal_accruals(self):
        """Calculate abnormal accruals using Modified Jones Model"""
        try:
            # Calculate total accruals
            df = self.df.copy()
            df = df.sort_values(['company_id', 'year'])
            
            # Calculate change in revenue and PPE
            df['delta_revenue'] = df.groupby('company_id')['revenue'].diff()
            df['delta_receivables'] = df.groupby('company_id')['receivables'].diff()
            
            # Estimate normal accruals (simplified version)
            # Normal accruals = α*(1/Total_Assets) + β1*(ΔRevenue - ΔReceivables)/Total_Assets + β2*(PPE/Total_Assets)
            df['normal_accruals'] = (
                0.05 * (1/df['total_assets']) + 
                0.35 * ((df['delta_revenue'] - df['delta_receivables']) / df['total_assets']) +
                0.15 * (df.get('ppe', 0) / df['total_assets'])
            )
            
            # Abnormal accruals = Total accruals - Normal accruals
            df['abnormal_accruals'] = (df['accruals'] / df['total_assets']) - df['normal_accruals']
            return df['abnormal_accruals']
        except:
            return np.zeros(len(self.df))
    
    def calculate_beneish_m_score(self):
        """Calculate Beneish M-Score for earnings manipulation detection"""
        try:
            df = self.df.copy()
            
            # Days Sales in Receivables Index
            dsri = (df['receivables'] / df['revenue']) / df.groupby('company_id')['receivables'].shift(1) / df.groupby('company_id')['revenue'].shift(1)
            
            # Gross Margin Index
            gmi = ((df.groupby('company_id')['revenue'].shift(1) - df.groupby('company_id')['cogs'].shift(1)) / df.groupby('company_id')['revenue'].shift(1)) / \
                  ((df['revenue'] - df.get('cogs', df['revenue'] * 0.7)) / df['revenue'])
            
            # Asset Quality Index
            aqi = (1 - (df.get('current_assets', 0) + df.get('ppe', 0)) / df['total_assets']) / \
                  (1 - (df.groupby('company_id')['current_assets'].shift(1) + df.groupby('company_id')['ppe'].shift(1)) / df.groupby('company_id')['total_assets'].shift(1))
            
            # Sales Growth Index
            sgi = df['revenue'] / df.groupby('company_id')['revenue'].shift(1)
            
            # Depreciation Index
            depi = (df.groupby('company_id')['depreciation'].shift(1) / df.groupby('company_id')['ppe'].shift(1)) / \
                   (df.get('depreciation', 0) / df.get('ppe', df['total_assets'] * 0.5))
            
            # Simplified M-Score calculation
            m_score = -4.84 + 0.92*dsri.fillna(1) + 0.528*gmi.fillna(1) + 0.404*aqi.fillna(1) + 0.892*sgi.fillna(1) + 0.115*depi.fillna(1)
            
            return m_score
        except:
            return np.zeros(len(self.df))
    
    def calculate_altman_z_score(self):
        """Calculate Altman Z-Score for financial distress"""
        try:
            df = self.df.copy()
            
            # Working capital to total assets
            X1 = (df['current_assets'] - df['current_liabilities']) / df['total_assets']
            
            # Retained earnings to total assets
            X2 = df['retained_earnings'] / df['total_assets']
            
            # EBIT to total assets
            X3 = (df['ebit'] if 'ebit' in df else df['net_profit'] * 1.2) / df['total_assets']
            
            # Market value of equity to book value of total liabilities
            X4 = (df['market_cap'] if 'market_cap' in df else df['total_equity'] * 1.5) / df['total_debt']
            
            # Sales to total assets
            X5 = df['revenue'] / df['total_assets']
            
            # Altman Z-Score
            z_score = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
            
            return z_score
        except:
            return np.zeros(len(self.df))
    
    def calculate_manipulation_risk(self):
        """Calculate comprehensive manipulation risk score"""
        try:
            df = self.df.copy()
            
            # Calculate individual risk factors
            accruals_risk = abs(df['abnormal_accruals']) / abs(df['abnormal_accruals']).max()
            beneish_risk = (df['beneish_m_score'] > -2.22).astype(int)
            earnings_quality_risk = (df['earnings_quality'] < 0.8).astype(int)
            
            # Composite risk score (0-1)
            risk_score = (accruals_risk * 0.4 + beneish_risk * 0.3 + earnings_quality_risk * 0.3)
            
            # Categorize risk levels
            risk_levels = pd.cut(risk_score, 
                                bins=[-0.1, 0.3, 0.6, 1.1], 
                                labels=['Low', 'Medium', 'High'])
            
            return risk_levels
        except:
            return ['Low'] * len(self.df)

class DIDAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        
    def perform_did_analysis(self, outcome_var):
        """Perform Difference-in-Differences analysis"""
        # Prepare data
        df_analysis = self.df.copy()
        df_analysis['post_covid'] = df_analysis['year'].apply(lambda x: 1 if x >= 2020 else 0)
        
        # Aggregate to company-level averages for pre and post periods
        company_agg = df_analysis.groupby(['company_id', 'treatment_group', 'post_covid'])[outcome_var].mean().reset_index()
        
        # Calculate DID manually
        treatment_pre = company_agg[(company_agg['treatment_group'] == 1) & (company_agg['post_covid'] == 0)][outcome_var].mean()
        treatment_post = company_agg[(company_agg['treatment_group'] == 1) & (company_agg['post_covid'] == 1)][outcome_var].mean()
        control_pre = company_agg[(company_agg['treatment_group'] == 0) & (company_agg['post_covid'] == 0)][outcome_var].mean()
        control_post = company_agg[(company_agg['treatment_group'] == 0) & (company_agg['post_covid'] == 1)][outcome_var].mean()
        
        # DID calculation
        did_effect = (treatment_post - treatment_pre) - (control_post - control_pre)
        
        # Regression-based DID for statistical significance
        try:
            df_reg = df_analysis.copy()
            df_reg['interaction'] = df_reg['treatment_group'] * df_reg['post_covid']
            
            model = smf.ols(f'{outcome_var} ~ treatment_group + post_covid + interaction', data=df_reg).fit()
            
            results = {
                'did_effect': did_effect,
                'treatment_pre': treatment_pre,
                'treatment_post': treatment_post,
                'control_pre': control_pre,
                'control_post': control_post,
                'treatment_change': treatment_post - treatment_pre,
                'control_change': control_post - control_pre,
                'model': model,
                'p_value': model.pvalues['interaction'],
                'ci_lower': model.conf_int().loc['interaction', 0],
                'ci_upper': model.conf_int().loc['interaction', 1]
            }
        except Exception as e:
            results = {
                'did_effect': did_effect,
                'treatment_pre': treatment_pre,
                'treatment_post': treatment_post,
                'control_pre': control_pre,
                'control_post': control_post,
                'treatment_change': treatment_post - treatment_pre,
                'control_change': control_post - control_pre,
                'model': None,
                'p_value': None,
                'error': str(e)
            }
        
        return results
    
    def parallel_trends_test(self, outcome_var):
        """Test parallel trends assumption"""
        df_test = self.df.copy()
        
        # Plot trends over time
        trend_data = df_test.groupby(['year', 'treatment_group'])[outcome_var].mean().reset_index()
        
        return trend_data

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 COVID-Era Earnings Manipulation Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Financial Analytics with Difference-in-Differences")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/data-analysis.png", width=100)
        st.markdown("## Data Upload")
        
        uploaded_file = st.file_uploader(
            "📤 Upload your financial data CSV",
            type=['csv'],
            help="Upload CSV with financial data for 2018-2021"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success(f"✅ Data loaded successfully! {len(df)} rows, {len(df.columns)} columns")
                
                # Show basic info
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head())
                
                with st.expander("📊 Data Summary"):
                    st.write(f"**Years:** {df['year'].unique()}")
                    st.write(f"**Companies:** {df['company_name'].nunique()}")
                    st.write(f"**Treatment Group:** {df[df['treatment_group']==1]['company_name'].nunique()} companies")
                    st.write(f"**Control Group:** {df[df['treatment_group']==0]['company_name'].nunique()} companies")
            
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        st.markdown("---")
        st.markdown("### ⚙️ Analysis Settings")
        
        if st.session_state.df is not None:
            outcome_options = ['net_profit', 'accruals', 'abnormal_accruals', 'profit_margin', 'roa', 'roe']
            selected_outcome = st.selectbox(
                "Select Outcome Variable for DiD Analysis",
                outcome_options,
                index=0
            )
            
            if st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True):
                with st.spinner("Running advanced financial analysis..."):
                    st.session_state.analysis_done = True
                    st.session_state.selected_outcome = selected_outcome
                st.rerun()
        
        st.markdown("---")
        st.markdown("#### 📚 About")
        st.info("""
        This tool analyzes potential earnings manipulation during COVID-19 using:
        - **Difference-in-Differences** methodology
        - **Advanced financial metrics** (Beneish M-Score, Abnormal Accruals)
        - **Statistical significance testing**
        - **Visual trend analysis**
        """)
    
    # Main content area
    if st.session_state.df is None:
        # Show upload instructions
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("## 📥 Get Started")
            st.info("""
            **Please upload your financial data CSV file in the sidebar.**
            
            Your data should include:
            - Company financials for 2018-2021
            - Treatment/Control group classification
            - Key financial metrics
            
            Need a template? [Download sample structure](#)
            """)
            
            with st.expander("📋 Required Data Columns"):
                st.code("""
                Required columns:
                - company_id, company_name, industry
                - treatment_group (1=treatment, 0=control)
                - year (2018, 2019, 2020, 2021)
                - revenue, net_profit, operating_cash_flow
                - total_assets, receivables, total_debt
                - current_liabilities, total_equity
                Optional:
                - inventory, ebit, depreciation, ppe
                - market_cap, retained_earnings
                """)
    else:
        if not st.session_state.analysis_done:
            st.warning("Click 'Run Complete Analysis' in sidebar to start analysis")
        else:
            # Initialize analyzers
            financial_analyzer = FinancialAnalyzer(st.session_state.df)
            df_analyzed = financial_analyzer.calculate_advanced_metrics()
            did_analyzer = DIDAnalyzer(df_analyzed)
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Overview Dashboard",
                "🔍 DiD Analysis",
                "📊 Advanced Metrics",
                "📉 Trend Analysis",
                "📋 Full Report"
            ])
            
            with tab1:
                # Overview Dashboard
                st.markdown('<h2 class="section-header">Executive Dashboard</h2>', unsafe_allow_html=True)
                
                # Key Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        label="Total Companies",
                        value=df_analyzed['company_id'].nunique(),
                        delta=f"{df_analyzed[df_analyzed['treatment_group']==1]['company_id'].nunique()} in Treatment"
                    )
                
                with col2:
                    high_risk = (df_analyzed['manipulation_risk'] == 'High').sum()
                    st.metric(
                        label="High Manipulation Risk",
                        value=high_risk,
                        delta=f"{high_risk/len(df_analyzed)*100:.1f}%"
                    )
                
                with col3:
                    avg_accruals = df_analyzed['abnormal_accruals'].mean()
                    st.metric(
                        label="Avg Abnormal Accruals",
                        value=f"{avg_accruals:.4f}",
                        delta="Post-COVID" if avg_accruals > 0 else "Pre-COVID"
                    )
                
                with col4:
                    beneish_red = (df_analyzed['beneish_m_score'] > -2.22).sum()
                    st.metric(
                        label="Beneish M-Score Alerts",
                        value=beneish_red,
                        delta=f"{beneish_red/len(df_analyzed)*100:.1f}%"
                    )
                
                # Industry Distribution
                st.markdown('<h3 class="section-header">Industry & Risk Distribution</h3>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = px.sunburst(
                        df_analyzed,
                        path=['industry', 'treatment_group', 'manipulation_risk'],
                        values='revenue',
                        title="Industry & Treatment Group Distribution"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = px.box(
                        df_analyzed,
                        x='treatment_group',
                        y='abnormal_accruals',
                        color='manipulation_risk',
                        title="Abnormal Accruals by Treatment Group & Risk Level",
                        labels={'treatment_group': 'Treatment Group (1=Treatment)'}
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            
            with tab2:
                # DiD Analysis
                st.markdown('<h2 class="section-header">Difference-in-Differences Analysis</h2>', unsafe_allow_html=True)
                
                # Perform DiD analysis
                did_results = did_analyzer.perform_did_analysis(st.session_state.selected_outcome)
                
                # Display DiD results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### DiD Results Summary")
                    
                    # Create results table
                    results_df = pd.DataFrame({
                        'Period': ['Pre-COVID (2018-19)', 'Post-COVID (2020-21)', 'Change'],
                        'Treatment Group': [
                            f"{did_results['treatment_pre']:,.2f}",
                            f"{did_results['treatment_post']:,.2f}",
                            f"{did_results['treatment_change']:,.2f}"
                        ],
                        'Control Group': [
                            f"{did_results['control_pre']:,.2f}",
                            f"{did_results['control_post']:,.2f}",
                            f"{did_results['control_change']:,.2f}"
                        ]
                    })
                    st.dataframe(results_df, use_container_width=True)
                    
                    # DiD effect
                    st.metric(
                        label="DiD Effect (Treatment - Control)",
                        value=f"{did_results['did_effect']:,.4f}",
                        delta=f"p-value: {did_results['p_value']:.4f}" if did_results.get('p_value') else "N/A"
                    )
                    
                    if did_results.get('p_value'):
                        if did_results['p_value'] < 0.05:
                            st.success("✅ **Statistically significant at 5% level**")
                            st.info("The treatment group showed significantly different changes compared to the control group.")
                        else:
                            st.warning("⚠️ **Not statistically significant at 5% level**")
                            st.info("No significant difference detected between treatment and control groups.")
                
                with col2:
                    st.markdown("#### DiD Visualization")
                    
                    # Create DiD plot
                    fig = go.Figure()
                    
                    # Add lines for treatment and control
                    fig.add_trace(go.Scatter(
                        x=['Pre-COVID', 'Post-COVID'],
                        y=[did_results['treatment_pre'], did_results['treatment_post']],
                        mode='lines+markers',
                        name='Treatment Group',
                        line=dict(color='red', width=3),
                        marker=dict(size=10)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=['Pre-COVID', 'Post-COVID'],
                        y=[did_results['control_pre'], did_results['control_post']],
                        mode='lines+markers',
                        name='Control Group',
                        line=dict(color='blue', width=3),
                        marker=dict(size=10)
                    ))
                    
                    # Add DiD effect annotation
                    fig.add_annotation(
                        x=1, y=did_results['treatment_post'],
                        text=f"DiD Effect: {did_results['did_effect']:.4f}",
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-40
                    )
                    
                    fig.update_layout(
                        title=f"DiD Analysis: {st.session_state.selected_outcome.replace('_', ' ').title()}",
                        xaxis_title="Period",
                        yaxis_title=st.session_state.selected_outcome.replace('_', ' ').title(),
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Parallel trends test
                st.markdown("#### Parallel Trends Assumption Test")
                trend_data = did_analyzer.parallel_trends_test(st.session_state.selected_outcome)
                
                fig_trend = px.line(
                    trend_data,
                    x='year',
                    y=st.session_state.selected_outcome,
                    color='treatment_group',
                    markers=True,
                    title=f"Trend of {st.session_state.selected_outcome.replace('_', ' ').title()} Over Time",
                    labels={'treatment_group': 'Treatment Group', st.session_state.selected_outcome: st.session_state.selected_outcome.replace('_', ' ').title()}
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Regression output
                if did_results.get('model'):
                    with st.expander("📊 Regression Output"):
                        st.text(did_results['model'].summary())
            
            with tab3:
                # Advanced Metrics
                st.markdown('<h2 class="section-header">Advanced Financial Metrics Analysis</h2>', unsafe_allow_html=True)
                
                # Select metric to analyze
                advanced_metrics = [
                    'abnormal_accruals', 'beneish_m_score', 'financial_distress',
                    'earnings_quality', 'profit_margin', 'roa', 'roe'
                ]
                
                selected_metric = st.selectbox(
                    "Select Advanced Metric to Analyze",
                    advanced_metrics,
                    index=0
                )
                
                # Create comparison visualization
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        f'{selected_metric.replace("_", " ").title()} Distribution',
                        'Treatment vs Control Comparison',
                        'Year-wise Trend',
                        'Risk Level Analysis'
                    )
                )
                
                # Distribution plot
                fig.add_trace(
                    go.Histogram(
                        x=df_analyzed[selected_metric],
                        nbinsx=30,
                        name='Distribution',
                        marker_color='lightblue'
                    ),
                    row=1, col=1
                )
                
                # Box plot by treatment group
                fig.add_trace(
                    go.Box(
                        x=df_analyzed['treatment_group'],
                        y=df_analyzed[selected_metric],
                        name='Treatment vs Control',
                        boxpoints='outliers'
                    ),
                    row=1, col=2
                )
                
                # Line plot by year
                yearly_avg = df_analyzed.groupby(['year', 'treatment_group'])[selected_metric].mean().reset_index()
                for group in [0, 1]:
                    group_data = yearly_avg[yearly_avg['treatment_group'] == group]
                    fig.add_trace(
                        go.Scatter(
                            x=group_data['year'],
                            y=group_data[selected_metric],
                            mode='lines+markers',
                            name=f'Group {group}',
                            line=dict(width=2)
                        ),
                        row=2, col=1
                    )
                
                # Violin plot by risk level
                fig.add_trace(
                    go.Violin(
                        x=df_analyzed['manipulation_risk'],
                        y=df_analyzed[selected_metric],
                        name='Risk Level',
                        box_visible=True,
                        meanline_visible=True
                    ),
                    row=2, col=2
                )
                
                fig.update_layout(height=800, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                st.markdown("#### 📝 Interpretation Guidelines")
                
                interpretation_guide = {
                    'abnormal_accruals': "Values far from 0 suggest potential earnings management. Positive values may indicate income-increasing accruals.",
                    'beneish_m_score': "Score > -2.22 suggests higher probability of earnings manipulation. Higher values indicate greater risk.",
                    'financial_distress': "Z-Score < 1.81 indicates high distress risk, 1.81-2.99 is gray area, >2.99 is safe zone.",
                    'earnings_quality': "Ratio < 1 suggests earnings are not supported by cash flows. Lower values indicate poorer quality.",
                    'profit_margin': "Compare pre vs post-COVID changes. Unusual increases during crisis may warrant investigation.",
                    'roa': "Return on Assets. Sudden improvements without operational changes may indicate manipulation.",
                    'roe': "Return on Equity. Watch for unsustainable increases through leverage or accruals."
                }
                
                st.info(f"**{selected_metric.replace('_', ' ').title()}**: {interpretation_guide.get(selected_metric, 'No specific interpretation available.')}")
            
            with tab4:
                # Trend Analysis
                st.markdown('<h2 class="section-header">Time-Series Trend Analysis</h2>', unsafe_allow_html=True)
                
                # Select variables for comparison
                var1, var2 = st.columns(2)
                with var1:
                    variable1 = st.selectbox(
                        "Primary Variable",
                        ['revenue', 'net_profit', 'accruals', 'operating_cash_flow'],
                        index=1
                    )
                with var2:
                    variable2 = st.selectbox(
                        "Secondary Variable",
                        ['profit_margin', 'roa', 'roe', 'current_ratio'],
                        index=0
                    )
                
                # Create animated bubble chart
                yearly_agg = df_analyzed.groupby(['year', 'treatment_group', 'manipulation_risk']).agg({
                    variable1: 'mean',
                    variable2: 'mean',
                    'company_id': 'count'
                }).reset_index()
                
                fig = px.scatter(
                    yearly_agg,
                    x=variable1,
                    y=variable2,
                    size='company_id',
                    color='manipulation_risk',
                    animation_frame='year',
                    hover_name='treatment_group',
                    size_max=50,
                    title=f"Evolution of {variable1.replace('_', ' ').title()} vs {variable2.replace('_', ' ').title()} (2018-2021)",
                    labels={
                        variable1: variable1.replace('_', ' ').title(),
                        variable2: variable2.replace('_', ' ').title(),
                        'manipulation_risk': 'Manipulation Risk'
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Heatmap of correlations
                st.markdown("#### Correlation Heatmap")
                
                numeric_cols = df_analyzed.select_dtypes(include=[np.number]).columns
                corr_matrix = df_analyzed[numeric_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu',
                    title="Correlation Matrix of Financial Metrics"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with tab5:
                # Full Report
                st.markdown('<h2 class="section-header">Comprehensive Analysis Report</h2>', unsafe_allow_html=True)
                
                # Generate report summary
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 📋 Executive Summary")
                    
                    # Calculate key findings
                    high_risk_companies = df_analyzed[df_analyzed['manipulation_risk'] == 'High']['company_name'].unique()
                    avg_abnormal_accruals_treatment = df_analyzed[df_analyzed['treatment_group'] == 1]['abnormal_accruals'].mean()
                    avg_abnormal_accruals_control = df_analyzed[df_analyzed['treatment_group'] == 0]['abnormal_accruals'].mean()
                    
                    report_text = f"""
                    ## COVID-Era Earnings Analysis Report
                    
                    ### Key Findings:
                    
                    1. **DiD Analysis Result**: The treatment group showed a DiD effect of **{did_results['did_effect']:.4f}** on {st.session_state.selected_outcome.replace('_', ' ')}.
                    
                    2. **Manipulation Risk Assessment**:
                       - High-risk companies: **{len(high_risk_companies)}** out of {df_analyzed['company_id'].nunique()}
                       - Treatment group abnormal accruals: **{avg_abnormal_accruals_treatment:.4f}**
                       - Control group abnormal accruals: **{avg_abnormal_accruals_control:.4f}**
                    
                    3. **Statistical Significance**:
                       - DiD p-value: **{did_results.get('p_value', 'N/A'):.4f}**
                       - {"✅ Statistically significant" if did_results.get('p_value', 1) < 0.05 else "⚠️ Not statistically significant"}
                    
                    4. **Trend Analysis**:
                       - Treatment group change: **{did_results['treatment_change']:.2f}**
                       - Control group change: **{did_results['control_change']:.2f}**
                    
                    ### Recommendations:
                    
                    {"1. **Further investigation recommended** for treatment group companies showing high abnormal accruals and Beneish M-Score alerts." if avg_abnormal_accruals_treatment > avg_abnormal_accruals_control else "1. No significant evidence of systematic earnings manipulation detected."}
                    
                    2. Monitor companies with High manipulation risk in quarterly reports.
                    
                    3. Consider qualitative factors: industry conditions, governance quality, auditor opinions.
                    
                    ### Limitations:
                    
                    - Analysis based on available financial data only
                    - Does not prove manipulation, only indicates risk
                    - Consider economic context and industry-specific factors
                    """
                    
                    st.markdown(report_text)
                
                with col2:
                    st.markdown("### 📊 Risk Distribution")
                    
                    risk_counts = df_analyzed['manipulation_risk'].value_counts()
                    fig_pie = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Manipulation Risk Distribution",
                        color=risk_counts.index,
                        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    st.markdown("### 🏢 High-Risk Companies")
                    if len(high_risk_companies) > 0:
                        for company in high_risk_companies[:5]:  # Show top 5
                            st.warning(f"⚠️ {company}")
                    else:
                        st.success("✅ No high-risk companies identified")
                
                # Download report
                st.markdown("---")
                if st.button("📥 Download Complete Report", use_container_width=True):
                    # Create downloadable report
                    report_data = {
                        'did_results': did_results,
                        'risk_distribution': df_analyzed['manipulation_risk'].value_counts().to_dict(),
                        'high_risk_companies': list(high_risk_companies),
                        'analysis_date': pd.Timestamp.now().strftime("%Y-%m-%d")
                    }
                    
                    # Convert to JSON for download
                    import json
                    report_json = json.dumps(report_data, indent=2)
                    
                    st.download_button(
                        label="Download Report JSON",
                        data=report_json,
                        file_name=f"earnings_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

if __name__ == "__main__":
    main()
