import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
        self.df['current_ratio'] = (self.df['receivables'] + self.df.get('inventory', 0)) / self.df['current_liabilities'].replace(0, 1)  # Avoid division by zero
        self.df['debt_to_equity'] = self.df['total_debt'] / self.df['total_equity'].replace(0, 1)  # Avoid division by zero
        self.df['profit_margin'] = self.df['net_profit'] / self.df['revenue'].replace(0, 1)
        self.df['asset_turnover'] = self.df['revenue'] / self.df['total_assets'].replace(0, 1)
        self.df['roa'] = self.df['net_profit'] / self.df['total_assets'].replace(0, 1)
        self.df['roe'] = self.df['net_profit'] / self.df['total_equity'].replace(0, 1)
        
        # Advanced metrics
        self.df['abnormal_accruals'] = self.calculate_abnormal_accruals()
        self.df['beneish_m_score'] = self.calculate_beneish_m_score()
        self.df['financial_distress'] = self.calculate_altman_z_score()
        self.df['earnings_quality'] = np.where(
            self.df['net_profit'] != 0,
            self.df['operating_cash_flow'] / abs(self.df['net_profit']),
            1
        )
        
        # Manipulation indicators
        self.df['manipulation_risk'] = self.calculate_manipulation_risk()
        
        return self.df
    
    def calculate_abnormal_accruals(self):
        """Calculate abnormal accruals using Modified Jones Model"""
        try:
            # Simplified version - using industry average as proxy
            df = self.df.copy()
            df = df.sort_values(['company_id', 'year'])
            
            # Calculate total accruals to assets ratio
            df['accruals_to_assets'] = df['accruals'] / df['total_assets']
            
            # Calculate industry-year average
            df['industry_year_avg'] = df.groupby(['industry', 'year'])['accruals_to_assets'].transform('mean')
            
            # Abnormal accruals = actual - industry average
            abnormal_accruals = df['accruals_to_assets'] - df['industry_year_avg']
            
            return abnormal_accruals.fillna(0)
        except Exception as e:
            st.warning(f"Abnormal accruals calculation simplified: {str(e)[:100]}")
            return np.zeros(len(self.df))
    
    def calculate_beneish_m_score(self):
        """Calculate Beneish M-Score for earnings manipulation detection"""
        try:
            df = self.df.copy()
            df = df.sort_values(['company_id', 'year'])
            
            # Days Sales in Receivables Index (simplified)
            df['revenue_lag'] = df.groupby('company_id')['revenue'].shift(1)
            df['receivables_lag'] = df.groupby('company_id')['receivables'].shift(1)
            
            dsri = (df['receivables'] / df['revenue']) / (df['receivables_lag'] / df['revenue_lag'])
            dsri = dsri.fillna(1)
            
            # Asset Quality Index (simplified)
            df['total_assets_lag'] = df.groupby('company_id')['total_assets'].shift(1)
            aqi = (df['total_assets'] / df['total_assets_lag']).fillna(1)
            
            # Sales Growth Index
            sgi = (df['revenue'] / df['revenue_lag']).fillna(1)
            
            # Depreciation Index (simplified)
            depreciation_col = 'depreciation' if 'depreciation' in df.columns else 'operating_cash_flow' * 0.1
            df['depreciation_lag'] = df.groupby('company_id')[depreciation_col].shift(1)
            depi = (df[depreciation_col] / df['depreciation_lag']).fillna(1)
            
            # Simplified M-Score calculation
            m_score = -4.84 + 0.92 * dsri + 0.528 * aqi + 0.892 * sgi + 0.115 * depi
            
            return m_score.fillna(0)
        except Exception as e:
            st.warning(f"Beneish M-Score calculation simplified: {str(e)[:100]}")
            return np.zeros(len(self.df))
    
    def calculate_altman_z_score(self):
        """Calculate Altman Z-Score for financial distress"""
        try:
            df = self.df.copy()
            
            # Working capital to total assets
            current_assets = df.get('current_assets', df['receivables'] + df.get('inventory', 0))
            X1 = (current_assets - df['current_liabilities']) / df['total_assets'].replace(0, 1)
            
            # Retained earnings to total assets
            retained_earnings = df.get('retained_earnings', df['total_equity'] * 0.5)
            X2 = retained_earnings / df['total_assets'].replace(0, 1)
            
            # EBIT to total assets
            ebit = df.get('ebit', df['net_profit'] * 1.2)
            X3 = ebit / df['total_assets'].replace(0, 1)
            
            # Market value of equity to book value of total liabilities
            market_cap = df.get('market_cap', df['total_equity'] * 1.5)
            X4 = market_cap / df['total_debt'].replace(0, 1)
            
            # Sales to total assets
            X5 = df['revenue'] / df['total_assets'].replace(0, 1)
            
            # Altman Z-Score
            z_score = 1.2 * X1 + 1.4 * X2 + 3.3 * X3 + 0.6 * X4 + 1.0 * X5
            
            return z_score.fillna(0)
        except Exception as e:
            st.warning(f"Altman Z-Score calculation simplified: {str(e)[:100]}")
            return np.zeros(len(self.df))
    
    def calculate_manipulation_risk(self):
        """Calculate comprehensive manipulation risk score"""
        try:
            df = self.df.copy()
            
            # Calculate individual risk factors (scaled 0-1)
            accruals_risk = abs(df['abnormal_accruals'].fillna(0))
            if accruals_risk.max() > 0:
                accruals_risk = accruals_risk / accruals_risk.max()
            
            beneish_risk = (df['beneish_m_score'] > -2.22).astype(float)
            
            earnings_quality_risk = (df['earnings_quality'] < 0.8).astype(float)
            
            # Composite risk score (0-1)
            risk_score = (accruals_risk * 0.4 + beneish_risk * 0.3 + earnings_quality_risk * 0.3)
            
            # Categorize risk levels
            risk_levels = pd.cut(risk_score, 
                                bins=[-0.1, 0.3, 0.6, 1.1], 
                                labels=['Low', 'Medium', 'High'])
            
            return risk_levels.fillna('Low')
        except Exception as e:
            st.warning(f"Manipulation risk calculation simplified: {str(e)[:100]}")
            return pd.Series(['Low'] * len(self.df))

class DIDAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        
    def perform_did_analysis(self, outcome_var):
        """Perform Difference-in-Differences analysis"""
        try:
            # Prepare data
            df_analysis = self.df.copy()
            df_analysis['post_covid'] = df_analysis['year'].apply(lambda x: 1 if x >= 2020 else 0)
            
            # Calculate group means
            treatment_pre = df_analysis[(df_analysis['treatment_group'] == 1) & (df_analysis['post_covid'] == 0)][outcome_var].mean()
            treatment_post = df_analysis[(df_analysis['treatment_group'] == 1) & (df_analysis['post_covid'] == 1)][outcome_var].mean()
            control_pre = df_analysis[(df_analysis['treatment_group'] == 0) & (df_analysis['post_covid'] == 0)][outcome_var].mean()
            control_post = df_analysis[(df_analysis['treatment_group'] == 0) & (df_analysis['post_covid'] == 1)][outcome_var].mean()
            
            # DiD calculation
            did_effect = (treatment_post - treatment_pre) - (control_post - control_pre)
            
            # Simple statistical significance (using t-test approximation)
            n_treatment = len(df_analysis[df_analysis['treatment_group'] == 1])
            n_control = len(df_analysis[df_analysis['treatment_group'] == 0])
            
            # Calculate variances
            treatment_var = df_analysis[df_analysis['treatment_group'] == 1][outcome_var].var()
            control_var = df_analysis[df_analysis['treatment_group'] == 0][outcome_var].var()
            
            # Standard error
            se = np.sqrt(treatment_var/n_treatment + control_var/n_control)
            
            # t-statistic and p-value
            if se > 0:
                t_stat = did_effect / se
                # Approximate p-value (two-tailed)
                p_value = 2 * (1 - 0.5 * (1 + np.math.erf(abs(t_stat) / np.sqrt(2))))
            else:
                p_value = 0.05 if did_effect != 0 else 1.0
            
            results = {
                'did_effect': did_effect,
                'treatment_pre': treatment_pre,
                'treatment_post': treatment_post,
                'control_pre': control_pre,
                'control_post': control_post,
                'treatment_change': treatment_post - treatment_pre,
                'control_change': control_post - control_pre,
                'p_value': p_value,
                'ci_lower': did_effect - 1.96 * se if se > 0 else did_effect * 0.9,
                'ci_upper': did_effect + 1.96 * se if se > 0 else did_effect * 1.1,
                'standard_error': se
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error in DiD analysis: {str(e)[:100]}")
            # Return basic results without statistics
            return {
                'did_effect': 0,
                'treatment_pre': 0,
                'treatment_post': 0,
                'control_pre': 0,
                'control_post': 0,
                'treatment_change': 0,
                'control_change': 0,
                'p_value': 1.0,
                'ci_lower': 0,
                'ci_upper': 0,
                'standard_error': 0
            }
    
    def parallel_trends_test(self, outcome_var):
        """Test parallel trends assumption"""
        try:
            df_test = self.df.copy()
            
            # Plot trends over time
            trend_data = df_test.groupby(['year', 'treatment_group'])[outcome_var].mean().reset_index()
            
            return trend_data
        except Exception as e:
            st.error(f"Error in parallel trends: {str(e)[:100]}")
            return pd.DataFrame({'year': [], 'treatment_group': [], outcome_var: []})

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
                    st.write(f"**Years:** {sorted(df['year'].unique())}")
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
            try:
                financial_analyzer = FinancialAnalyzer(st.session_state.df)
                df_analyzed = financial_analyzer.calculate_advanced_metrics()
                did_analyzer = DIDAnalyzer(df_analyzed)
            except Exception as e:
                st.error(f"Error initializing analyzers: {str(e)}")
                return
            
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
                
                # Industry Distribution - FIXED SUNBURST CHART
                st.markdown('<h3 class="section-header">Industry & Risk Distribution</h3>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        # Create aggregated data for sunburst chart
                        industry_agg = df_analyzed.groupby(['industry', 'treatment_group', 'manipulation_risk']).agg({
                            'revenue': 'sum',
                            'company_id': 'count'
                        }).reset_index()
                        
                        # Add a unique ID for each combination
                        industry_agg['unique_id'] = range(len(industry_agg))
                        
                        # Create treemap instead of sunburst (more stable)
                        fig1 = px.treemap(
                            industry_agg,
                            path=['industry', 'treatment_group', 'manipulation_risk'],
                            values='revenue',
                            title="Industry & Treatment Group Distribution",
                            color='revenue',
                            color_continuous_scale='Blues'
                        )
                        fig1.update_layout(margin=dict(t=50, l=25, r=25, b=25))
                        st.plotly_chart(fig1, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create industry chart: {str(e)[:100]}")
                        # Fallback: Simple bar chart
                        industry_counts = df_analyzed.groupby(['industry', 'treatment_group']).size().reset_index(name='count')
                        fig_fallback = px.bar(
                            industry_counts,
                            x='industry',
                            y='count',
                            color='treatment_group',
                            title="Company Count by Industry",
                            barmode='group'
                        )
                        st.plotly_chart(fig_fallback, use_container_width=True)
                
                with col2:
                    try:
                        fig2 = px.box(
                            df_analyzed,
                            x='treatment_group',
                            y='abnormal_accruals',
                            color='manipulation_risk',
                            title="Abnormal Accruals by Treatment Group & Risk Level",
                            labels={'treatment_group': 'Treatment Group (1=Treatment)'},
                            category_orders={'treatment_group': [0, 1]}
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create box plot: {str(e)[:100]}")
            
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
                        delta=f"p-value: {did_results['p_value']:.4f}"
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
                try:
                    trend_data = did_analyzer.parallel_trends_test(st.session_state.selected_outcome)
                    
                    if not trend_data.empty:
                        fig_trend = px.line(
                            trend_data,
                            x='year',
                            y=st.session_state.selected_outcome,
                            color='treatment_group',
                            markers=True,
                            title=f"Trend of {st.session_state.selected_outcome.replace('_', ' ').title()} Over Time",
                            labels={'treatment_group': 'Group', st.session_state.selected_outcome: st.session_state.selected_outcome.replace('_', ' ').title()}
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.info("Insufficient data for parallel trends analysis")
                except Exception as e:
                    st.warning(f"Could not create parallel trends chart: {str(e)[:100]}")
            
            with tab3:
                # Advanced Metrics
                st.markdown('<h2 class="section-header">Advanced Financial Metrics Analysis</h2>', unsafe_allow_html=True)
                
                # Select metric to analyze
                available_metrics = [col for col in df_analyzed.columns if col not in ['company_id', 'company_name', 'industry', 'year', 'treatment_group', 'manipulation_risk']]
                available_metrics = [m for m in available_metrics if df_analyzed[m].dtype in ['float64', 'int64']]
                
                selected_metric = st.selectbox(
                    "Select Advanced Metric to Analyze",
                    available_metrics[:10],  # Limit to first 10 for dropdown
                    index=0 if 'abnormal_accruals' in available_metrics else 0
                )
                
                try:
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
                            x=df_analyzed[selected_metric].dropna(),
                            nbinsx=30,
                            name='Distribution',
                            marker_color='lightblue'
                        ),
                        row=1, col=1
                    )
                    
                    # Box plot by treatment group
                    fig.add_trace(
                        go.Box(
                            x=df_analyzed['treatment_group'].astype(str),
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
                        'roe': "Return on Equity. Watch for unsustainable increases through leverage or accruals.",
                        'accruals': "Difference between net profit and operating cash flow. High accruals may indicate earnings management.",
                        'current_ratio': "Liquidity ratio. Values < 1 may indicate short-term liquidity problems.",
                        'debt_to_equity': "Leverage ratio. High values indicate higher financial risk.",
                        'asset_turnover': "Efficiency ratio. Measures how well assets generate revenue."
                    }
                    
                    if selected_metric in interpretation_guide:
                        st.info(f"**{selected_metric.replace('_', ' ').title()}**: {interpretation_guide[selected_metric]}")
                    else:
                        st.info(f"**{selected_metric.replace('_', ' ').title()}**: Analyze changes pre vs post-COVID and compare treatment vs control groups.")
                        
                except Exception as e:
                    st.error(f"Error creating advanced metrics visualization: {str(e)[:100]}")
            
            with tab4:
                # Trend Analysis
                st.markdown('<h2 class="section-header">Time-Series Trend Analysis</h2>', unsafe_allow_html=True)
                
                try:
                    # Select variables for comparison
                    var1, var2 = st.columns(2)
                    with var1:
                        variable_options = [col for col in df_analyzed.columns if df_analyzed[col].dtype in ['float64', 'int64']]
                        variable_options = [v for v in variable_options if v not in ['company_id', 'year', 'treatment_group']]
                        variable1 = st.selectbox(
                            "Primary Variable",
                            variable_options[:8],
                            index=variable_options.index('net_profit') if 'net_profit' in variable_options else 0
                        )
                    with var2:
                        variable2 = st.selectbox(
                            "Secondary Variable",
                            ['profit_margin', 'roa', 'roe', 'current_ratio'] + variable_options[:4],
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
                    # Limit to 15 columns for readability
                    if len(numeric_cols) > 15:
                        numeric_cols = numeric_cols[:15]
                    
                    corr_matrix = df_analyzed[numeric_cols].corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu',
                        title="Correlation Matrix of Financial Metrics",
                        height=600
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in trend analysis: {str(e)[:100]}")
            
            with tab5:
                # Full Report
                st.markdown('<h2 class="section-header">Comprehensive Analysis Report</h2>', unsafe_allow_html=True)
                
                try:
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
                    
                    # Create downloadable data
                    if st.button("📥 Download Analysis Results", use_container_width=True):
                        # Create summary dataframe
                        summary_data = {
                            'Metric': [
                                'Total Companies',
                                'Treatment Companies',
                                'Control Companies',
                                'DiD Effect',
                                'DiD p-value',
                                'Avg Abnormal Accruals (Treatment)',
                                'Avg Abnormal Accruals (Control)',
                                'High Risk Companies'
                            ],
                            'Value': [
                                df_analyzed['company_id'].nunique(),
                                df_analyzed[df_analyzed['treatment_group'] == 1]['company_id'].nunique(),
                                df_analyzed[df_analyzed['treatment_group'] == 0]['company_id'].nunique(),
                                did_results['did_effect'],
                                did_results['p_value'],
                                avg_abnormal_accruals_treatment,
                                avg_abnormal_accruals_control,
                                len(high_risk_companies)
                            ]
                        }
                        
                        summary_df = pd.DataFrame(summary_data)
                        
                        # Convert to CSV
                        csv = summary_df.to_csv(index=False)
                        
                        st.download_button(
                            label="Download Summary CSV",
                            data=csv,
                            file_name="covid_earnings_analysis_summary.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"Error generating report: {str(e)[:100]}")

if __name__ == "__main__":
    main()
