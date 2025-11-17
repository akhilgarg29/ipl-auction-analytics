# dashboard.py - Save this as a separate file and run with: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="IPL Auction Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load data function with caching
@st.cache_data
def load_data():
    """Load and prepare all datasets"""
    auction_df = pd.read_csv('data/cleaned/ipl_auction_cleaned.csv')
    batting_stats = pd.read_csv('data/analysis/seasonwise_batting_stats.csv')
    bowling_stats = pd.read_csv('data/analysis/seasonwise_bowling_stats.csv')
    matches = pd.read_csv('data/cleaned/ipl_matches_cleaned.csv')
    
    # Normalize seasons
    def normalize_season(season):
        season_str = str(season).strip()
        if '/' in season_str:
            parts = season_str.split('/')
            first_year = int(parts[0])
            second_part = parts[1]
            if first_year == 2020:
                return 2020
            if len(second_part) == 2:
                return int(str(first_year)[:2] + second_part)
            else:
                return int(second_part)
        else:
            try:
                return int(float(season_str))
            except:
                return 0
    
    for df in [auction_df, batting_stats, bowling_stats, matches]:
        if 'Season' not in df.columns and 'Year' in df.columns:
            df.rename(columns={'Year': 'Season'}, inplace=True)
        df['Season'] = df['Season'].apply(normalize_season)
        df = df[df['Season'] > 0]
    
    return auction_df, batting_stats, bowling_stats, matches

# Load data
auction_df, batting_stats, bowling_stats, matches = load_data()

# Merge performance with auction data
@st.cache_data
def merge_performance_data(_auction_df, _batting_stats, _bowling_stats):
    """Merge auction data with performance stats"""
    
    # Get batting columns
    batting_cols = ['Player', 'Season']
    for col in ['Runs_Scored', 'Batting_Average', 'Strike_Rate', 'Matches', 'Innings']:
        if col in _batting_stats.columns:
            batting_cols.append(col)
    
    # Get bowling columns
    bowling_cols = ['Player', 'Season']
    for col in ['Wickets_Taken', 'Economy_Rate', 'Strike_Rate', 'Bowling_Average']:
        if col in _bowling_stats.columns:
            bowling_cols.append(col)
    
    # Merge
    merged = _auction_df.merge(
        _batting_stats[batting_cols],
        on=['Player', 'Season'],
        how='left'
    )
    
    bowling_merge_cols = [col for col in bowling_cols if col not in ['Player', 'Season']]
    if bowling_merge_cols:
        merged = merged.merge(
            _bowling_stats[['Player', 'Season'] + bowling_merge_cols],
            on=['Player', 'Season'],
            how='left',
            suffixes=('', '_bowl')
        )
    
    # Fill NaN
    for col in merged.columns:
        if col not in ['Player', 'Team', 'Nationality', 'Role', 'Price', 'Season']:
            merged[col] = merged[col].fillna(0)
    
    # Calculate value metrics
    if 'Runs_Scored' in merged.columns and 'Price' in merged.columns:
        merged['Value_Ratio'] = np.where(
            merged['Price'] > 0,
            (merged['Runs_Scored'] * 0.01 + merged.get('Wickets_Taken', 0) * 0.5) / merged['Price'],
            0
        )
    else:
        merged['Value_Ratio'] = 0
    
    return merged

performance_df = merge_performance_data(auction_df, batting_stats, bowling_stats)

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<div class="main-header">🏏 IPL Auction Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# SIDEBAR - FILTERS
# ============================================================================
st.sidebar.header("🔍 Filters")

# Season filter
available_seasons = sorted(auction_df['Season'].unique())
selected_season = st.sidebar.selectbox(
    "Select Season",
    options=available_seasons,
    index=len(available_seasons)-1  # Default to latest season
)

# Team filter
all_teams = ['All Teams'] + sorted(auction_df['Team'].dropna().unique().tolist())
selected_team = st.sidebar.selectbox("Select Team", options=all_teams)

# Role filter
if 'Role' in auction_df.columns:
    all_roles = ['All Roles'] + sorted(auction_df['Role'].dropna().unique().tolist())
    selected_role = st.sidebar.selectbox("Select Role", options=all_roles)
else:
    selected_role = 'All Roles'

# Nationality filter
all_nationalities = ['All'] + sorted(auction_df['Nationality'].dropna().unique().tolist())
selected_nationality = st.sidebar.selectbox("Select Nationality", options=all_nationalities)

# Filter data
filtered_df = performance_df[performance_df['Season'] == selected_season].copy()

if selected_team != 'All Teams':
    filtered_df = filtered_df[filtered_df['Team'] == selected_team]

if selected_role != 'All Roles' and 'Role' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Role'] == selected_role]

if selected_nationality != 'All':
    filtered_df = filtered_df[filtered_df['Nationality'] == selected_nationality]

# ============================================================================
# KEY METRICS
# ============================================================================
st.header(f"📊 Season {selected_season} Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_players = len(filtered_df)
    st.metric("Total Players", total_players)

with col2:
    total_spent = filtered_df['Price'].sum()
    st.metric("Total Spent", f"₹{total_spent:.1f} Cr")

with col3:
    avg_price = filtered_df['Price'].mean()
    st.metric("Avg Price", f"₹{avg_price:.2f} Cr")

with col4:
    highest_paid = filtered_df['Price'].max()
    st.metric("Highest Paid", f"₹{highest_paid:.2f} Cr")

st.markdown("---")

# ============================================================================
# TAB LAYOUT
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["🏏 Player Analysis", "💰 Value Analysis", "📈 Performance Trends", "🔍 Player Search"])

# ============================================================================
# TAB 1: PLAYER ANALYSIS
# ============================================================================
with tab1:
    st.subheader(f"Player Distribution - Season {selected_season}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution
        fig_price = px.histogram(
            filtered_df,
            x='Price',
            nbins=20,
            title='Price Distribution',
            labels={'Price': 'Price (Crores)'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_price.update_layout(showlegend=False)
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        # Top 10 expensive players
        top_10 = filtered_df.nlargest(10, 'Price')[['Player', 'Price', 'Team']]
        fig_top = px.bar(
            top_10,
            x='Price',
            y='Player',
            orientation='h',
            title='Top 10 Most Expensive Players',
            color='Price',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_top, use_container_width=True)
    
    # Team-wise spending
    st.subheader("Team-wise Auction Spending")
    team_spending = filtered_df.groupby('Team')['Price'].agg(['sum', 'mean', 'count']).reset_index()
    team_spending.columns = ['Team', 'Total Spent', 'Avg Price', 'Players']
    team_spending = team_spending.sort_values('Total Spent', ascending=False)
    
    fig_team = px.bar(
        team_spending,
        x='Team',
        y='Total Spent',
        title='Total Spending by Team',
        color='Total Spent',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_team, use_container_width=True)
    
    # Data table
    st.subheader("Detailed Player Data")
    display_cols = ['Player', 'Team', 'Price', 'Nationality']
    if 'Role' in filtered_df.columns:
        display_cols.append('Role')
    if 'Runs_Scored' in filtered_df.columns:
        display_cols.append('Runs_Scored')
    if 'Wickets_Taken' in filtered_df.columns:
        display_cols.append('Wickets_Taken')
    
    st.dataframe(
        filtered_df[display_cols].sort_values('Price', ascending=False),
        use_container_width=True,
        height=400
    )

# ============================================================================
# TAB 2: VALUE ANALYSIS
# ============================================================================
with tab2:
    st.subheader("💎 Player Value Analysis - Justified vs Overpriced")
    
    # Calculate performance metrics
    value_df = filtered_df.copy()
    
    if 'Runs_Scored' in value_df.columns:
        # Performance score
        value_df['Performance_Score'] = (
            value_df.get('Runs_Scored', 0) * 0.5 +
            value_df.get('Wickets_Taken', 0) * 30 +
            value_df.get('Batting_Average', 0) * 1
        )
        
        # Calculate expected price based on performance percentile
        value_df['Performance_Percentile'] = value_df['Performance_Score'].rank(pct=True)
        value_df['Expected_Price'] = value_df['Performance_Percentile'] * value_df['Price'].max()
        
        # Price gap
        value_df['Price_Gap'] = value_df['Price'] - value_df['Expected_Price']
        value_df['Value_Category'] = pd.cut(
            value_df['Price_Gap'],
            bins=[-np.inf, -2, 2, np.inf],
            labels=['Undervalued', 'Fair Value', 'Overvalued']
        )
        
        # Value summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            undervalued_count = (value_df['Value_Category'] == 'Undervalued').sum()
            st.metric("🟢 Undervalued Players", undervalued_count)
        
        with col2:
            fair_count = (value_df['Value_Category'] == 'Fair Value').sum()
            st.metric("🟡 Fair Value Players", fair_count)
        
        with col3:
            overvalued_count = (value_df['Value_Category'] == 'Overvalued').sum()
            st.metric("🔴 Overvalued Players", overvalued_count)
        
        # Scatter plot: Price vs Performance
        fig_scatter = px.scatter(
            value_df,
            x='Performance_Score',
            y='Price',
            color='Value_Category',
            size='Price',
            hover_data=['Player', 'Team'],
            title='Price vs Performance Analysis',
            labels={'Performance_Score': 'Performance Score', 'Price': 'Auction Price (Crores)'},
            color_discrete_map={
                'Undervalued': '#2ecc71',
                'Fair Value': '#f39c12',
                'Overvalued': '#e74c3c'
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Top undervalued and overvalued
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Top 10 Undervalued Players")
            undervalued = value_df[value_df['Value_Category'] == 'Undervalued'].nlargest(10, 'Performance_Score')
            display_undervalued = undervalued[['Player', 'Team', 'Price', 'Performance_Score', 'Price_Gap']].copy()
            display_undervalued['Price_Gap'] = display_undervalued['Price_Gap'].round(2)
            st.dataframe(display_undervalued, use_container_width=True)
        
        with col2:
            st.subheader("🔴 Top 10 Overvalued Players")
            overvalued = value_df[value_df['Value_Category'] == 'Overvalued'].nlargest(10, 'Price_Gap')
            display_overvalued = overvalued[['Player', 'Team', 'Price', 'Performance_Score', 'Price_Gap']].copy()
            display_overvalued['Price_Gap'] = display_overvalued['Price_Gap'].round(2)
            st.dataframe(display_overvalued, use_container_width=True)
    
    else:
        st.info("Performance data not available for value analysis.")

# ============================================================================
# TAB 3: PERFORMANCE TRENDS
# ============================================================================
with tab3:
    st.subheader("📈 Multi-Season Performance Trends")
    
    # Season-wise trends
    season_trends = performance_df.groupby('Season').agg({
        'Price': ['mean', 'sum', 'count'],
        'Runs_Scored': 'sum' if 'Runs_Scored' in performance_df.columns else lambda x: 0,
        'Wickets_Taken': 'sum' if 'Wickets_Taken' in performance_df.columns else lambda x: 0
    }).reset_index()
    
    season_trends.columns = ['Season', 'Avg_Price', 'Total_Spent', 'Players', 'Total_Runs', 'Total_Wickets']
    
    # Price trends
    fig_trends = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Price Trend', 'Total Spending Trend', 
                       'Total Runs per Season', 'Total Wickets per Season')
    )
    
    fig_trends.add_trace(
        go.Scatter(x=season_trends['Season'], y=season_trends['Avg_Price'], 
                  mode='lines+markers', name='Avg Price'),
        row=1, col=1
    )
    
    fig_trends.add_trace(
        go.Scatter(x=season_trends['Season'], y=season_trends['Total_Spent'], 
                  mode='lines+markers', name='Total Spent'),
        row=1, col=2
    )
    
    fig_trends.add_trace(
        go.Bar(x=season_trends['Season'], y=season_trends['Total_Runs'], name='Total Runs'),
        row=2, col=1
    )
    
    fig_trends.add_trace(
        go.Bar(x=season_trends['Season'], y=season_trends['Total_Wickets'], name='Total Wickets'),
        row=2, col=2
    )
    
    fig_trends.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Nationality trends
    st.subheader("Nationality-wise Auction Trends")
    nationality_trends = performance_df.groupby(['Season', 'Nationality'])['Price'].mean().reset_index()
    top_nationalities = performance_df.groupby('Nationality')['Price'].sum().nlargest(5).index
    nationality_trends_filtered = nationality_trends[nationality_trends['Nationality'].isin(top_nationalities)]
    
    fig_nat = px.line(
        nationality_trends_filtered,
        x='Season',
        y='Price',
        color='Nationality',
        title='Top 5 Nationalities - Average Price Trend',
        markers=True
    )
    st.plotly_chart(fig_nat, use_container_width=True)

# ============================================================================
# TAB 4: PLAYER SEARCH
# ============================================================================
with tab4:
    st.subheader("🔍 Individual Player Analysis")
    
    # Player search
    all_players = sorted(performance_df['Player'].unique())
    selected_player = st.selectbox("Search Player", options=all_players)
    
    if selected_player:
        player_data = performance_df[performance_df['Player'] == selected_player].sort_values('Season')
        
        if len(player_data) > 0:
            # Player summary
            st.markdown(f"### {selected_player}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                seasons_played = len(player_data)
                st.metric("Seasons Played", seasons_played)
            
            with col2:
                total_price = player_data['Price'].sum()
                st.metric("Total Earnings", f"₹{total_price:.1f} Cr")
            
            with col3:
                if 'Runs_Scored' in player_data.columns:
                    total_runs = player_data['Runs_Scored'].sum()
                    st.metric("Total Runs", int(total_runs))
                else:
                    st.metric("Total Runs", "N/A")
            
            with col4:
                if 'Wickets_Taken' in player_data.columns:
                    total_wickets = player_data['Wickets_Taken'].sum()
                    st.metric("Total Wickets", int(total_wickets))
                else:
                    st.metric("Total Wickets", "N/A")
            
            # Player price history
            fig_player = px.line(
                player_data,
                x='Season',
                y='Price',
                markers=True,
                title=f"{selected_player} - Price History",
                labels={'Price': 'Price (Crores)', 'Season': 'Season'}
            )
            st.plotly_chart(fig_player, use_container_width=True)
            
            # Player performance by season
            st.subheader("Season-wise Performance")
            display_player_cols = ['Season', 'Team', 'Price']
            for col in ['Runs_Scored', 'Batting_Average', 'Strike_Rate', 'Wickets_Taken', 'Economy_Rate']:
                if col in player_data.columns:
                    display_player_cols.append(col)
            
            st.dataframe(player_data[display_player_cols], use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    IPL Auction Analytics Dashboard | Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
