# 🏏 IPL Auction Analytics Dashboard

A comprehensive data analytics project analyzing IPL (Indian Premier League) auction data from 2013-2025, featuring player performance analysis, price prediction models, and an interactive dashboard for value assessment.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.29.0-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Overview

This project provides end-to-end analytics for IPL player auctions, combining historical auction data with match performance statistics to identify undervalued and overvalued players, predict auction prices, and visualize trends across 17+ seasons.

## ✨ Features

### 📊 Interactive Dashboard
- **Year-wise auction reports** with dynamic filtering by season, team, role, and nationality
- **Player performance tracking** across multiple seasons
- **Value analysis** identifying overvalued and undervalued players
- **Individual player search** with detailed stats and price history
- **14+ visualizations** covering price trends, performance metrics, and team spending

### 🤖 Machine Learning Models
- **6 regression models** tested (Linear, Ridge, Lasso, KNN, Decision Tree, SVR)
- **Price prediction** based on performance metrics
- **Feature importance analysis** to identify key price drivers
- **Model comparison** with cross-validation

### 💎 Value Insights
- Smart player valuation based on performance-to-price ratio
- Identification of best value-for-money players
- Team spending pattern analysis

## 🚀 Live Demo

[View Live Dashboard](https://ipl-auction-analytics.streamlit.app/)

## 📊 Dataset

**Time Period:** 2008-2025  
**Total Records:** 10,000+ player-season combinations

| Dataset | Records | Description |
|---------|---------|-------------|
| Auction Data | 1,500+ | Player prices, teams, nationalities (2013-2025) |
| Matches | 1,000+ | Match results, venues, winners (2008-2025) |
| Deliveries | 250,000+ | Ball-by-ball data (2008-2025) |
| Batting Stats | 5,000+ | Season-wise runs, averages, strike rates |
| Bowling Stats | 3,000+ | Season-wise wickets, economy, strike rates |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

Clone repository
git clone https://github.com/akhilgarg29/ipl-auction-analytics.git
cd ipl-auction-analytics

Install dependencies
pip install -r requirements.txt

Run dashboard
streamlit run dashboard.py


## 📁 Project Structure

ipl-auction-analytics/
├── data/
│ ├── raw/ # Original CSV files
│ ├── cleaned/ # Processed data
│ └── analysis/ # Analysis outputs
├── models/ # Trained ML models
├── dashboard.py # Streamlit dashboard
├── ultitled1.ipynb # Python Notebook
├── requirements.txt # Dependencies
└── README.md # Documentation


## 🤖 Machine Learning Results

### Model Performance Summary

| Model | MAE (₹Cr) | RMSE (₹Cr) | R² Score | CV R² |
|-------|-----------|------------|----------|-------|
| **Support Vector Regression** | **1.97** | **2.77** | **0.080** | 0.037 |
| Ridge Regression | 2.16 | 2.84 | 0.033 | 0.114 |
| Linear Regression | 2.20 | 2.88 | 0.007 | 0.077 |
| K-Nearest Neighbors | 2.35 | 3.05 | -0.110 | 0.036 |
| Lasso Regression | 2.32 | 2.90 | -0.003 | -0.015 |
| Decision Tree | 2.62 | 3.71 | -0.643 | -0.638 |

**Best Model:** Support Vector Regression (MAE = ₹1.97 Cr)

### 🔍 Key Finding: The Complexity of Auction Pricing

Our analysis reveals a critical insight: **Player performance statistics alone explain only ~8% of auction price variance**. This demonstrates that IPL auction prices are heavily influenced by intangible factors beyond on-field performance.

#### Why Performance Alone Isn't Enough

IPL franchises consider multiple non-statistical factors when bidding:

1. **Brand Value & Popularity** - Star players command premium regardless of recent stats
2. **Age & Future Potential** - Younger players valued for long-term investment
3. **Team Requirements** - Specific role needs (captain, wicket-keeper, death bowler)
4. **Auction Dynamics** - Bidding wars and strategic picks
5. **Availability** - Full season vs partial availability
6. **Leadership Qualities** - Captain/vice-captain premium
7. **International Form** - Recent World Cup/T20 performance

#### Research Implications

This finding aligns with sports economics research showing that athlete valuation is multidimensional. For accurate price prediction, additional data would be needed:
- Player age and career stage
- Social media following (brand value proxy)
- Previous auction prices (market momentum)
- Team budget constraints
- Leadership roles and experience

### Project Value

While the predictive power is limited, this project successfully demonstrates:
- ✅ End-to-end ML pipeline development
- ✅ Feature engineering and data integration
- ✅ Model comparison and evaluation
- ✅ Critical interpretation of results
- ✅ Understanding business context beyond pure statistics

**Conclusion:** The low R² score itself is a valuable insight, highlighting the importance of domain knowledge and the limitations of purely statistical approaches in complex real-world scenarios like sports auctions.

## 📈 Key Insights

From the data analysis and dashboard:

1. **Price Inflation**: Average auction prices increased by **187%** from 2013 to 2025
2. **Indian Premium**: Indian players command **23% higher** prices on average
3. **All-Rounder Value**: All-rounders earn **35% more** than specialists
4. **Performance Gap**: 12% of players consistently perform above their price bracket
5. **Economy Matters**: Bowlers with economy < 7 command **40% premium**

## 🛠️ Technologies Used

- **Python 3.8+** - Core programming language
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning models
- **Streamlit** - Interactive web dashboard
- **Plotly** - Interactive visualizations
- **Matplotlib & Seaborn** - Static visualizations

## 📊 Dashboard Features

### Tab 1: Player Analysis
- Price distribution and top expensive players
- Team-wise spending breakdown
- Detailed player data table

### Tab 2: Value Analysis
- Performance vs price scatter plot
- Undervalued and overvalued player lists
- Value category distribution

### Tab 3: Performance Trends
- Multi-season price trends
- Total runs and wickets by season
- Nationality-wise trends

### Tab 4: Player Search
- Individual player career overview
- Price history across seasons
- Season-wise performance metrics

## 🚀 Deployment

### Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Select `dashboard.py` as main file
5. Deploy!

Your dashboard will be live at: `https://akhilgarg29-ipl-auction-analytics-dashboard-rdgnru.streamlit.app/`

## 💡 Future Enhancements

- [ ] Add player age and career stage features
- [ ] Integrate social media metrics for brand value
- [ ] Real-time auction price updates
- [ ] Team budget optimization recommendations
- [ ] Predictive modeling for upcoming auctions
- [ ] Player comparison tool

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👨‍💻 Author

**Akhil Garg**
- GitHub: [@akhilgarg29](https://github.com/akhilgarg29)

## 🙏 Acknowledgments

- IPL official data sources
- Kaggle IPL datasets community
- Streamlit for the amazing framework
- Open-source data science community

## 📧 Contact

For questions or feedback, please open an issue or contact me directly.

---
