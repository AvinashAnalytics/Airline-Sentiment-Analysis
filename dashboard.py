import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from streamlit.components.v1 import html

# Configure visual settings
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelcolor'] = '#2c3e50'
plt.rcParams['figure.titlesize'] = 16

@st.cache_data
def load_data():
    return pd.read_csv("Data/Cleaned_Tweets.csv")

df = load_data()

# ========== Custom CSS ==========
st.markdown("""
<style>
    .header-style { 
        font-size: 36px !important; 
        color: #1f567d !important;
        border-bottom: 3px solid #1f567d;
        padding-bottom: 12px;
        margin-bottom: 1.5rem;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    .hover-effect {
        transition: all 0.3s ease;
    }
    .hover-effect:hover {
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ========== Sidebar ==========
with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/158772098?v=4", width=100)
    st.markdown("""
    <div style="margin-top: -15px;">
        <h2 style='color: #1f567d; margin-bottom: 5px;'>Avinash Rai</h2>
        <p style='color: #6c757d; margin-bottom: 20px;'>Data Analyst</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🛠️ Technical Showcase")
    st.markdown("""
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn
    - **NLP**: VADER Sentiment
    - **Web App**: Streamlit
    """)
    
    st.markdown("---")
    st.markdown("**Let's Connect:**  \n"
                "📧 [Email](mailto:masteravinashrai@gmail.com)  \n"
                "💼 [LinkedIn](https://linkedin.com/in/avinashanalytics)  \n"
                "👨💻 [GitHub](https://github.com/AvinashAnalytics)")

# ========== Main Dashboard ==========
st.markdown('<h1 class="header-style">Airline Sentiment Analysis Project</h1>', unsafe_allow_html=True)

# Project Overview Section
with st.expander("📌 **Project Overview**", expanded=True):
    st.markdown("""
    **Objective:** Analyze customer sentiment from airline tweets to understand service quality perceptions
    
    **Key Components:**
    1. Processed & cleaned 15,000+ tweets
    2. Performed sentiment analysis using VADER
    3. Created interactive visualizations
    4. Built dynamic reporting dashboard
    
    **Technical Highlights:**
    - Data cleaning & preprocessing
    - Sentiment classification
    - Interactive data exploration
    - Automated report generation
    """)

# Filters
with st.container():
    col1, col2 = st.columns([2, 3])
    with col1:
        airline = st.selectbox("**Select Airline**", df['airline'].unique(), index=0)
    with col2:
        sentiment_filter = st.radio(
            "**Filter Sentiment**",
            ["All", "Positive", "Negative", "Neutral"],
            horizontal=True,
            label_visibility="visible"
        )

# Data Filtering
filtered_df = df[df['airline'] == airline]
if sentiment_filter != "All":
    filtered_df = filtered_df[filtered_df['airline_sentiment'] == sentiment_filter.lower()]

# Metrics with Hover Effects
with st.container():
    st.markdown("### 📊 Key Insights")
    cols = st.columns(4)
    metrics = [
        ("Total Tweets", len(filtered_df), "# tweets analyzed"),
        ("Avg. Sentiment", 
         filtered_df['vader_score'].mean() if not filtered_df.empty else 0,
         "Range: -1 (Negative) to 1 (Positive)"),
        ("Positive %", 
         (filtered_df['airline_sentiment'].value_counts(normalize=True).get('positive', 0) * 100),
         "% of positive feedback"),
        ("Negative %", 
         (filtered_df['airline_sentiment'].value_counts(normalize=True).get('negative', 0) * 100),
         "% of negative feedback")
    ]
    
    for col, (label, value, tooltip) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card hover-effect" title="{tooltip}">
                <div class="metric-label" style="color: #5a5a5a; font-size: 14px;">{label}</div>
                <div class="metric-value" style="color: #1f567d; font-size: 24px; font-weight: 600;">
                    {f'{value:.1f}%' if '%' in label else f'{value:.2f}' if isinstance(value, float) else f'{value:,}'}
                </div>
            </div>
            """, unsafe_allow_html=True)

# Visualization Section
with st.container():
    st.markdown("### 📈 Sentiment Analysis")
    viz_col1, viz_col2 = st.columns([2, 1])
    
    with viz_col1:
        st.markdown("**Sentiment Distribution**")
        if not filtered_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            palette = {'negative':'#ff6b6b', 'neutral':'#a5d8dd', 'positive':'#7bc043'}
            
            sns.countplot(
                data=filtered_df,
                y='airline_sentiment',
                order=['positive', 'neutral', 'negative'],
                palette=palette,
                ax=ax
            )
            
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(axis='x', alpha=0.3)
            sns.despine(left=True)
            
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_width():.0f}",
                    (p.get_width(), p.get_y() + p.get_height()/2),
                    ha='left', va='center', 
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=12
                )
            
            st.pyplot(fig)
        else:
            st.info("No data available for current filters")
    
    with viz_col2:
        st.markdown("**Sentiment Proportions**")
        if not filtered_df.empty:
            fig, ax = plt.subplots(figsize=(6, 6))
            counts = filtered_df['airline_sentiment'].value_counts()
            
            ax.pie(
                counts,
                labels=counts.index.str.title(),
                autopct='%1.1f%%',
                colors=[palette[k] for k in counts.index],
                startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                textprops={'fontsize': 12}
            )
            ax.set_title("")
            st.pyplot(fig)
        else:
            st.info("No data for pie chart")

# Word Cloud Section
with st.container():
    st.markdown("### 🔍 Customer Feedback Analysis")
    if not filtered_df.empty:
        with st.spinner('Generating word cloud...'):
            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='RdYlGn',
                max_words=50
            ).generate(' '.join(filtered_df['text']))
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    else:
        st.info("No data available for word cloud")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 20px">
    <p style="font-size: 0.9rem;">
        Portfolio Project • Built with Python & Streamlit • 
        <a href="https://github.com/AvinashAnalytics/Airline-Sentiment-Analysis" style="color: #1f567d; text-decoration: none;">View Source Code</a>
    </p>
</div>
""", unsafe_allow_html=True)

# Add subtle animation
html("""
<script>
const elements = window.parent.document.querySelectorAll('.metric-card, .hover-effect');
elements.forEach(element => {
    element.addEventListener('mouseover', () => {
        element.style.transition = 'all 0.3s ease';
        element.style.boxShadow = '0 8px 15px rgba(0,0,0,0.1)';
    });
    element.addEventListener('mouseout', () => {
        element.style.boxShadow = '0 4px 6px rgba(0,0,0,0.05)';
    });
});
</script>
""")