import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import warnings

# Optional imports for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Luxury Handbag Social Media Analysis",
    page_icon="👜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #E91E63;
        font-weight: bold;
    }
    h2, h3 {
        color: #333;
    }
    .highlight-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #E91E63;
        border: 1px solid #e0e0e0;
        margin: 20px 0;
        color: #333333 !important;
    }
    .highlight-box h2 {
        color: #E91E63 !important;
        margin-top: 0;
        margin-bottom: 15px;
    }
    .highlight-box p {
        color: #333333 !important;
        line-height: 1.6;
        font-size: 16px;
    }
    .highlight-box li {
        color: #333333 !important;
        line-height: 1.6;
        margin-bottom: 8px;
    }
    .highlight-box ul {
        padding-left: 20px;
        margin-bottom: 0;
    }
    section[data-testid="stSidebar"] {
        width: 250px !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("cdatasetlv1.csv")
    return df


# Prepare data for modeling
@st.cache_data
def prepare_data(df, target_column):
    df_model = df.copy()

    # Encode categorical variables
    le_dict = {}
    for col in df_model.select_dtypes(include=["object"]).columns:
        if col != target_column:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            le_dict[col] = le

    # Separate features and target
    if target_column in df_model.columns:
        X = df_model.drop(columns=[target_column])
        y = df_model[target_column]
    else:
        X = df_model
        y = None

    return X, y, le_dict


# Train classification models
@st.cache_resource
def train_classification_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    trained_models = {}
    metrics = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = (
            model.predict_proba(X_test_scaled)
            if hasattr(model, "predict_proba")
            else None
        )

        trained_models[name] = model
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1 Score": f1_score(y_test, y_pred, average="weighted"),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

    return trained_models, metrics, scaler, X_train, X_test, y_train, y_test


# Main app
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Classification Prediction",
            "Data Insights",
            "Model Interpretability",
        ],
    )

    # Load data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading Dataset.csv: {e}")
        st.info("Please ensure 'Dataset.csv' is in the same directory as this script.")
        return

    # Page routing
    if page == "Overview":
        show_overview(df)
    elif page == "Classification Prediction":
        show_classification(df)
    elif page == "Data Insights":
        show_insights(df)
    elif page == "Model Interpretability":
        show_interpretability(df)


def show_overview(df):
    st.title("👜 Luxury Handbag Social Media Analysis Dashboard")

    st.markdown("---")

    # Problem Statement
    st.markdown(
        """
    <div class="highlight-box">
    <h2>The Challenge</h2>
    <p>The luxury handbag market is driven by social media influence, brand reputation, and consumer engagement.
    Understanding what drives engagement rates, premium pricing, and high ratings requires analyzing multiple factors
    including platform performance, demographics, product categories, and social media metrics. This dashboard helps
    brands optimize their social media strategy and predict key success indicators.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("##")

    # Solution
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            """
        <div class="highlight-box">
        <h2>Our Solution</h2>
        <p>This interactive dashboard provides comprehensive analysis through:</p>
        <ul>
            <li><strong>Classification Models:</strong> Predict premium engagement, high ratings, and luxury segments</li>
            <li><strong>Data Visualization:</strong> Interactive charts revealing engagement patterns and trends</li>
            <li><strong>Model Transparency:</strong> SHAP and LIME analysis to understand prediction drivers</li>
            <li><strong>Brand Comparison:</strong> Compare performance across brands, platforms, and regions</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="highlight-box">
        <h2>Key Features</h2>
        <ul>
            <li><strong>Classification Prediction:</strong> Predict categorical outcomes like premium engagement</li>
            <li><strong>Data Insights:</strong> Explore engagement patterns, brand performance, and demographics</li>
            <li><strong>Model Interpretability:</strong> Feature importance and prediction explanations</li>
            <li><strong>Performance Metrics:</strong> Accuracy, precision, recall, and F1 scores</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("##")

    # Dataset Overview
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 5px; border-radius: 10px; text-align: center;">
                <h3 style="color: #E91E63; margin: 0; font-size: 18px;">Total Records</h3>
                <p style="color: #333; font-size: 32px; font-weight: bold; margin: 5px 0 0 0;">{len(df):,}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 5px; border-radius: 10px; text-align: center;">
                <h3 style="color: #E91E63; margin: 0; font-size: 18px;">Features</h3>
                <p style="color: #333; font-size: 32px; font-weight: bold; margin: 5px 0 0 0;">{len(df.columns)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        unique_brands = df["brand_name"].nunique() if "brand_name" in df.columns else 0
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 5px; border-radius: 10px; text-align: center;">
                <h3 style="color: #E91E63; margin: 0; font-size: 18px;">Brands</h3>
                <p style="color: #333; font-size: 32px; font-weight: bold; margin: 5px 0 0 0;">{unique_brands}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        platforms = df["platform"].nunique() if "platform" in df.columns else 0
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 5px; border-radius: 10px; text-align: center;">
                <h3 style="color: #E91E63; margin: 0; font-size: 18px;">Platforms</h3>
                <p style="color: #333; font-size: 32px; font-weight: bold; margin: 5px 0 0 0;">{platforms}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("##")

    # Key Insights
    st.header("Key Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_engagement = df["engagement_rate"].mean() if "engagement_rate" in df.columns else 0
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                <h3 style="color: #E91E63; margin: 0; font-size: 18px;">Avg Engagement Rate</h3>
                <p style="color: #333; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">{avg_engagement:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        avg_rating = df["review_rating"].mean() if "review_rating" in df.columns else 0
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                <h3 style="color: #E91E63; margin: 0; font-size: 18px;">Avg Review Rating</h3>
                <p style="color: #333; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">{avg_rating:.2f}/5.0</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        premium_pct = (
            df["premium_engagement"].mean() * 100
            if "premium_engagement" in df.columns
            else 0
        )
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                <h3 style="color: #E91E63; margin: 0; font-size: 18px;">Premium Engagement %</h3>
                <p style="color: #333; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">{premium_pct:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("##")

    # Data Preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True, height=300)

    st.markdown("##")

    # Quick Statistics
    st.subheader("Quick Statistics")
    st.dataframe(df.describe(), use_container_width=True)


def show_classification(df):
    st.title("Classification Prediction")

    st.markdown(
        """
    Predict binary outcomes such as premium engagement, high ratings, or luxury segment classification.
    Select a target variable and adjust input parameters to see predictions.
    """
    )

    st.markdown("---")

    # Select target variable - only binary classification targets
    binary_cols = [
        col
        for col in df.columns
        if df[col].dtype in [np.int64, np.bool_]
        and df[col].nunique() == 2
    ]

    col1, col2 = st.columns([1, 3])

    with col1:
        target_col = st.selectbox(
            "Select Target Variable to Predict",
            binary_cols,
            index=binary_cols.index("premium_engagement")
            if "premium_engagement" in binary_cols
            else 0,
        )

    with col2:
        model_choice = st.selectbox(
            "Select Classification Model",
            ["Logistic Regression", "Random Forest", "Gradient Boosting"],
        )

    st.markdown("##")

    # Prepare data
    X, y, le_dict = prepare_data(df, target_col)

    if y is not None:
        # Train models
        trained_models, metrics, scaler, X_train, X_test, y_train, y_test = (
            train_classification_models(X, y)
        )

        # Display model performance
        st.subheader(f"Model Performance: {model_choice}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #E91E63; margin: 0; font-size: 18px;">Accuracy</h3>
                    <p style="color: #333; font-size: 28px; font-weight: bold; margin: 10px 0 0 0;">{metrics[model_choice]['Accuracy']:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #E91E63; margin: 0; font-size: 18px;">Precision</h3>
                    <p style="color: #333; font-size: 28px; font-weight: bold; margin: 10px 0 0 0;">{metrics[model_choice]['Precision']:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #E91E63; margin: 0; font-size: 18px;">Recall</h3>
                    <p style="color: #333; font-size: 28px; font-weight: bold; margin: 10px 0 0 0;">{metrics[model_choice]['Recall']:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #E91E63; margin: 0; font-size: 18px;">F1 Score</h3>
                    <p style="color: #333; font-size: 28px; font-weight: bold; margin: 10px 0 0 0;">{metrics[model_choice]['F1 Score']:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Input section
        st.subheader("Adjust Input Parameters")

        input_data = {}

        # Create columns for input fields
        feature_cols = st.columns(3)

        for idx, col in enumerate(X.columns):
            with feature_cols[idx % 3]:
                if col in df.select_dtypes(include=[np.number]).columns:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    mean_val = float(df[col].mean())
                    step_val = (max_val - min_val) / 100 if max_val != min_val else 1.0
                    input_data[col] = st.slider(
                        col,
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=step_val,
                    )
                else:
                    unique_vals = df[col].unique()
                    selected = st.selectbox(col, unique_vals, key=f"input_{col}")
                    if col in le_dict:
                        input_data[col] = le_dict[col].transform([selected])[0]
                    else:
                        input_data[col] = selected

        st.markdown("##")

        # Make prediction
        if st.button("Generate Prediction", type="primary"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)

            prediction = trained_models[model_choice].predict(input_scaled)[0]
            prediction_proba = trained_models[model_choice].predict_proba(
                input_scaled
            )[0]

            st.markdown("##")
            st.success("Prediction Complete!")

            # Display prediction
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                pred_label = "Yes" if prediction == 1 else "No"
                confidence = prediction_proba[prediction] * 100

                st.markdown(
                    f"""
                <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #E91E63 0%, #F48FB1 100%);
                border-radius: 15px; color: white;">
                    <h2>Predicted {target_col.replace('_', ' ').title()}</h2>
                    <h1 style="font-size: 48px; margin: 20px 0;">{pred_label}</h1>
                    <h3>Confidence: {confidence:.1f}%</h3>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("##")

            # Show probability distribution
            st.subheader("Prediction Probabilities")

            prob_df = pd.DataFrame(
                {
                    "Class": ["No (0)", "Yes (1)"],
                    "Probability": [prediction_proba[0], prediction_proba[1]],
                }
            )

            fig = px.bar(
                prob_df,
                x="Class",
                y="Probability",
                title="Class Probability Distribution",
                color="Probability",
                color_continuous_scale="RdPu",
            )
            fig.update_layout(showlegend=False, height=400)

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("##")

            # Show class distribution in dataset
            st.subheader("Target Variable Distribution in Dataset")

            class_dist = df[target_col].value_counts()
            class_dist_pct = (class_dist / len(df) * 100).round(1)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                        <h3 style="color: #E91E63; margin: 0; font-size: 18px;">Class 0 (No)</h3>
                        <p style="color: #333; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">{class_dist_pct[0]}%</p>
                        <p style="color: #666; font-size: 14px; margin: 5px 0 0 0;">({class_dist[0]:,} records)</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                        <h3 style="color: #E91E63; margin: 0; font-size: 18px;">Class 1 (Yes)</h3>
                        <p style="color: #333; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">{class_dist_pct[1]}%</p>
                        <p style="color: #666; font-size: 14px; margin: 5px 0 0 0;">({class_dist[1]:,} records)</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def show_insights(df):
    st.title("Data Insights")

    st.markdown(
        "Explore comprehensive visualizations of the luxury handbag social media dataset."
    )

    st.markdown("---")

    # Brand Analysis
    st.header("Brand Performance Analysis")

    if "brand_name" in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            # Top brands by engagement
            brand_engagement = (
                df.groupby("brand_name")["engagement_rate"].mean().sort_values(ascending=False).head(10)
            )
            fig = px.bar(
                x=brand_engagement.values,
                y=brand_engagement.index,
                orientation="h",
                title="Top 10 Brands by Average Engagement Rate",
                labels={"x": "Engagement Rate", "y": "Brand"},
                color=brand_engagement.values,
                color_continuous_scale="Purples",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Brand distribution
            brand_counts = df["brand_name"].value_counts().head(10)
            fig = px.pie(
                values=brand_counts.values,
                names=brand_counts.index,
                title="Top 10 Brands by Post Count",
                color_discrete_sequence=px.colors.sequential.RdPu,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Platform Analysis
    st.header("Platform Performance")

    if "platform" in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            platform_engagement = (
                df.groupby("platform")["engagement_rate"].mean().sort_values(ascending=False)
            )
            fig = px.bar(
                x=platform_engagement.index,
                y=platform_engagement.values,
                title="Average Engagement Rate by Platform",
                labels={"x": "Platform", "y": "Engagement Rate"},
                color=platform_engagement.values,
                color_continuous_scale="Pinkyl",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            platform_dist = df["platform"].value_counts()
            fig = px.pie(
                values=platform_dist.values,
                names=platform_dist.index,
                title="Post Distribution by Platform",
                color_discrete_sequence=px.colors.sequential.Reds,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Engagement Metrics
    st.header("Engagement Metrics Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Engagement rate distribution
        fig = px.histogram(
            df,
            x="engagement_rate",
            title="Distribution of Engagement Rate",
            color_discrete_sequence=["#E91E63"],
            nbins=50,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Box plot of engagement by product type
        if "product_type" in df.columns:
            fig = px.box(
                df,
                x="product_type",
                y="engagement_rate",
                title="Engagement Rate by Product Type",
                color="product_type",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Demographics Analysis
    st.header("Demographics Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if "gender" in df.columns:
            gender_engagement = df.groupby("gender")["engagement_rate"].mean()
            fig = px.bar(
                x=gender_engagement.index,
                y=gender_engagement.values,
                title="Average Engagement Rate by Gender",
                labels={"x": "Gender", "y": "Engagement Rate"},
                color=gender_engagement.values,
                color_continuous_scale="Purples",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "age_range" in df.columns:
            age_engagement = df.groupby("age_range")["engagement_rate"].mean()
            fig = px.bar(
                x=age_engagement.index,
                y=age_engagement.values,
                title="Average Engagement Rate by Age Range",
                labels={"x": "Age Range", "y": "Engagement Rate"},
                color=age_engagement.values,
                color_continuous_scale="RdPu",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Price Analysis
    st.header("Price and Rating Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if "price_usd" in df.columns and "engagement_rate" in df.columns:
            fig = px.scatter(
                df,
                x="price_usd",
                y="engagement_rate",
                title="Engagement Rate vs Price",
                color="high_price_tier" if "high_price_tier" in df.columns else None,
                labels={"price_usd": "Price (USD)", "engagement_rate": "Engagement Rate"},
                color_continuous_scale="Reds",
                opacity=0.6,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "review_rating" in df.columns:
            rating_dist = df["review_rating"].value_counts().sort_index()
            fig = px.bar(
                x=rating_dist.index,
                y=rating_dist.values,
                title="Distribution of Review Ratings",
                labels={"x": "Rating", "y": "Count"},
                color=rating_dist.values,
                color_continuous_scale="Greens",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Regional Analysis
    st.header("Regional Analysis")

    if "region" in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            region_engagement = (
                df.groupby("region")["engagement_rate"].mean().sort_values(ascending=False)
            )
            fig = px.bar(
                x=region_engagement.values,
                y=region_engagement.index,
                orientation="h",
                title="Average Engagement Rate by Region",
                labels={"x": "Engagement Rate", "y": "Region"},
                color=region_engagement.values,
                color_continuous_scale="Purples",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            region_dist = df["region"].value_counts()
            fig = px.pie(
                values=region_dist.values,
                names=region_dist.index,
                title="Post Distribution by Region",
                color_discrete_sequence=px.colors.sequential.Magenta,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Correlation Heatmap
    st.header("Feature Correlation Heatmap")

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    corr_matrix = df[numerical_cols].corr()

    fig = px.imshow(
        corr_matrix,
        title="Correlation Matrix of Numerical Features",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        labels=dict(color="Correlation"),
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(height=700)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Time-based Analysis
    st.header("Time-based Analysis")

    if "post_day_of_week" in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            day_engagement = (
                df.groupby("post_day_of_week")["engagement_rate"]
                .mean()
                .reindex(
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    fill_value=0,
                )
            )
            fig = px.line(
                x=day_engagement.index,
                y=day_engagement.values,
                title="Average Engagement Rate by Day of Week",
                labels={"x": "Day of Week", "y": "Engagement Rate"},
                markers=True,
            )
            fig.update_traces(line_color="#E91E63", line_width=3)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "is_weekend" in df.columns:
                weekend_engagement = df.groupby("is_weekend")["engagement_rate"].mean()
                fig = px.bar(
                    x=["Weekday", "Weekend"],
                    y=weekend_engagement.values,
                    title="Average Engagement: Weekday vs Weekend",
                    labels={"x": "Period", "y": "Engagement Rate"},
                    color=weekend_engagement.values,
                    color_continuous_scale="Pinkyl",
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Social Media Metrics
    st.header("Social Media Metrics")

    col1, col2 = st.columns(2)

    with col1:
        if "likes" in df.columns and "comments" in df.columns and "shares" in df.columns:
            metrics_df = pd.DataFrame(
                {
                    "Metric": ["Likes", "Comments", "Shares"],
                    "Average": [
                        df["likes"].mean(),
                        df["comments"].mean(),
                        df["shares"].mean(),
                    ],
                }
            )
            fig = px.bar(
                metrics_df,
                x="Metric",
                y="Average",
                title="Average Social Media Metrics",
                color="Average",
                color_continuous_scale="Purples",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "comment_to_like_ratio" in df.columns and "share_to_like_ratio" in df.columns:
            fig = px.scatter(
                df,
                x="comment_to_like_ratio",
                y="share_to_like_ratio",
                title="Comment-to-Like vs Share-to-Like Ratio",
                labels={
                    "comment_to_like_ratio": "Comment-to-Like Ratio",
                    "share_to_like_ratio": "Share-to-Like Ratio",
                },
                color="engagement_rate",
                color_continuous_scale="RdPu",
                opacity=0.6,
            )
            st.plotly_chart(fig, use_container_width=True)


def show_interpretability(df):
    st.title("Model Interpretability")

    st.markdown(
        "Understand how classification models make predictions through feature importance analysis and performance comparison."
    )

    st.markdown("---")

    # Select target variable
    binary_cols = [
        col
        for col in df.columns
        if df[col].dtype in [np.int64, np.bool_]
        and df[col].nunique() == 2
    ]

    target_col = st.selectbox(
        "Select Target Variable for Analysis",
        binary_cols,
        index=binary_cols.index("premium_engagement")
        if "premium_engagement" in binary_cols
        else 0,
    )

    # Prepare data
    X, y, le_dict = prepare_data(df, target_col)

    if y is not None:
        # Train models
        trained_models, metrics, scaler, X_train, X_test, y_train, y_test = (
            train_classification_models(X, y)
        )

        st.markdown("##")

        # Model Performance Comparison
        st.header("Model Performance Comparison")

        metrics_df = pd.DataFrame(
            {
                "Model": list(metrics.keys()),
                "Accuracy": [metrics[m]["Accuracy"] for m in metrics.keys()],
                "Precision": [metrics[m]["Precision"] for m in metrics.keys()],
                "Recall": [metrics[m]["Recall"] for m in metrics.keys()],
                "F1 Score": [metrics[m]["F1 Score"] for m in metrics.keys()],
            }
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                metrics_df,
                x="Model",
                y="Accuracy",
                title="Accuracy Comparison",
                color="Accuracy",
                color_continuous_scale="Purples",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                metrics_df,
                x="Model",
                y="F1 Score",
                title="F1 Score Comparison",
                color="F1 Score",
                color_continuous_scale="RdPu",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Metrics table
        st.subheader("Detailed Metrics")
        st.dataframe(
            metrics_df.style.highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1 Score"],
                color="lightpink",
            ),
            use_container_width=True,
        )

        st.markdown("---")

        # Confusion Matrix
        st.header("Confusion Matrix Analysis")

        model_for_cm = st.selectbox(
            "Select Model for Confusion Matrix", list(trained_models.keys())
        )

        cm = metrics[model_for_cm]["confusion_matrix"]

        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Class 0", "Class 1"],
            y=["Class 0", "Class 1"],
            title=f"Confusion Matrix - {model_for_cm}",
            color_continuous_scale="RdPu",
            text_auto=True,
        )
        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)

        # Classification metrics breakdown
        col1, col2, col3, col4 = st.columns(4)

        tn, fp, fn, tp = cm.ravel()

        with col1:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #E91E63; margin: 0; font-size: 16px;">True Positives</h3>
                    <p style="color: #333; font-size: 28px; font-weight: bold; margin: 10px 0 0 0;">{tp}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #E91E63; margin: 0; font-size: 16px;">True Negatives</h3>
                    <p style="color: #333; font-size: 28px; font-weight: bold; margin: 10px 0 0 0;">{tn}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #E91E63; margin: 0; font-size: 16px;">False Positives</h3>
                    <p style="color: #333; font-size: 28px; font-weight: bold; margin: 10px 0 0 0;">{fp}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #E91E63; margin: 0; font-size: 16px;">False Negatives</h3>
                    <p style="color: #333; font-size: 28px; font-weight: bold; margin: 10px 0 0 0;">{fn}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ROC Curve
        st.header("ROC Curve Analysis")

        model_for_roc = st.selectbox(
            "Select Model for ROC Curve", list(trained_models.keys()), key="roc_model"
        )

        if metrics[model_for_roc]["y_pred_proba"] is not None:
            y_test_roc = metrics[model_for_roc]["y_test"]
            y_pred_proba_roc = metrics[model_for_roc]["y_pred_proba"][:, 1]

            fpr, tpr, _ = roc_curve(y_test_roc, y_pred_proba_roc)
            roc_auc = auc(fpr, tpr)

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"ROC Curve (AUC = {roc_auc:.3f})",
                    line=dict(color="#E91E63", width=3),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random Classifier",
                    line=dict(color="gray", dash="dash"),
                )
            )

            fig.update_layout(
                title=f"ROC Curve - {model_for_roc}",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)

            st.info(
                f"**AUC Score: {roc_auc:.4f}** - A higher AUC (closer to 1.0) indicates better model performance."
            )

        st.markdown("---")

        # Feature Importance
        st.header("Feature Importance Analysis")

        model_for_fi = st.selectbox(
            "Select Model for Feature Importance",
            list(trained_models.keys()),
            key="fi_model",
        )

        if hasattr(trained_models[model_for_fi], "feature_importances_"):
            importances = trained_models[model_for_fi].feature_importances_
            feature_importance_df = pd.DataFrame(
                {"Feature": X.columns, "Importance": importances}
            ).sort_values("Importance", ascending=True)

            fig = px.bar(
                feature_importance_df.tail(15),
                x="Importance",
                y="Feature",
                orientation="h",
                title=f"Top 15 Feature Importances - {model_for_fi}",
                color="Importance",
                color_continuous_scale="Purples",
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            # For Logistic Regression, use coefficients
            if hasattr(trained_models[model_for_fi], "coef_"):
                coef = np.abs(trained_models[model_for_fi].coef_[0])
                feature_importance_df = pd.DataFrame(
                    {"Feature": X.columns, "Importance": coef}
                ).sort_values("Importance", ascending=True)

                fig = px.bar(
                    feature_importance_df.tail(15),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title=f"Top 15 Feature Coefficients (Absolute) - {model_for_fi}",
                    color="Importance",
                    color_continuous_scale="RdPu",
                )

                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # SHAP Analysis
        st.header("SHAP Feature Importance Analysis")

        st.markdown(
            """
        SHAP values show how much each feature contributes to pushing the model's prediction from the base value
        to the actual prediction. Higher absolute values indicate more important features.
        """
        )

        model_for_shap = st.selectbox(
            "Select Model for SHAP Analysis",
            list(trained_models.keys()),
            key="shap_model",
        )

        with st.spinner("Generating SHAP analysis..."):
            try:
                # Use a sample of data for SHAP
                sample_size = min(100, len(X_train))
                X_train_sample = X_train.iloc[:sample_size]
                X_train_scaled_sample = scaler.transform(X_train_sample)

                if model_for_shap == "Logistic Regression":
                    explainer = shap.LinearExplainer(
                        trained_models[model_for_shap], X_train_scaled_sample
                    )
                    shap_values = explainer.shap_values(X_train_scaled_sample)
                else:
                    explainer = shap.Explainer(
                        trained_models[model_for_shap], X_train_scaled_sample
                    )
                    shap_values_obj = explainer(X_train_scaled_sample)
                    if hasattr(shap_values_obj, "values"):
                        shap_values = shap_values_obj.values
                    else:
                        shap_values = shap_values_obj

                # Handle multi-output SHAP values
                if len(shap_values.shape) == 3:
                    shap_values = shap_values[:, :, 1]

                # Feature importance bar plot
                shap_importance = np.abs(shap_values).mean(axis=0)
                feature_importance_df = pd.DataFrame(
                    {"Feature": X.columns, "SHAP Importance": shap_importance}
                ).sort_values("SHAP Importance", ascending=True)

                fig = px.bar(
                    feature_importance_df.tail(15),
                    x="SHAP Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 15 Features by SHAP Importance",
                    color="SHAP Importance",
                    color_continuous_scale="Viridis",
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not generate SHAP analysis: {e}")

        st.markdown("---")

        # LIME Analysis
        st.header("LIME Individual Prediction Explanation")

        st.markdown(
            """
        LIME explains individual predictions by approximating the model locally with an interpretable model.
        This shows which features contributed most to a specific prediction.
        """
        )

        model_for_lime = st.selectbox(
            "Select Model for LIME Analysis",
            list(trained_models.keys()),
            key="lime_model",
        )

        instance_idx = st.slider(
            "Select Instance to Explain", 0, len(X_test) - 1, 0, key="lime_instance"
        )

        with st.spinner("Generating LIME explanation..."):
            try:
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                explainer = LimeTabularExplainer(
                    X_train_scaled,
                    feature_names=X.columns.tolist(),
                    class_names=["Class 0", "Class 1"],
                    mode="classification",
                    random_state=42,
                )

                exp = explainer.explain_instance(
                    X_test_scaled[instance_idx],
                    trained_models[model_for_lime].predict_proba,
                    num_features=15,
                )

                # Extract LIME values
                lime_values = exp.as_list()
                lime_df = pd.DataFrame(lime_values, columns=["Feature", "Importance"])
                lime_df = lime_df.sort_values("Importance", ascending=True)

                fig = px.bar(
                    lime_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title=f"LIME Feature Importance for Instance {instance_idx}",
                    color="Importance",
                    color_continuous_scale="RdYlGn",
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show actual prediction
                actual_value = y_test.iloc[instance_idx]
                predicted_value = trained_models[model_for_lime].predict(
                    X_test_scaled[instance_idx].reshape(1, -1)
                )[0]
                pred_proba = trained_models[model_for_lime].predict_proba(
                    X_test_scaled[instance_idx].reshape(1, -1)
                )[0]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(
                        f"""
                        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                            <h3 style="color: #E91E63; margin: 0; font-size: 18px;">Actual Class</h3>
                            <p style="color: #333; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">{actual_value}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(
                        f"""
                        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                            <h3 style="color: #E91E63; margin: 0; font-size: 18px;">Predicted Class</h3>
                            <p style="color: #333; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">{predicted_value}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col3:
                    st.markdown(
                        f"""
                        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                            <h3 style="color: #E91E63; margin: 0; font-size: 18px;">Confidence</h3>
                            <p style="color: #333; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">{pred_proba[predicted_value]:.2%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Show feature values for this instance
                st.markdown("##")
                st.subheader("Feature Values for Selected Instance")

                instance_features = X_test.iloc[instance_idx].to_dict()
                feature_df = pd.DataFrame(
                    [
                        {"Feature": k, "Value": v}
                        for k, v in instance_features.items()
                    ]
                )

                st.dataframe(feature_df, use_container_width=True, height=300)

            except Exception as e:
                st.warning(f"Could not generate LIME analysis: {e}")


if __name__ == "__main__":
    main()