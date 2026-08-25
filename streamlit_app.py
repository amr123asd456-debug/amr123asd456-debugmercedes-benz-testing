import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go


# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Mercedes-Benz Testing Time Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injection
st.markdown("""
<style>
    :root {
        --bg-main: #0B1220;
        --bg-card: #111C2E;
        --accent-blue: #2563EB;
        --accent-light: #38BDF8;
        --text-main: #F8FAFC;
        --text-muted: #94A3B8;
        --border-color: #263449;
        --success: #22C55E;
    }
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-main);
    }
    [data-testid="stSidebar"] {
        background-color: var(--bg-card);
        border-right: 1px solid var(--border-color);
    }
    .metric-container {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        padding: 16px;
        border-radius: 8px;
        text-align: center;
    }
    .result-card {
        background: linear-gradient(135deg, #111C2E 0%, #1E293B 100%);
        border: 2px solid #2563EB;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA & MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource
def load_assets():
    model_path = "models/final_gradient_boosting.joblib"
    preprocessor_path = "models/preprocessor.joblib"
    
    # Fallback paths for direct execution structure safety
    if not os.path.exists(model_path):
        model_path = "../models/final_gradient_boosting.joblib"
    if not os.path.exists(preprocessor_path):
        preprocessor_path = "../models/preprocessor.joblib"
        
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
    return model, preprocessor

@st.cache_data
def load_data():
    test_path = "test.csv"
    pred_path = "test_predictions.csv"
    
    if not os.path.exists(test_path):
        test_path = "../data/test.csv"
    if not os.path.exists(pred_path):
        pred_path = "../test_predictions.csv"
        
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()
    preds_df = pd.read_csv(pred_path) if os.path.exists(pred_path) else pd.DataFrame()
    return test_df, preds_df

model, preprocessor = load_assets()
test_df, preds_df = load_data()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("### 🚗 Benz Manufacturing AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation", 
    ["🏠 Home", "⚡ Live Prediction", "📊 Model Comparison", "📈 Analytics Dashboard"]
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Project Specs**\n\n"
    "• Model: Gradient Boosting\n"
    "• Final RMSE: 7.873\n"
    "• Feature Reduction: 95.6%"
)

# ==========================================
# 4. PAGE 1: HOME PAGE
# ==========================================
if page == "🏠 Home":
    st.title("Mercedes-Benz Testing Time Predictor")
    st.subheader("Machine Learning for Manufacturing Optimization & Feature Shrinkage")
    
    st.markdown("""
    Welcome to the enterprise analytics dashboard for predicting vehicle testing times based on anonymized 
    manufacturing configurations. This application showcases end-to-end machine learning engineering, rigorous 
    feature selection, and high-performance predictive modeling.
    """)
    
    st.markdown("### 🔑 Key Project KPIs")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Training Rows", "4,209")
    with col2:
        st.metric("Original Features", "376")
    with col3:
        st.metric("Encoded Features", "504")
    with col4:
        st.metric("Lasso Selected", "22")
    with col5:
        st.metric("Feature Reduction", "95.6%", delta="Optimized")
        
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🏭 Business Problem")
        st.info(
            "Mercedes-Benz testing facilities require varying durations to test vehicle configurations. "
            "Optimizing this testing timeline reduces bottlenecks on the assembly line, saves capital resources, "
            "and maintains strict quality assurance standards."
        )
    with c2:
        st.markdown("### 🚀 Winning Architecture")
        st.success(
            "**Gradient Boosting Regressor** emerged as the top-performing model, achieving a validation RMSE of **7.873** "
            "and an R² of **0.602**, outperforming Linear models, Random Forests, and PCA-engineered pipelines."
        )

# ==========================================
# 5. PAGE 2: PREDICTION PAGE
# ==========================================
elif page == "⚡ Live Prediction":
    st.title("Interactive Vehicle Testing Time Prediction")
    st.markdown("Select an evaluation test row to instantly query the trained Gradient Boosting inference engine.")
    
    if test_df.empty:
        st.warning("Test dataset not found in `data/test.csv`. Please check file paths.")
    else:
        selected_idx = st.selectbox("Select Test Row Index", options=test_df.index[:100])
        selected_row = test_df.loc[[selected_idx]]
        
        with st.expander("🔍 View Raw Vehicle Configuration Features", expanded=False):
            st.dataframe(selected_row, use_container_width=True)
            
        if st.button("Run Prediction", type="primary", use_container_width=True):
            if model is None or preprocessor is None:
                st.error("Model or preprocessor artifacts missing. Check `models/` directory.")
            else:
                with st.spinner("Processing configuration features through preprocessing pipeline..."):
                    try:
                        # Transform and predict
                        X_processed = preprocessor.transform(selected_row.drop(columns=['ID'], errors='ignore'))
                        prediction = model.predict(X_processed)[0]
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="result-card">
                            <h4 style="color: #38BDF8; margin-bottom: 5px;">PREDICTED TESTING TIME</h4>
                            <h1 style="font-size: 48px; color: #F8FAFC; margin: 0;">{prediction:.2f} <span style="font-size: 20px; color: #94A3B8;">seconds</span></h1>
                            <p style="color: #22C55E; margin-top: 10px; font-weight: 500;">✓ Inference successful using Gradient Boosting pipeline</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")

# ==========================================
# 6. PAGE 3: MODEL COMPARISON PAGE
# ==========================================
elif page == "📊 Model Comparison":
    st.title("Model Benchmark & Feature Shrinkage Analysis")
    st.markdown("Comparing regression models evaluated during the experiment lifecycle.")
    
    metrics_data = {
        "Model": ["Linear Regression", "Lasso", "Ridge", "Random Forest", "Gradient Boosting", "PCA + Ridge"],
        "MAE": [5.727, 5.388, 5.694, 5.919, 5.246, 5.570],
        "RMSE": [8.359, 8.028, 8.315, 9.160, 7.873, 8.255],
        "R²": [0.551, 0.586, 0.556, 0.461, 0.602, 0.562]
    }
    df_metrics = pd.DataFrame(metrics_data)
    
    st.dataframe(df_metrics.style.highlight_min(subset=["MAE", "RMSE"], color="#113023").highlight_max(subset=["R²"], color="#113023"), use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        fig_rmse = px.bar(df_metrics, x="Model", y="RMSE", title="Model RMSE Comparison (Lower is Better)", color="RMSE", color_continuous_scale="blues")
        fig_rmse.update_layout(plot_bgcolor="#111C2E", paper_bgcolor="#0B1220", font_color="#F8FAFC")
        st.plotly_chart(fig_rmse, use_container_width=True)
        
    with c2:
        fig_r2 = px.bar(df_metrics, x="Model", y="R²", title="Model R² Comparison (Higher is Better)", color="R²", color_continuous_scale="blues")
        fig_r2.update_layout(plot_bgcolor="#111C2E", paper_bgcolor="#0B1220", font_color="#F8FAFC")
        st.plotly_chart(fig_r2, use_container_width=True)
        
    st.markdown("---")
    st.markdown("### 📉 Feature Shrinkage Focus (Lasso)")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.info(
            "Lasso regularization successfully pruned the encoded feature space down from **504 features to 22 non-zero features**, "
            "delivering a **95.6% feature reduction** while maintaining strong predictive performance ($R^2 = 0.586$). "
            "This highlights extreme collinearity and redundancy in the raw manufacturing configuration space."
        )
    with col_b:
        st.metric("Feature Reduction", "95.6%", "504 → 22 features")

## ==========================================
# 7. PAGE 4: ANALYTICS DASHBOARD
# ==========================================
elif page == "📈 Analytics Dashboard":

    st.title("📈 Manufacturing Analytics Dashboard")
    st.markdown(
        "Explore prediction behavior, model performance, and feature reduction "
        "across the Mercedes-Benz testing-time pipeline."
    )

    # ======================================================
    # 1. BATCH PREDICTION KPIs
    # ======================================================

    if preds_df.empty:
        st.warning("Batch predictions file `test_predictions.csv` not found.")

    else:
        prediction_col = preds_df.columns[1]

        total_predictions = len(preds_df)
        mean_prediction = preds_df[prediction_col].mean()
        min_prediction = preds_df[prediction_col].min()
        max_prediction = preds_df[prediction_col].max()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Predictions",
                f"{total_predictions:,}"
            )

        with col2:
            st.metric(
                "Mean Testing Time",
                f"{mean_prediction:.2f} s"
            )

        with col3:
            st.metric(
                "Minimum Testing Time",
                f"{min_prediction:.2f} s"
            )

        with col4:
            st.metric(
                "Maximum Testing Time",
                f"{max_prediction:.2f} s"
            )

        st.markdown("---")

        # ==================================================
        # 2. PREDICTION DISTRIBUTION
        # ==================================================

        st.subheader("⏱️ Prediction Distribution")

        fig_dist = px.histogram(
            preds_df,
            x=prediction_col,
            nbins=35,
            title="Distribution of Predicted Testing Times",
            labels={
                prediction_col: "Predicted Testing Time (seconds)",
                "count": "Number of Vehicles"
            },
            color_discrete_sequence=["#2563EB"]
        )

        fig_dist.add_vline(
            x=mean_prediction,
            line_dash="dash",
            line_color="#38BDF8",
            annotation_text=f"Mean: {mean_prediction:.2f}s",
            annotation_position="top"
        )

        fig_dist.update_layout(
            plot_bgcolor="#111C2E",
            paper_bgcolor="#0B1220",
            font_color="#F8FAFC",
            title_x=0.02,
            height=450
        )

        st.plotly_chart(
            fig_dist,
            use_container_width=True
        )

        # ==================================================
        # 3. MODEL PERFORMANCE SECTION
        # ==================================================

        st.markdown("---")
        st.subheader("🏆 Model Performance Comparison")

        model_results = pd.DataFrame({
            "Model": [
                "Linear Regression",
                "Lasso",
                "Ridge",
                "Random Forest",
                "Gradient Boosting",
                "PCA + Ridge"
            ],
            "MAE": [
                5.727,
                5.388,
                5.694,
                5.919,
                5.246,
                5.570
            ],
            "RMSE": [
                8.359,
                8.028,
                8.315,
                9.160,
                7.873,
                8.255
            ],
            "R2": [
                0.551,
                0.586,
                0.556,
                0.461,
                0.602,
                0.562
            ]
        })

        col1, col2 = st.columns(2)

        # --------------------------------------------------
        # RMSE
        # --------------------------------------------------

        with col1:

            fig_rmse = px.bar(
                model_results.sort_values("RMSE"),
                x="RMSE",
                y="Model",
                orientation="h",
                title="RMSE Comparison",
                text="RMSE",
                color_discrete_sequence=["#2563EB"]
            )

            fig_rmse.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside"
            )

            fig_rmse.update_layout(
                plot_bgcolor="#111C2E",
                paper_bgcolor="#0B1220",
                font_color="#F8FAFC",
                height=450
            )

            st.plotly_chart(
                fig_rmse,
                use_container_width=True
            )

        # --------------------------------------------------
        # R²
        # --------------------------------------------------

        with col2:

            fig_r2 = px.bar(
                model_results.sort_values("R2"),
                x="R2",
                y="Model",
                orientation="h",
                title="R² Comparison",
                text="R2",
                color_discrete_sequence=["#38BDF8"]
            )

            fig_r2.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside"
            )

            fig_r2.update_layout(
                plot_bgcolor="#111C2E",
                paper_bgcolor="#0B1220",
                font_color="#F8FAFC",
                height=450
            )

            st.plotly_chart(
                fig_r2,
                use_container_width=True
            )

        # ==================================================
        # 4. FEATURE SHRINKAGE
        # ==================================================

        st.markdown("---")
        st.subheader("🔥 Feature Shrinkage")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Processed Features",
                "504"
            )

        with col2:
            st.metric(
                "Lasso Selected",
                "22"
            )

        with col3:
            reduction = (504 - 22) / 504 * 100

            st.metric(
                "Feature Reduction",
                f"{reduction:.1f}%"
            )

        col1, col2 = st.columns([1.2, 1])

        # --------------------------------------------------
        # Donut
        # --------------------------------------------------

        with col1:

            shrinkage_df = pd.DataFrame({
                "Status": [
                    "Lasso Selected",
                    "Removed by Shrinkage"
                ],
                "Features": [
                    22,
                    504 - 22
                ]
            })

            fig_donut = px.pie(
                shrinkage_df,
                names="Status",
                values="Features",
                hole=0.65,
                title="Lasso Feature Shrinkage",
                color_discrete_sequence=[
                    "#38BDF8",
                    "#263449"
                ]
            )

            fig_donut.update_layout(
                plot_bgcolor="#111C2E",
                paper_bgcolor="#0B1220",
                font_color="#F8FAFC",
                height=450,
                annotations=[
                    dict(
                        text="22 / 504",
                        x=0.5,
                        y=0.5,
                        font_size=24,
                        showarrow=False
                    )
                ]
            )

            st.plotly_chart(
                fig_donut,
                use_container_width=True
            )

        # --------------------------------------------------
        # Feature Pipeline
        # --------------------------------------------------

        with col2:

            st.markdown("### Feature Pipeline")

            st.info(
                """
                **376 Original Features**

                ↓ Cleaning

                **319 Clean Features**

                ↓ One-Hot Encoding

                **504 Processed Features**

                ↓ Lasso Shrinkage

                **22 Selected Features**
                """
            )

            st.success(
                f"Lasso reduced the processed feature space by "
                f"**{reduction:.1f}%**."
            )

        # ==================================================
        # 5. CATEGORICAL FEATURE COMPLEXITY
        # ==================================================

        st.markdown("---")
        st.subheader("🔤 Categorical Feature Complexity")

        categorical_summary = pd.DataFrame({
            "Feature": [
                "X0",
                "X1",
                "X2",
                "X3",
                "X4",
                "X5",
                "X6",
                "X8"
            ],
            "Unique Categories": [
                47,
                27,
                44,
                7,
                4,
                29,
                12,
                25
            ]
        })

        fig_cat = px.bar(
            categorical_summary.sort_values(
                "Unique Categories",
                ascending=True
            ),
            x="Unique Categories",
            y="Feature",
            orientation="h",
            text="Unique Categories",
            title="Number of Categories per Categorical Feature",
            color_discrete_sequence=["#2563EB"]
        )

        fig_cat.update_traces(
            textposition="outside"
        )

        fig_cat.update_layout(
            plot_bgcolor="#111C2E",
            paper_bgcolor="#0B1220",
            font_color="#F8FAFC",
            height=450
        )

        st.plotly_chart(
            fig_cat,
            use_container_width=True
        )

        # ==================================================
        # 6. TARGET DISTRIBUTION
        # ==================================================

        if os.path.exists("data/train.csv"):

            train_dashboard = pd.read_csv("data/train.csv")

            if "y" in train_dashboard.columns:

                st.markdown("---")
                st.subheader("🎯 Target Distribution")

                target_mean = train_dashboard["y"].mean()
                target_median = train_dashboard["y"].median()

                fig_target = px.histogram(
                    train_dashboard,
                    x="y",
                    nbins=35,
                    title="Distribution of Actual Testing Time",
                    labels={
                        "y": "Testing Time (seconds)",
                        "count": "Number of Vehicles"
                    },
                    color_discrete_sequence=["#38BDF8"]
                )

                fig_target.add_vline(
                    x=target_mean,
                    line_dash="dash",
                    line_color="#22C55E",
                    annotation_text=f"Mean: {target_mean:.2f}s"
                )

                fig_target.add_vline(
                    x=target_median,
                    line_dash="dot",
                    line_color="#F59E0B",
                    annotation_text=f"Median: {target_median:.2f}s"
                )

                fig_target.update_layout(
                    plot_bgcolor="#111C2E",
                    paper_bgcolor="#0B1220",
                    font_color="#F8FAFC",
                    height=450
                )

                st.plotly_chart(
                    fig_target,
                    use_container_width=True
                )

                st.caption(
                    "50 observations were identified as statistical outliers "
                    "using the IQR rule and were retained because they may "
                    "represent legitimate vehicle configurations."
                )

        # ==================================================
        # 7. MODEL INSIGHTS
        # ==================================================

        st.markdown("---")
        st.subheader("💡 Model Insights")

        insight1, insight2 = st.columns(2)

        with insight1:

            st.info(
                """
                **🏆 Best Prediction Model**

                Gradient Boosting achieved the best validation performance.

                **RMSE:** 7.873  
                **R²:** 0.602
                """
            )

        with insight2:

            st.success(
                """
                **🔥 Strongest Feature Shrinkage**

                Lasso reduced the processed feature space:

                **504 → 22 Features**

                **95.6% reduction**
                """
            )

        insight3, insight4 = st.columns(2)

        with insight3:

            st.warning(
                """
                **🧩 PCA Comparison**

                PCA + Ridge achieved:

                **RMSE:** 8.255  
                **R²:** 0.562

                This was weaker than Lasso in this experiment.
                """
            )

        with insight4:

            st.info(
                """
                **📌 Data Quality**

                No missing values, duplicate rows, or duplicate IDs
                were found in the original training/test datasets.
                """
            )

        # ==================================================
        # 8. DOWNLOAD PREDICTIONS
        # ==================================================

        st.markdown("---")
        st.subheader("📥 Prediction Export")

        csv_data = preds_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Test Predictions CSV",
            data=csv_data,
            file_name="test_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )               
