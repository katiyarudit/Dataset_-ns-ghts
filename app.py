import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt



st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://unsplash.com/photos/purple-and-blue-light-digital-wallpaper-8bghKxNU1j0");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.sidebar.title("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.title("ℹ️ About This App")
st.sidebar.info(
    "This is an **Automated EDA & Insights App** that helps users quickly analyze datasets. "
    "It supports **advanced visualizations, missing values analysis, and dark/light mode customization.** "
    "Built with **Streamlit, Pandas, Matplotlib, and Plotly**."
)

st.sidebar.title("👨‍💻 About Me")
st.sidebar.info(
    "**Udit Katiyar**\n\n"
    "🚀 **Computer Science Engineer | Tech Enthusiast**\n\n"
    "💡 Passionate about **Web Development, AI/ML, and Open-Source Contributions**\n\n"
    "📝 Sharing thoughts on **cutting-edge technologies, problem-solving, and innovation**\n\n"
    "📚 Exploring **Cloud Computing, DevOps, and Blockchain**\n\n"
    "🔥 Always eager to learn and build amazing projects!"
)

st.sidebar.title("📢 Contact Me")
st.sidebar.info(
    "📧 **Email:** [uditkatiyar2005@gmail.com](mailto:uditkatiyar2005@gmail.com)\n"
    "🔗 **GitHub:** [github.com/katiyarudit](https://github.com/katiyarudit/dataset_insights)\n"
    "💼 **LinkedIn:** [linkedin.com/in/udit1105](https://www.linkedin.com/in/udit1105/)\n"
)

st.sidebar.markdown("---")
st.sidebar.title("🎨 Theme Settings")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown(
        """
        <style>
            body { background-color: #1E1E1E; color: white; }
            .stApp { background-color: #1E1E1E; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("🚀 Smart Data Explorer: AI-Powered EDA & Visualization")
st.markdown("Upload your dataset and uncover powerful insights with interactive charts, AI-driven analysis, and advanced visualizations!")

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    
    if df.empty or df.shape[1] == 0:
        st.error("⚠️ Uploaded dataset is empty or contains no valid columns. Please upload a valid dataset.")
    else:
        tab1, tab2 = st.tabs(["📊 Graphical Analysis", "📌 Summary Analysis"])

        with tab1:
            st.subheader("📌 Dataset Overview")
            st.dataframe(df.head())

            st.sidebar.subheader("🛠 Data Preprocessing")

            missing_value_option = st.sidebar.radio("Handle Missing Values", ["Do Nothing", "Drop Rows", "Fill with Mean"])
            if missing_value_option == "Drop Rows":
                df = df.dropna()
            elif missing_value_option == "Fill with Mean":
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

            numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if not numerical_cols:
                st.error("⚠️ No numerical columns found in dataset.")
            else:
                selected_cols = st.sidebar.multiselect("📊 Select Columns for Analysis", numerical_cols, default=numerical_cols)

                st.subheader("📊 Descriptive Statistics")
                if selected_cols:
                    st.write(df[selected_cols].describe())
                else:
                    st.warning("⚠️ Please select at least one column to analyze.")

                st.subheader("🔗 Correlation Heatmap")
                if len(selected_cols) > 1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.heatmap(df[selected_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ Need at least two columns to show correlation.")

                st.subheader("📈 Distribution of Features")
                for col in selected_cols:
                    fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("📊 Scatter Plot")
                if len(selected_cols) > 1:
                    scatter_x = st.selectbox("Select X-axis", selected_cols)
                    scatter_y = st.selectbox("Select Y-axis", selected_cols)
                    fig = px.scatter(df, x=scatter_x, y=scatter_y, title=f"{scatter_x} vs {scatter_y}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Need at least two columns to create a scatter plot.")
                
                st.subheader("📊 Pie Chart Visualization")

                # Select categorical columns
                categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

                if categorical_cols:
                        selected_pie_col = st.selectbox("Select a categorical column for Pie Chart", categorical_cols)

                        # Generate Pie Chart
                        fig = px.pie(df, names=selected_pie_col, title=f"Distribution of {selected_pie_col}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                        st.warning("⚠️ No categorical columns found for pie chart visualization.")


                st.subheader("⚠️ Outlier Detection")
                for col in selected_cols:
                    fig = px.box(df, y=col, title=f"Outliers in {col}")
                    st.plotly_chart(fig, use_container_width=True)

                if "target" in df.columns:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor()
                    X = df[selected_cols]
                    y = df["target"]
                    model.fit(X, y)
                    feature_importance = pd.Series(model.feature_importances_, index=selected_cols).sort_values(ascending=False)

                    st.subheader("🔥 Feature Importance")
                    st.bar_chart(feature_importance)
                    

                


        with tab2:
            st.subheader("📌 Automated Dataset Summary")
            num_rows, num_cols = df.shape
            st.write(f"✅ The dataset contains **{num_rows} rows** and **{num_cols} columns**.")

            st.subheader("📌 Feature-Wise Summary")
            for col in df.columns:
                unique_values = df[col].nunique()
                missing_values = df[col].isnull().sum()
                dtype = df[col].dtype
                st.write(f"🔹 **{col}**: {dtype} | Unique Values: {unique_values} | Missing: {missing_values}")

            st.subheader("🚨 Missing Data Summary")
            missing_summary = df.isnull().sum()
            missing_summary = missing_summary[missing_summary > 0]
            if not missing_summary.empty:
                st.write(f"⚠️ **{missing_summary.sum()} missing values** detected.")
                st.write(missing_summary.to_frame(name="Missing Count"))
            else:
                st.write("✅ No missing values detected.")

            st.subheader("📌 Categorical Feature Summary")
            categorical_cols = df.select_dtypes(include=["object"]).columns
            for col in categorical_cols:
                st.write(f"🔹 **{col}** has **{df[col].nunique()} unique categories**.")
                st.write(df[col].value_counts().head(5))

            st.success("✅ Full Analysis Completed!")
            

else:
    st.warning("📌 Please upload a dataset to proceed.")

st.markdown("---")
st.markdown("🚀 Built by **Udit Katiyar** | Powered by **Streamlit & Python**")
