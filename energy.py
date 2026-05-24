import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import io
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import plotly.graph_objects as go

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="⚡ Renewable Energy BI Dashboard", layout="wide")




# ------------------------------
# STYLING (BACKGROUND IMAGE)
# ------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://png.pngtree.com/thumb_back/fw800/background/20241014/pngtree-inspiring-landscape-of-renewable-energy-sources-image_16388491.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;

    /* Overlay effect */
    background-color: rgba(255,255,255,0.6);
    background-blend-mode: overlay;
}

/* Title glow */
h1 {
    text-align: center;
    color: #0D47A1;
    text-shadow: 0 0 10px #64B5F6, 0 0 20px #42A5F5;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #E3F2FD, #E8F5E9);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# MAIN HEADER
# ------------------------------
st.title("⚡ Renewable Energy Performance Monitoring using BI")

# ------------------------------
# FILE UPLOAD
# ------------------------------
file = st.file_uploader("📂 Upload Dataset", type=["csv"])

# ------------------------------
# EMAIL FUNCTION
# ------------------------------
def send_email_alert(to, role, forecast_df, recommendation):

    sender_email = "hojhaskannan@gmail.com"
    sender_password = "kzap wtkz kjos zqlb"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to
    msg["Subject"] = f"⚡ {role} Renewable Energy Forecast Report"

    table = forecast_df[['ds','yhat']].to_string(index=False)

    body = f"""
Hello {role},

⚡ Renewable Energy Forecast Report

📊 Forecast Table:
{table}

💡 Recommendation:
{recommendation}

⚠️ Generated using AI Forecasting Model

Regards,
Energy BI System
"""
    msg.attach(MIMEText(body, "plain"))

    attach = MIMEApplication(forecast_df.to_csv(index=False))
    attach.add_header('Content-Disposition','attachment',filename="forecast.csv")
    msg.attach(attach)

    server = smtplib.SMTP("smtp.gmail.com",587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.send_message(msg)
    server.quit()

# ------------------------------
# MAIN LOGIC
# ------------------------------
if file:

    df = pd.read_csv(file)

    # ✅ FIXED COLUMN MAPPING
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.rename(columns={"Timestamp":"ds","Energy_Produced_MWh":"y"})

    # ------------------------------
    # SIDEBAR (UPDATED)
    # ------------------------------
    st.sidebar.header("⚙️ Controls")

    # 1️⃣ Role
    role = st.sidebar.radio("Select Role", ["User","Higher Authority"])

    # 2️⃣ Energy Source
    energy_sources = df["Energy_Source"].dropna().unique()
    selected_source = st.sidebar.selectbox("Select Energy Source", energy_sources)

    # 3️⃣ Location
    locations = df[df["Energy_Source"] == selected_source]["Location"].dropna().unique()
    selected_location = st.sidebar.selectbox("Select Location", locations)

    # ------------------------------
    # FILTER DATA
    # ------------------------------
    filtered_df = df[
        (df["Energy_Source"] == selected_source) &
        (df["Location"] == selected_location)
    ]

    data = filtered_df[['ds','y']].dropna()

    # Show selected filters
    st.markdown(f"""
    ### 📍 Selected Filters:
    - ⚡ Energy Source: **{selected_source}**
    - 📌 Location: **{selected_location}**
    - 👤 Role: **{role}**
    """)

    # 📊 Historical Graph
    st.subheader("📊 Historical Graph")
    st.line_chart(data.set_index("ds"))
    st.download_button("Download Historical", data.to_csv().encode(),"history.csv")

    # 🔄 Rolling Average
    st.subheader("🔄 Rolling Average")
    data['rolling'] = data['y'].rolling(7).mean()
    st.line_chart(data.set_index("ds")[['y','rolling']])
    st.download_button("Download Rolling", data.to_csv().encode(),"rolling.csv")

    # 🧮 Stats
    st.subheader("🧮 Descriptive Statistics")
    stats = data['y'].describe()
    st.write(stats)
    st.download_button("Download Stats", stats.to_csv().encode(),"stats.csv")

    # 🧭 Distribution
    st.subheader("🧭 Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data['y'], kde=True, ax=ax)
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf)
    st.download_button("Download Distribution", buf.getvalue(),"dist.png")

    # 🔮 Forecast
    st.subheader("🔮 Forecast")
    days = st.slider("Forecast Days", 30, 180, 60)

    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    st.line_chart(forecast.set_index("ds")["yhat"])

    # 📊 Evaluation
    st.subheader("📊 Model Evaluation")
    mae = mean_absolute_error(data['y'], forecast['yhat'][:len(data)])
    rmse = np.sqrt(mean_squared_error(data['y'], forecast['yhat'][:len(data)]))
    st.write("MAE:", mae)
    st.write("RMSE:", rmse)

    fig2, ax2 = plt.subplots()
    ax2.plot(data['ds'], data['y'], label="Actual")
    ax2.plot(forecast['ds'], forecast['yhat'], label="Forecast")
    ax2.legend()
    st.pyplot(fig2)

    buf2 = io.BytesIO()
    fig2.savefig(buf2)
    st.download_button("Download Forecast Graph", buf2.getvalue(),"forecast.png")

    # CSV
    st.download_button("Download Forecast CSV", forecast.to_csv().encode(),"forecast.csv")

    # ⚠️ Early Warning
    st.subheader("⚠️ Early Warning System")
    threshold = st.slider("Surge %", 5, 30, 10)/100
    forecast['change'] = forecast['yhat'].pct_change()
    alerts = forecast[forecast['change'] > threshold]

    if not alerts.empty:
        st.warning("🚨 Surge Detected")
        st.dataframe(alerts)
        st.download_button("Download Alerts", alerts.to_csv().encode(),"alerts.csv")

    # Summary
    latest = data['y'].iloc[-1]
    future_avg = forecast['yhat'].tail(days).mean()
    change = ((future_avg - latest)/latest)*100

    st.subheader("📊 Forecast Summary")
    st.write(f"Current: {latest:.2f}")
    st.write(f"Future Avg: {future_avg:.2f}")
    st.write(f"Change: {change:.2f}%")

    # AI Recommendation
    st.subheader("💡 AI Recommendation")

    if role == "User":
        if change > 5:
            recommendation = "Energy demand rising — conserve usage"
        elif change < -5:
            recommendation = "Energy demand dropping — good usage time"
        else:
            recommendation = "Stable usage recommended"
    else:
        if change > 5:
            recommendation = "Increase generation capacity"
        elif change < -5:
            recommendation = "Reduce production load"
        else:
            recommendation = "Maintain supply balance"

    st.success(recommendation)

    # Email
    if st.button("Send Email Alert"):
        send_email_alert("kannan5045@gmail.com", role, forecast.tail(days), recommendation)
        st.success("Email Sent")

    # Chatbot
    st.subheader("💬 Smart Trend Chatbot")
    q = st.text_input("Ask trend")
    if q:
        if change > 5:
            st.write("Increasing trend")
        elif change < -5:
            st.write("Decreasing trend")
        else:
            st.write("Stable")
    # 🤖 Original Chatbot (10 questions)
    st.subheader("🤖 Chatbot")
    q2 = st.text_input("Ask data question")
    if q2:
        if "mean" in q2: st.write(data['y'].mean())
        elif "max" in q2: st.write(data['y'].max())
        elif "min" in q2: st.write(data['y'].min())
        elif "std" in q2: st.write(data['y'].std())
        elif "last 7" in q2: st.line_chart(data.tail(7))
        elif "top" in q2: st.write(data.sort_values("y",ascending=False).head())
        elif "count" in q2: st.write(len(data))
        elif "median" in q2: st.write(data['y'].median())
        elif "variance" in q2: st.write(data['y'].var())
        else: st.write("Invalid question")
    # Top Movers
    st.subheader("🏆 Top Movers")
    filtered_df['change'] = filtered_df['y'].pct_change()*100
    st.dataframe(filtered_df.sort_values("change",ascending=False).head())

    # Volatility
    st.subheader("📊 Volatility Index")
    vol = data['y'].std()
    st.write(vol)

    # ------------------------------
    # 📏 ADVANCED STABILITY GAUGE
    # ------------------------------
    st.subheader("📏 Energy Stability Gauge")

    volatility = data["y"].std()
    mean_val = data["y"].mean()

    volatility_ratio = (volatility / mean_val) * 100
    stability = max(0, 100 - volatility_ratio)

    filled = int(stability // 10)

    # 🎨 Emoji + Color Logic
    if stability >= 80:
        emoji = "🟢😊 Very Stable"
        color = "#2ECC71"
    elif stability >= 50:
        emoji = "🟡😐 Moderately Stable"
        color = "#F1C40F"
    else:
        emoji = "🔴⚠️ Unstable"
        color = "#E74C3C"

    # 🟩 Emoji Bar
    gauge_bar = "🟩" * filled + "⬜" * (10 - filled)

    # 🌈 HTML Progress Bar
    st.markdown(f"""
    <div style='background-color:#eee;border-radius:10px;height:25px;width:80%;margin:auto'>
      <div style='width:{stability:.1f}%;background-color:{color};height:25px;text-align:center;color:white;font-weight:bold'>
        {stability:.1f}% Stable
      </div>
    </div>
    <p style='text-align:center;font-size:18px;margin-top:8px;'>{gauge_bar}</p>
    <p style='text-align:center;font-size:16px;'>{emoji}</p>
    """, unsafe_allow_html=True)

    # ------------------------------
    # 🚨 3D ANOMALY DETECTOR (PREMIUM)
    # ------------------------------
    st.subheader("🚨 3D Energy Anomaly Detector")

    # Z-score calculation
    data["z_score"] = (data["y"] - data["y"].mean()) / data["y"].std()
    anomalies = data[np.abs(data["z_score"]) > 3]

    # Sort for plotting
    data_sorted = data.sort_values("ds").reset_index(drop=True)
    data_sorted["index"] = np.arange(len(data_sorted))

    # Normalize colors
    colors = np.interp(data_sorted['y'],
                       (data_sorted['y'].min(), data_sorted['y'].max()),
                       (0, 1))

    fig_3d = go.Figure()

    # 🔵 Main 3D Line
    fig_3d.add_trace(go.Scatter3d(
        x=data_sorted["index"],
        y=np.zeros(len(data_sorted)),
        z=data_sorted["y"],
        mode='lines+markers',
        line=dict(width=6, color=colors, colorscale='Viridis'),
        marker=dict(size=4, color=colors, colorscale='Viridis'),
        name='Energy Output'
    ))

    # 🔴 Anomalies
    if not anomalies.empty:
        anomalies_sorted = anomalies.sort_values("ds").reset_index(drop=True)
        anomalies_sorted["index"] = np.arange(len(anomalies_sorted))

        fig_3d.add_trace(go.Scatter3d(
            x=anomalies_sorted["index"],
            y=np.zeros(len(anomalies_sorted)),
            z=anomalies_sorted["y"] + 0.05 * data_sorted["y"].max(),
            mode='markers',
            marker=dict(
                size=6 + anomalies_sorted['z_score']*2,
                color='red',
                symbol='diamond',
                opacity=0.9
            ),
            name='Anomalies'
    ))

    # 🟣 Surface effect (premium look)
    Z_surface = np.zeros((2, len(data_sorted))) + np.expand_dims(data_sorted['y'].values, axis=0)

    fig_3d.add_trace(go.Surface(
        z=Z_surface,
        x=np.tile(data_sorted['index'], (2,1)),
        y=np.array([[0]*len(data_sorted), [0.5]*len(data_sorted)]),
        colorscale='Viridis',
        opacity=0.3,
        showscale=False
    ))

    # Layout
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='Time Index',
            yaxis_title='Plant',
            zaxis_title='Energy Output',
        ),
        title="🚨 3D Anomaly Detection",
        height=650
    )

    st.plotly_chart(fig_3d, use_container_width=True)

    # Info
    if anomalies.empty:
        st.success("✅ No anomalies detected")
    else:
        st.warning(f"⚠️ {len(anomalies)} anomalies detected")
        st.dataframe(anomalies[["ds", "y", "z_score"]])

    # Report
    report = f"Trend:{change:.2f}%, Stability:{stability:.2f}%"
    st.download_button("Download Report", report, "report.txt")

    # ------------------------------
    # 📰 AI SENTIMENT - RENEWABLE ENERGY (DOMAIN SPECIFIC)
    # ------------------------------
    st.subheader("📰 Renewable Energy Sentiment Analysis (BI Insights)")

    news_input = st.text_area(
        "Paste Renewable Energy News (one per line):",
        placeholder="Example:\nSolar plant efficiency improved\nWind turbine failure reported\nGrid stability remains steady"
    )

    if news_input:

        lines = news_input.strip().split("\n")

        # ⚡ Domain-specific keywords
        positive_words = [
            "increase","growth","surge","high output","efficient",
            "improved efficiency","record production","high generation",
            "low downtime","stable grid","optimization","upgrade",
            "strong performance","capacity expansion"
        ]

        negative_words = [
            "drop","decline","low output","failure","breakdown",
            "high downtime","outage","grid failure","low efficiency",
            "maintenance issue","shutdown","overload","instability"
        ]

        neutral_words = [
            "stable","steady","normal","consistent","balanced"
        ]

        score = 0
        results = []

        for line in lines:
            text = line.lower()
            sentiment = "Neutral"

            if any(word in text for word in positive_words):
                score += 1
                sentiment = "🟢 Positive"
            elif any(word in text for word in negative_words):
                score -= 1
                sentiment = "🔴 Negative"
            elif any(word in text for word in neutral_words):
                sentiment = "🟡 Neutral"

            results.append((line, sentiment))

        # 📊 Show Result Table
        sentiment_df = pd.DataFrame(results, columns=["News Headline","Sentiment"])
        st.dataframe(sentiment_df)

        # 🎯 Overall Sentiment
        st.subheader("📊 Overall Market Sentiment")

        if score > 0:
            st.success(f"🟢 Positive Energy Market (Score: {score})")
            insight = "⚡ Energy production and efficiency are improving. System performance is strong."

        elif score < 0:
            st.error(f"🔴 Negative Energy Market (Score: {score})")
            insight = "⚠️ Issues detected like outages, downtime, or efficiency drops. Maintenance required."

        else:
            st.info(f"🟡 Neutral Energy Market (Score: {score})")
            insight = "⚖️ Energy system is stable with no major changes."

        # 💡 BI Insight (IMPORTANT FOR YOUR PROJECT)
        st.markdown(f"""
        ### 💡 BI Insight:
        👉 {insight}
        """)

        # 🎯 Role-Based Insight (VERY IMPORTANT 🔥)
        st.subheader("👥 Role-Based Recommendation")

        if role == "User":
            if score > 0:
                st.success("🟢 User Advice: Energy supply is strong — normal usage is safe.")
            elif score < 0:
                st.warning("🔴 User Advice: Possible issues — conserve energy usage.")
            else:
                st.info("🟡 User Advice: Stable supply — moderate usage.")

        else:  # Higher Authority
            if score > 0:
                st.success("🟢 Authority Action: Maintain performance & optimize further.")
            elif score < 0:
                st.error("🔴 Authority Action: Immediate maintenance & grid monitoring required.")
            else:
                st.info("🟡 Authority Action: Maintain operational stability.")

    # ------------------------------
    # 📅 SMART CALENDAR FORECAST (FIXED)
    # ------------------------------
    st.subheader("📅 Smart Calendar Forecast (10-Year Analysis)")

    future_long = model.make_future_dataframe(periods=365*10)
    forecast_long = model.predict(future_long)

    # Remove time part for matching
    forecast_long['ds'] = pd.to_datetime(forecast_long['ds']).dt.date

    selected_date = st.date_input("Select a date")

    if selected_date:

        # ✅ Find nearest available date instead of exact match
        forecast_long['diff'] = abs(pd.to_datetime(forecast_long['ds']) - pd.to_datetime(selected_date))
        nearest_row = forecast_long.loc[forecast_long['diff'].idxmin()]

        predicted_value = nearest_row['yhat']

        st.success(f"📊 Predicted Energy on {selected_date}: {predicted_value:.2f}")

        # ✅ Prediction Text (IMPORTANT)
        if predicted_value > latest * 1.1:
            st.warning("📈 Energy production/demand will significantly increase.")
        elif predicted_value < latest * 0.9:
            st.success("📉 Energy production/demand will decrease — efficient usage opportunity.")
        else:
            st.info("⚖️ Energy levels will remain stable.")

        # Show confidence interval
        st.write(f"🔹 Range: {nearest_row['yhat_lower']:.2f} to {nearest_row['yhat_upper']:.2f}")
else:
    st.info("Upload dataset to start")
