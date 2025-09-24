import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

st.set_page_config(page_title="Indian Crop Yield Predictor", page_icon="ðŸŒ¾")

st.title("ðŸŒ¾ Indian Crop Yield Predictor")
st.sidebar.header("Enter Farm Details")

states = ["Maharashtra", "Punjab", "Uttar Pradesh", "West Bengal", "Gujarat"]
crops = ["Rice", "Wheat", "Cotton", "Sugarcane", "Pulses"]

state = st.sidebar.selectbox("State", states, index=0, help="Select the state")
crop = st.sidebar.selectbox("Crop", crops, index=0, help="Select the crop")
rainfall = st.sidebar.slider(
    "Rainfall (mm)",
    min_value=200,
    max_value=3000,
    value=1000,
    help="Average rainfall in your area"
)

rng = np.random.RandomState(42)
base_state = {s: i for i, s in enumerate(states)}
base_crop = {c: i for i, c in enumerate(crops)}

def make_synth_data(n=750):
    S = rng.choice(states, size=n)
    C = rng.choice(crops, size=n)
    R = rng.uniform(200, 3000, size=n)
    y = (
        1.5 * np.vectorize(base_state.get)(S)
        + 2.0 * np.vectorize(base_crop.get)(C)
        + 0.02 * R
        + rng.normal(0, 3, size=n)
    )
    return pd.DataFrame({"state": S, "crop": C, "rainfall": R, "yield": y})

df = make_synth_data()

X = pd.get_dummies(df[["state", "crop"]])
X["rainfall"] = df["rainfall"].values
y = df["yield"].values

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

x_user = pd.DataFrame({"rainfall": [float(rainfall)]}, index=[0])
for s in states:
    x_user[f"state_{s}"] = 1.0 if s == state else 0.0
for c in crops:
    x_user[f"crop_{c}"] = 1.0 if c == crop else 0.0
x_user = x_user.reindex(columns=X.columns, fill_value=0.0)

pred = model.predict(x_user)[0]
st.subheader("Predicted Yield (arbitrary units)")
st.metric(label="Estimated yield", value=f"{pred:.2f}")

rain_range = np.linspace(200, 3000, 25)
viz = []
for r in rain_range:
    row = {"rainfall": r}
    row.update({f"state_{s}": (1.0 if s == state else 0.0) for s in states})
    row.update({f"crop_{c}": (1.0 if c == crop else 0.0) for c in crops})
    viz.append(row)
X_viz = pd.DataFrame(viz).reindex(columns=X.columns, fill_value=0.0)
y_viz = model.predict(X_viz)
fig = px.line(x=rain_range, y=y_viz, labels={"x": "Rainfall (mm)", "y": "Predicted Yield"})
st.plotly_chart(fig, use_container_width=True)
