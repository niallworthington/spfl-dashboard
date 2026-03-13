import streamlit as st
import pandas as pd
import numpy as np
import math

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="SPFL 2026 Analytics", layout="wide")
st.title("🏴 Scottish Premiership 2026/27 - NiallW Football")

# ---------------------------------------------------
# MODEL CONSTANTS
# ---------------------------------------------------
LEAGUE_AVG_GOALS_HOME = 1.578
LEAGUE_AVG_GOALS_AWAY = 1.203
ITERATIONS = 5000


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():

    results = pd.read_csv("spfl_current_results.csv")
    fixtures = pd.read_csv("spfl_fixtures.csv")
    ratings = pd.read_csv("spfl_ratings.csv")

    for df in [results, fixtures, ratings]:
        for col in ["Home", "Away", "Club"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

    return results, fixtures, ratings


results, fixtures, ratings = load_data()


# ---------------------------------------------------
# MANUAL RESULT STORAGE
# ---------------------------------------------------
if "manual_results" not in st.session_state:
    st.session_state.manual_results = []

# ---------------------------------------------------
# SIDEBAR MANUAL INPUTS
# ---------------------------------------------------
st.sidebar.header("🛠 What-If Scenario")

with st.sidebar.expander("Add Manual Result"):

    fixture_list = fixtures.apply(
        lambda x: f"{x.Home} vs {x.Away}", axis=1
    )

    selected = st.selectbox("Fixture", fixture_list)

    col1, col2 = st.columns(2)

    h = col1.number_input("Home Goals", 0, 10, 0)
    a = col2.number_input("Away Goals", 0, 10, 0)

    if st.button("Add Result"):

        row = fixtures.iloc[fixture_list.tolist().index(selected)]

        st.session_state.manual_results.append({
            "Home": row["Home"],
            "Away": row["Away"],
            "HomeGoals": h,
            "AwayGoals": a
        })

        st.success("Added!")

# ---------------------------------------------------
# TABLE CALCULATION
# ---------------------------------------------------
def calculate_table(results):

    results = results.copy()

    results["HomePoints"] = np.where(
        results["HomeGoals"] > results["AwayGoals"], 3,
        np.where(results["HomeGoals"] == results["AwayGoals"], 1, 0)
    )

    results["AwayPoints"] = np.where(
        results["AwayGoals"] > results["HomeGoals"], 3,
        np.where(results["AwayGoals"] == results["HomeGoals"], 1, 0)
    )

    home = results.groupby("Home").agg(
        Played=("Home", "count"),
        Points=("HomePoints", "sum"),
        GF=("HomeGoals", "sum"),
        GA=("AwayGoals", "sum")
    )

    away = results.groupby("Away").agg(
        Played=("Away", "count"),
        Points=("AwayPoints", "sum"),
        GF=("AwayGoals", "sum"),
        GA=("HomeGoals", "sum")
    )

    table = home.add(away, fill_value=0)

    table["GD"] = table["GF"] - table["GA"]

    table = table.reset_index()
    table = table.rename(columns={"Home": "Team", "index": "Team"})

    return table.sort_values(["Points", "GD"], ascending=False)
manual_df = pd.DataFrame(st.session_state.manual_results)

if not manual_df.empty:
    combined_results = pd.concat([results, manual_df], ignore_index=True)
else:
    combined_results = results.copy()

current_table = calculate_table(combined_results)
st.subheader("🏁 Current Table")
st.dataframe(current_table, use_container_width=True)


# ---------------------------------------------------
# SIMULATION FUNCTION
# ---------------------------------------------------
def sim_score(home_att, home_def, away_att, away_def):

    lam_h = LEAGUE_AVG_GOALS_HOME * home_att / away_def
    lam_a = LEAGUE_AVG_GOALS_AWAY * away_att / home_def

    g_h = np.random.poisson(lam_h)
    g_a = np.random.poisson(lam_a)

    return g_h, g_a


@st.cache_data
def simulate_season(ratings, fixtures, table):

    team_names = ratings["Club"].tolist()
    team_map = {t: i for i, t in enumerate(team_names)}

    start_points = dict(zip(table["Team"], table["Points"]))

    final_points = np.zeros((ITERATIONS, len(team_names)))

    for sim in range(ITERATIONS):

        pts = np.array([start_points.get(t, 0) for t in team_names])

        for _, f in fixtures.iterrows():

            h = f["Home"]
            a = f["Away"]

            h_idx = team_map[h]
            a_idx = team_map[a]

            h_row = ratings.iloc[h_idx]
            a_row = ratings.iloc[a_idx]

            g_h, g_a = sim_score(
                h_row["AttackRating"],
                h_row["DefenceRating"],
                a_row["AttackRating"],
                a_row["DefenceRating"]
            )

            if g_h > g_a:
                pts[h_idx] += 3
            elif g_h == g_a:
                pts[h_idx] += 1
                pts[a_idx] += 1
            else:
                pts[a_idx] += 3

        final_points[sim] = pts

    order = np.argsort(-final_points, axis=1)

    probs = pd.DataFrame(index=team_names)

    probs["Title %"] = (order[:,0:1] == np.arange(len(team_names))).mean(axis=0) * 100
    probs["Avg Points"] = final_points.mean(axis=0)

    proj = pd.DataFrame({
        "Club": team_names,
        "Current": [start_points.get(t,0) for t in team_names],
        "Forecasted": final_points.mean(axis=0)
    })

    proj = proj.sort_values("Forecasted", ascending=False)

    return proj, probs.round(1)


# ---------------------------------------------------
# RUN SIMULATION
# ---------------------------------------------------
if st.button("🎲 Run Season Simulation"):

    proj, probs = simulate_season(ratings, fixtures, current_table)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Title Probability")
        st.dataframe(probs)

    with col2:
        st.subheader("📈 Projected Table")
        st.dataframe(proj)


# ---------------------------------------------------
# MATCH PREDICTOR
# ---------------------------------------------------
st.markdown("---")
st.subheader("⚽ Match Predictor")

teams = sorted(ratings["Club"].tolist())

c1, c2 = st.columns(2)

home_team = c1.selectbox("Home", teams)
away_team = c2.selectbox("Away", teams, index=1)


if st.button("Predict"):

    h = ratings[ratings["Club"] == home_team].iloc[0]
    a = ratings[ratings["Club"] == away_team].iloc[0]

    lam_h = LEAGUE_AVG_GOALS_HOME * h["AttackRating"] / a["DefenceRating"]
    lam_a = LEAGUE_AVG_GOALS_AWAY * a["AttackRating"] / h["DefenceRating"]

    st.write(f"xG: {home_team} {lam_h:.2f} - {lam_a:.2f} {away_team}")

    probs = {}

    for i in range(6):
        for j in range(6):

            p = (
                (lam_h**i * np.exp(-lam_h) / math.factorial(i)) *
                (lam_a**j * np.exp(-lam_a) / math.factorial(j))
            )

            probs[f"{i}-{j}"] = p

    top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]

    for score, prob in top5:
        st.write(f"{score} → {prob:.1%}")	