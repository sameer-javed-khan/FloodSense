# 🌊 FloodSense — Complete Master Guide
**The single document with everything you need to ship this project.**

This guide assumes **zero prior experience**. Follow it top-to-bottom. By the end you'll have a working ML model, a deployed dashboard, a polished GitHub repo, and a submission ready for judges.

---

## 📋 Table of Contents

1. [What you're building](#1-what-youre-building)
2. [Required files — full list](#2-required-files--full-list)
3. [Step-by-step execution](#3-step-by-step-execution)
4. [GitHub upload walkthrough](#4-github-upload-walkthrough)
5. [Hackathon submission checklist](#5-hackathon-submission-checklist)
6. [Live demo playbook](#6-live-demo-playbook)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. What you're building

A flood early warning system with three parts:

1. **An ML model** (XGBoost classifier) that predicts flood risk from rainfall, soil, terrain, and atmospheric data — outputs Low/Medium/High/Critical with confidence scores.
2. **A bilingual web dashboard** (Streamlit + custom HTML/CSS) where a non-technical district official enters 5 inputs and gets a colour-coded risk assessment in English and Urdu.
3. **A GitHub repository** with everything documented — judges can clone it and run it themselves.

**Tech stack:** Python · pandas · XGBoost · SHAP · Streamlit · Custom CSS · Jupyter

**Performance:** 72% accuracy · 59% flood recall · 0.74 ROC AUC

---

## 2. Required files — full list

Your project folder will contain exactly these files. Every one is described below.

```
floodsense/
├── data/
│   ├── floodsense_training_data.csv    ← from organizers
│   ├── district_elevation_reference.csv ← from organizers
│   ├── ndma_flood_impact_2022.csv      ← from organizers
│   └── data_dictionary.txt              ← from organizers
├── artifacts/                           ← created by training
│   ├── model.pkl
│   ├── feature_cols.pkl
│   ├── metrics.json
│   └── shap_summary.png
├── floodsense_eda.ipynb                 ← I built this for you
├── floodsense_training.ipynb            ← I built this for you
├── train.py                             ← I built this for you (alt to notebook)
├── app.py                               ← I built this for you
├── requirements.txt                     ← I built this for you
├── README.md                            ← I built this for you
└── .gitignore                           ← I built this for you
```

### File-by-file explanation

| File | Purpose | Source |
|------|---------|--------|
| `data/*.csv` | Training datasets | Cloned from `https://github.com/menahil-fatima-727/neural_nova_datadrop.git` |
| `floodsense_eda.ipynb` | 14-section exploratory data analysis with plots | Download from this chat |
| `floodsense_training.ipynb` | Full training pipeline with explanations | Download from this chat |
| `train.py` | Same training pipeline as a Python script | Download from this chat |
| `app.py` | Streamlit dashboard with HTML/CSS UI | Download from this chat |
| `requirements.txt` | Pip dependencies | Download from this chat |
| `README.md` | Repo description for GitHub | Download from this chat |
| `.gitignore` | Tells Git which files to skip | Download from this chat |
| `artifacts/*` | Outputs of training (model, metrics, plots) | Auto-created when you run training |

---

## 3. Step-by-step execution

### Stage A — One-time setup (15 minutes)

#### A.1 — Open a terminal

- **Windows:** Press `Win` key, type `cmd`, press Enter
- **Mac:** Press `Cmd+Space`, type `Terminal`, press Enter
- **Linux:** Press `Ctrl+Alt+T`

#### A.2 — Verify Python (3.10 or higher)

In the terminal:
```bash
python --version
```

If you see `Python 3.10.x` or higher, skip to A.3. Otherwise install from https://python.org/downloads/ — **on Windows, check the "Add Python to PATH" box** during installation. Close and reopen your terminal after.

#### A.3 — Verify Git

```bash
git --version
```

If error: install from https://git-scm.com/download (Windows) or `brew install git` (Mac) or `sudo apt install git` (Linux).

#### A.4 — Create your workspace folder

**Windows:**
```bash
cd %USERPROFILE%\Documents
mkdir floodsense
cd floodsense
```

**Mac/Linux:**
```bash
cd ~/Documents
mkdir floodsense
cd floodsense
```

#### A.5 — Get the dataset

```bash
git clone https://github.com/menahil-fatima-727/neural_nova_datadrop.git temp_data
mkdir data
```

Now copy the CSVs from `temp_data/` to `data/`:

**Windows (in cmd):**
```bash
copy temp_data\floodsense_training_data.csv data\
copy temp_data\district_elevation_reference.csv data\
copy temp_data\ndma_flood_impact_2022.csv data\
copy temp_data\data_dictionary.txt data\
rmdir /s /q temp_data
```

**Mac/Linux:**
```bash
cp temp_data/*.csv data/
cp temp_data/*.txt data/
rm -rf temp_data
```

#### A.6 — Save my files into the folder

From this chat, **download these 8 files** and put them in your `floodsense` folder:

1. `floodsense_eda.ipynb`
2. `floodsense_training.ipynb`
3. `app.py`
4. `train.py`
5. `requirements.txt`
6. `README.md`
7. `.gitignore`
8. `MASTER_GUIDE.md` (this file)

Your folder now looks like:
```
floodsense/
├── data/
│   └── (4 CSV/txt files)
├── floodsense_eda.ipynb
├── floodsense_training.ipynb
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
└── MASTER_GUIDE.md
```

#### A.7 — Install dependencies

```bash
pip install -r requirements.txt
```

This downloads pandas, scikit-learn, XGBoost, Streamlit, SHAP, etc. Takes 2–3 minutes.

If `pip` errors say "not recognized": try `pip3` or `python -m pip`.

✅ **Setup complete.**

---

### Stage B — Run the EDA notebook (10 minutes)

The EDA notebook proves we understand the data. **Run it first** — it produces plots and findings you'll cite in the pitch.

#### B.1 — Launch Jupyter

In your terminal (still in the `floodsense` folder):
```bash
jupyter notebook
```

A browser tab opens automatically at `http://localhost:8888/tree`. **Don't close the terminal.**

#### B.2 — Open and run the EDA notebook

1. Click `floodsense_eda.ipynb` in the file list
2. Menu bar → **Cell → Run All**
3. Wait ~30 seconds

**What you should see:**
- ✅ 14 sections of analysis with charts
- ✅ Histograms, correlation heatmap, monthly flood probability bar chart
- ✅ Bottom: "FLOODSENSE DATA QUALITY — PITCH NUMBERS" summary

**If anything errors:** check that `data/` folder exists and contains the CSVs. The notebook expects `DATA_DIR = Path('data')` in its first cell.

---

### Stage C — Train the model (5 minutes)

#### C.1 — Open the training notebook

Back in the Jupyter browser tab (from B.1), click `floodsense_training.ipynb`.

#### C.2 — Run all cells

Menu → **Cell → Run All**. Wait 30–60 seconds.

**What you should see:**
- ✅ Cleaning logs: "Removed 2 phantom rows", "Removed 67 duplicates"
- ✅ Test metrics: accuracy ~72%, recall_flood ~59%, roc_auc ~0.74
- ✅ Confusion matrix heatmap
- ✅ SHAP summary plot
- ✅ Bottom: "✅ Notebook complete"

#### C.3 — Verify artifacts were created

In your file explorer, open `floodsense/artifacts/`. You should see:
- `model.pkl` (~570 KB)
- `feature_cols.pkl`
- `metrics.json`
- `shap_summary.png`

These four files are what the dashboard will load.

---

### Stage D — Launch the dashboard (2 minutes)

#### D.1 — Open a SECOND terminal

Keep Jupyter running in the first terminal. Open a brand new terminal window.

#### D.2 — Navigate to the project folder

**Windows:**
```bash
cd %USERPROFILE%\Documents\floodsense
```

**Mac/Linux:**
```bash
cd ~/Documents/floodsense
```

#### D.3 — Launch Streamlit

```bash
streamlit run app.py
```

After 5–10 seconds, a browser tab opens at `http://localhost:8501`.

**What you should see:**
- A blue gradient header with "🌊 FloodSense" and Urdu subtitle
- A white card with 5 input fields (district, date, rainfall, soil, visible water)
- A blue "🔍 Assess Flood Risk" button

#### D.4 — Test 4 scenarios

Click each input, set the values, then click "Assess Flood Risk":

| # | District | Date | Rainfall | Soil | Visible water | Expected |
|---|----------|------|----------|------|---------------|----------|
| 1 | KP_District | 2025-08-15 | 150 | Saturated | Yes | 🔴 **CRITICAL** with safety override panel |
| 2 | Sindh_District | 2024-09-05 | 50 | Moist | No | 🟠 **HIGH** |
| 3 | Balochistan_District | 2024-01-15 | 2 | Dry | No | 🟢 **LOW** |
| 4 | KP_District | 2024-07-01 | 30 | Saturated | No | 🟡 **MEDIUM** |

If all four match → your system works. ✅

---

## 4. GitHub upload walkthrough

### G.1 — Create a GitHub account (if you don't have one)

Go to https://github.com → "Sign up" → use a school or personal email → choose a username.

### G.2 — Create an empty repository

1. Click the **+** icon (top right) → **New repository**
2. **Repository name:** `floodsense` (or your preferred name)
3. **Description:** `AI-powered flood early warning system for Pakistan`
4. Set to **Public** (judges need to view it)
5. **DO NOT** check "Add a README", "Add .gitignore", or "Choose a license" — leave everything blank
6. Click **Create repository**

GitHub now shows a setup page. **Keep this tab open** — you'll copy a command from it in G.5.

### G.3 — Initialize Git in your project folder

In your terminal (inside `floodsense`):

```bash
git init
git branch -M main
```

### G.4 — First commit

```bash
git add .
git commit -m "FloodSense: initial submission"
```

**If `git commit` complains** about identity, run these once:
```bash
git config --global user.email "your-email@example.com"
git config --global user.name "Your Name"
```
Then re-run the commit.

### G.5 — Connect to GitHub and push

Go back to the GitHub tab from G.2. Under **"...or push an existing repository from the command line"**, you'll see two lines like this (with **your username**):

```bash
git remote add origin https://github.com/YOUR-USERNAME/floodsense.git
git push -u origin main
```

Copy and paste them into your terminal. Press Enter.

### G.6 — Authenticate

GitHub will ask for credentials. **Don't use your GitHub password** — it won't work. Instead:

**Option 1 (easier — browser popup):** A browser window opens, click "Authorize"

**Option 2 (manual — if no popup):**
1. Go to GitHub → click your avatar (top right) → **Settings**
2. Scroll to **Developer settings** (bottom left)
3. **Personal access tokens** → **Tokens (classic)** → **Generate new token (classic)**
4. Note: `floodsense submission`. Expiration: 7 days.
5. **Check the `repo` scope** (top of the scope list)
6. Generate token → **copy it**
7. In your terminal, when asked for password, paste the token

### G.7 — Verify the upload

Refresh your GitHub repo page. You should see all your files. ✅

Click `README.md` to confirm it renders nicely. Click `floodsense_training.ipynb` to confirm GitHub displays it as a notebook (with charts and outputs visible).

### G.8 — Lock in your submission

The hackathon rule: **"Submit a GitHub link with commits timestamped after the Day 2 code freeze at 12:00 PM"** = instant disqualification.

To verify your timestamp is in time:
1. Click on your latest commit on GitHub (top of the file list)
2. You'll see "committed X minutes ago" or a date — make sure it's **before** the freeze deadline

If you need to make small changes after the freeze, **don't** push them. Test locally instead.

---

## 5. Hackathon submission checklist

Run through this checklist 1 hour before the freeze. If anything is unchecked, fix it.

### Code & repo
- [ ] All required files present in repo (see Section 2 list)
- [ ] `README.md` displays correctly on GitHub
- [ ] `floodsense_eda.ipynb` opens and shows charts on GitHub
- [ ] `floodsense_training.ipynb` opens and shows test metrics on GitHub
- [ ] `data/` folder contains all 4 dataset files
- [ ] `artifacts/` folder contains `model.pkl`, `feature_cols.pkl`, `metrics.json`, `shap_summary.png`
- [ ] Last commit timestamp is **before** the code freeze

### Working system
- [ ] `pip install -r requirements.txt` succeeds on a fresh machine
- [ ] `python train.py` (or running the notebook) produces artifacts without errors
- [ ] `streamlit run app.py` opens a browser tab with the dashboard
- [ ] All 4 test scenarios from Stage D produce the expected risk levels
- [ ] The Buner-150mm scenario shows the safety override expander

### Pitch readiness
- [ ] You can recite the 3-minute pitch (see Section 6)
- [ ] You have prepared answers for the 3 standard Q&A questions
- [ ] You have memorized: 72% accuracy, 59% recall, 0.74 ROC AUC, 6.9M affected in 2025, 1.4M Dadu population
- [ ] Live demo has been practiced ≥3 times with random inputs from a teammate

### Demo backup plan
- [ ] (Optional) Streamlit Cloud deployment is live at a working URL
- [ ] (Optional) Phone has the cloud URL bookmarked
- [ ] Laptop is fully charged
- [ ] Laptop's HDMI / display adapter is in your bag

---

## 6. Live demo playbook

The brief: a judge hands you a card with a district + rainfall. You enter the values live. No slides, no recordings. 5-minute demo + 3-minute pitch + 2-minute Q&A.

### Setup (BEFORE walking on stage)

- Laptop powered on
- App already running (`streamlit run app.py` in a visible terminal)
- Browser tab open at `http://localhost:8501`
- Phone in pocket with backup URL ready
- One teammate beside you, ready to handle technical issues

### The 3-minute pitch (memorize this)

> "Pakistan does not just have a flood problem — it has a warning problem.
>
> In 2022, monsoon floods affected 14.5 million Sindhis. In 2025, 6.9 million more were displaced — including 275 children killed. Communities lost everything in hours, because warnings came too late, in formats district officials couldn't act on.
>
> **FloodSense** is an AI-powered flood risk classifier with a bilingual dashboard that any district official can use in five seconds. Five inputs in: district, date, rainfall, soil condition, visible water. One output: a colour-coded risk badge in English and Urdu, with one specific recommended action.
>
> **Technically:** we used XGBoost, a gradient-boosted ensemble of decision trees, trained as a supervised binary classifier. We dropped the `ds_idx` column when our EDA found it was target leakage. We dropped two phantom rows with sentinel values. We engineered lag and rolling rainfall features per district. We chose XGBoost with `scale_pos_weight` to handle the 32/68 class imbalance.
>
> We hit **72% accuracy with 59% flood recall and 0.74 ROC AUC**. We deliberately chose a forecast-only framing over a higher-accuracy concurrent model — because a system that uses today's water area to detect today's flood is not an early warning system.
>
> Now let me show you. *[Click Assess Flood Risk on the demo card]*
>
> The system runs on a slow laptop with no internet, in under a second. It speaks the official's language. It shows confidence, population at risk, and one specific action.
>
> What if our model is wrong? Two failure modes — false negatives miss real floods, false positives waste resources. We address both. Our rule-based safety override forces a Critical alert any time rainfall exceeds 100mm in 24 hours, regardless of model output. You saw it fire on the Buner case. And we show a borderline-result warning between 40 and 60% confidence, prompting human review before alerts go out.
>
> **For Dadu's 1.4 million residents, this could mean 6 to 12 additional hours of warning.** That's the difference between losing everything and losing nothing."

### Q&A — prepared answers

**Q: "What ML technique did you use?"**
> "Supervised binary classification using XGBoost — a gradient-boosted ensemble of decision trees. We chose this over a single tree because the brief rewards ensemble methods, and over a neural network because trees dominate tabular problems at this data size while remaining interpretable through SHAP."

**Q: "What's your accuracy?"**
> "72% on a held-out test split. But accuracy isn't our headline metric — for a flood warning system, recall matters more, because missing a flood costs lives while a false alarm costs money. Our flood-class recall is 59%, ROC AUC is 0.74."

**Q: "What if your model is wrong?"**
> "Two failure modes. False negative: we miss a real flood. Recall is 59%, meaning we'd miss roughly 4 in 10 events without safeguards. Mitigation: the rule-based hard rules I just demonstrated, which force Critical alerts on extreme rainfall. False positive: an unnecessary evacuation costs lost workdays, supply expense, and erodes trust. Mitigation: our 'borderline result' warning between 40 and 60% confidence, prompting human review before triggering alerts."

**Q: "Did you do EDA?"**
> "Yes — we have a separate 14-section EDA notebook in the repo. Every cleaning decision and every feature engineering choice traces back to a specific finding in that notebook. Want me to walk through it?"

**Q: "How early would FloodSense have warned about Buner on August 15, 2025?"**
> "Running our model with the conditions preceding August 15th — saturated soil from prior monsoon weeks, atmospheric pressure drop, KP terrain — combined with our 100mm-rainfall safeguard rule that fires the moment sensors register the cloudburst, we estimate **6 to 12 hours of additional warning** for downstream districts. Our key assumption: rainfall sensors report in near-real-time."

### Live demo choreography

| Time | Action |
|------|--------|
| 0:00 – 0:30 | Take the card from judge, read aloud: *"District: Sindh, Rainfall: 80 millimetres."* |
| 0:30 – 1:30 | Type values into dashboard, narrating each one out loud |
| 1:30 – 1:45 | Click "Assess Flood Risk" |
| 1:45 – 2:15 | Read the badge **in both English and Urdu** |
| 2:15 – 3:00 | If safety override appears, expand it and explain the rule |
| 3:00 – 5:00 | Q&A — use the prepared answers above |

### If something fails on stage

| Failure | Recovery |
|---------|----------|
| Localhost dashboard won't load | Pull out phone with Streamlit Cloud URL, mirror to projector |
| Browser shows error | `Ctrl+R` to reload; if still broken, kill terminal with `Ctrl+C` and re-run `streamlit run app.py` |
| Both URLs fail | Open Jupyter tab with the training notebook, walk through Cell 8 (training output) and Cell 12 (confusion matrix). Frame it as: *"Our deployed app is having transient issues — this is the model that powers it."* |
| Wi-Fi dies | Streamlit running locally needs zero internet. As long as your laptop and HDMI cable work, you're fine. |

---

## 7. Troubleshooting

| Problem | Fix |
|---------|-----|
| `python` not recognized | Reinstall Python with "Add to PATH" checked |
| `pip` not recognized | Try `pip3` or `python -m pip install -r requirements.txt` |
| `jupyter` not recognized | `pip install jupyter` |
| `streamlit` not recognized | `pip install streamlit` |
| FileNotFoundError on data CSVs | Check that `data/` folder exists in your project root |
| FileNotFoundError on model.pkl | You haven't trained yet. Run `floodsense_training.ipynb` first. |
| Notebook charts don't render | `pip install matplotlib seaborn` and reload the page |
| Streamlit page is blank | Refresh browser; if persists, check terminal for Python errors |
| `git push` rejects with "Permission denied" | You're using your password instead of a Personal Access Token. See G.6. |
| Streamlit Cloud build fails "module not found" | Add the missing module to `requirements.txt`, commit, push — auto-redeploys |
| Demo accuracy ≠ 72% on first run | Library version drift — anywhere from 70% to 73% is acceptable |

---

## 8. Time budget

If you follow this guide top-to-bottom:

| Stage | Time |
|-------|------|
| A. One-time setup | 15 min |
| B. EDA notebook | 10 min |
| C. Training notebook | 10 min |
| D. Dashboard test | 5 min |
| E. GitHub upload | 15 min |
| Pitch practice (3 runs) | 30 min |
| **Total to ready** | **~85 min** |

You will have a working, defensible, deployed submission in **under 1.5 hours**. Anything beyond that is polish.

Now: open Stage A.1, open a terminal, and start. If something errors out, **paste the exact error into the chat** and I'll debug it with you in real time.

Good luck. 🌊
