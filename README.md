# 🚢 Titanic Survival Predictor

A fully deployable Flask web app that predicts passenger survival on the Titanic using pre-trained KNN models.

---

## Project Structure

```
titanic_app/
├── app.py                  # Flask backend — prediction API
├── requirements.txt        # Python dependencies
├── README.md
├── models/
│   ├── Scaler.pkl          # StandardScaler (numeric features)
│   ├── encoder.pkl         # OneHotEncoder (categorical features)
│   ├── model.pkl           # KNeighborsClassifier (k=5)
│   └── model1.pkl          # KNeighborsClassifier (k=7)
├── templates/
│   └── index.html          # Main HTML page (Jinja2)
└── static/
    └── style.css           # Sunny ocean theme stylesheet
```

---

## Quick Start (Local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠ The models were saved with **scikit-learn 1.6.1**.  
> Using a different version may cause warnings or errors.  
> It is strongly recommended to match: `pip install scikit-learn==1.6.1`

### 2. Run the app

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## Deployment

### Option A — Heroku

```bash
# Install Heroku CLI, then:
heroku create your-app-name
git init && git add . && git commit -m "initial"
heroku git:remote -a your-app-name
git push heroku main
```

Add a `Procfile`:
```
web: gunicorn app:app
```

Add `gunicorn` to `requirements.txt`.

### Option B — Railway / Render

1. Push this folder to a GitHub repo.
2. Connect the repo in Railway or Render.
3. Set start command: `gunicorn app:app`
4. Deploy.

### Option C — Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t titanic-app .
docker run -p 5000:5000 titanic-app
```

---

## API Reference

### `POST /predict`

**Request body (JSON):**

| Field        | Type    | Values                          |
|--------------|---------|---------------------------------|
| `pclass`     | int     | 1, 2, 3                         |
| `age`        | float   | 1 – 80                          |
| `sibsp`      | int     | 0 – 8                           |
| `parch`      | int     | 0 – 9                           |
| `fare`       | float   | 0 – 500                         |
| `sex`        | string  | "female" / "male"               |
| `embarked`   | string  | "C" / "Q" / "S"                 |
| `class`      | string  | "First" / "Second" / "Third"    |
| `who`        | string  | "woman" / "man" / "child"       |
| `adult_male` | string  | "true" / "false"                |
| `alive`      | string  | "yes" / "no"                    |
| `alone`      | string  | "true" / "false"                |
| `model`      | string  | "knn5" / "knn7" (optional)      |

**Response (JSON):**

```json
{
  "survived":       true,
  "survival_prob":  0.8,
  "perished_prob":  0.2,
  "model_used":     "KNN (k=5)"
}
```

---

## Models

| File        | Type                   | k  | Features |
|-------------|------------------------|----|----------|
| model.pkl   | KNeighborsClassifier   | 5  | 22       |
| model1.pkl  | KNeighborsClassifier   | 7  | 22       |

**Feature pipeline:**
1. Numeric features `[pclass, age, sibsp, parch, fare]` → `StandardScaler`
2. Categorical features `[sex, embarked, class, who, adult_male, alive, alone]` → `OneHotEncoder`
3. Concatenated → 22-dimensional feature vector → KNN classifier
