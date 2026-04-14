import subprocess, sys

PACKAGES = {
    'joblib':      'joblib',
    'sklearn':     'scikit-learn',
    'pandas':      'pandas',
    'numpy':       'numpy',
    'matplotlib':  'matplotlib',
    'seaborn':     'seaborn',
}

for import_name, pip_name in PACKAGES.items():
    try:
        __import__(import_name)
    except ImportError:
        print(f"Dang cai: {pip_name} ...")
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', pip_name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"  Da cai xong: {pip_name}")
# ────────────────────────────────────────────────────────────

import warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# ── LOAD DATA ────────────────────────────────────────────────
df = pd.read_csv('ITA105_Lab_8.csv')
print(f"Dataset: {df.shape}\n")

TARGET   = 'SalePrice'
NUM_COLS = ['LotArea', 'Rooms', 'HasGarage', 'NoiseFeature']
CAT_COLS = ['Neighborhood', 'Condition']
TEXT_COL = 'Description'
DATE_COL = 'SaleDate'

X = df.drop(columns=[TARGET, 'ImagePath'])
y = df[TARGET]

# BAI 1: Xay dung Pipeline tong quat
class IQRClipper(BaseEstimator, TransformerMixin):
    """Outlier removal bang IQR clipping."""
    def fit(self, X, y=None):
        arr = np.array(X, dtype=float)
        q1  = np.nanpercentile(arr, 25, axis=0)
        q3  = np.nanpercentile(arr, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - 1.5 * iqr
        self.upper_ = q3 + 1.5 * iqr
        return self
    def transform(self, X, y=None):
        return np.clip(np.array(X, dtype=float), self.lower_, self.upper_)

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Trich thang va quy tu cot ngay."""
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        s = pd.to_datetime(pd.DataFrame(X).iloc[:, 0], errors='coerce')
        month   = s.dt.month.fillna(s.dt.month.median()).values
        quarter = s.dt.quarter.fillna(2).values
        return np.column_stack([month, quarter])

# Sub-pipelines
num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('clip',   IQRClipper()),
    ('scale',  StandardScaler()),
    ('power',  PowerTransformer(method='yeo-johnson')),
])
cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
text_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20, stop_words='english')),
])
date_pipe = Pipeline([
    ('extract', DateFeatureExtractor()),
    ('scale',   StandardScaler()),
])

preprocessor = ColumnTransformer([
    ('num',  num_pipe,  NUM_COLS),
    ('cat',  cat_pipe,  CAT_COLS),
    ('text', text_pipe, TEXT_COL),
    ('date', date_pipe, [DATE_COL]),
], remainder='drop')

# Smoke test
print("=== SMOKE TEST (10 dong) ===")
out_demo = preprocessor.fit_transform(X.head(10))
print(f"Output shape: {out_demo.shape}")

preprocessor.fit(X)
cat_names   = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(CAT_COLS).tolist()
tfidf_names = preprocessor.named_transformers_['text']['tfidf'].get_feature_names_out().tolist()
all_features = NUM_COLS + cat_names + tfidf_names + ['month', 'quarter']
print(f"Tong features: {len(all_features)}")
print("Vi du:", all_features[:5], "...", all_features[-3:])

# BAI 2: Kiem thu 5 bo du lieu
print("\n=== BAI 2: KIEM THU 5 BO DU LIEU ===")

def run_test(name, X_input):
    try:
        out = preprocessor.transform(X_input)
        print(f"  [{name}] OK  shape={out.shape} | nan={np.isnan(out).any()}")
    except Exception as e:
        print(f"  [{name}] LOI: {e}")

run_test("Du lieu day du",        X.head(50))

X_miss = X.head(50).copy()
X_miss.loc[X_miss.sample(frac=0.4, random_state=0).index, 'LotArea']      = np.nan
X_miss.loc[X_miss.sample(frac=0.4, random_state=1).index, 'Neighborhood'] = np.nan
run_test("Missing nhieu",         X_miss)

X_skew = X.head(50).copy()
X_skew['LotArea'] = X_skew['LotArea'] ** 3
run_test("Lech phan phoi",        X_skew)

X_unseen = X.head(50).copy()
X_unseen['Neighborhood'] = 'UNKNOWN_Z'
run_test("Unseen categories",     X_unseen)

X_bad = X.head(50).copy()
X_bad['LotArea'] = pd.to_numeric(
    X_bad['LotArea'].astype(str).replace({'2346': 'N/A'}), errors='coerce')
run_test("Sai dinh dang (fixed)", X_bad)

# Bieu do truoc/sau
X_full = preprocessor.transform(X)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['LotArea'],  bins=40, color='tomato',    edgecolor='white')
axes[0].set_title('LotArea - Truoc pipeline')
axes[1].hist(X_full[:, 0],   bins=40, color='steelblue', edgecolor='white')
axes[1].set_title('LotArea - Sau pipeline')
plt.tight_layout()
plt.savefig('bai2_before_after.png', dpi=100)
plt.show()
print("Bieu do luu: bai2_before_after.png")

print("""
Bao cao loi & cach sua:
  1. Missing        -> SimpleImputer tu xu ly (median / most_frequent)
  2. Lech phan phoi -> IQRClipper cat outlier + PowerTransformer can bang
  3. Unseen cat     -> handle_unknown='ignore' encode thanh vector 0
  4. Sai dinh dang  -> pd.to_numeric(errors='coerce') truoc pipeline
  5. Shape nhat quan -> ColumnTransformer tao cung so cot sau khi fit
""")

# BAI 3: Pipeline + Mo hinh + Cross-validation
print("=== BAI 3: PIPELINE + MO HINH ===")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

models = {
    'LinearRegression': LinearRegression(),
    'RandomForest':     RandomForestRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    cv   = cross_validate(pipe, X_train, y_train, cv=5,
                          scoring=['neg_root_mean_squared_error', 'r2'])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    results[name] = {
        'RMSE':         round(np.sqrt(mean_squared_error(y_test, pred)), 2),
        'MAE':          round(mean_absolute_error(y_test, pred), 2),
        'R2':           round(r2_score(y_test, pred), 4),
        'CV_RMSE_avg':  round(-cv['test_neg_root_mean_squared_error'].mean(), 2),
        'CV_RMSE_std':  round( cv['test_neg_root_mean_squared_error'].std(),  2),
    }

    if name == 'RandomForest':
        imps    = pipe.named_steps['model'].feature_importances_
        top_idx = np.argsort(imps)[-10:][::-1]
        print(f"\nTop 10 features ({name}):")
        for i in top_idx:
            fname = all_features[i] if i < len(all_features) else f'feat_{i}'
            print(f"  {fname:<28} {imps[i]:.4f}")

print("\n=== KET QUA SO SANH ===")
print(pd.DataFrame(results).T.to_string())
print("""
Danh gia:
  - Pipeline tranh data leak: fit() chi hoc tren train, transform() ap rieng
    tren validation moi fold -> CV chuan hon xu ly thu cong.
  - Pipeline giam loi tay: khong quen buoc nao khi predict du lieu moi.
""")

# BAI 4: Deploy – ham predict_price()
print("=== BAI 4: DEPLOY PIPELINE ===")

best_pipe = Pipeline([
    ('prep',  preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42)),
])
best_pipe.fit(X_train, y_train)
joblib.dump(best_pipe, 'house_price_pipeline.pkl')
print("Pipeline luu: house_price_pipeline.pkl")

def predict_price(new_data: pd.DataFrame) -> np.ndarray:
    """
    Nhan DataFrame moi (khong can cot SalePrice / ImagePath).
    Tra ve mang numpy 1D chua gia du bao.

    Cot bat buoc: LotArea, Rooms, HasGarage, NoiseFeature,
                  Neighborhood, Condition, Description, SaleDate
    """
    pipeline = joblib.load('house_price_pipeline.pkl')
    required = NUM_COLS + CAT_COLS + [TEXT_COL, DATE_COL]
    for col in required:
        if col not in new_data.columns:
            raise ValueError(f"Thieu cot: {col}")
    for col in NUM_COLS:
        new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
    return pipeline.predict(new_data[required])

prices = predict_price(X_test.head(5).copy())
print("\nDu bao 5 nha moi:")
for i, p in enumerate(prices):
    print(f"  Nha {i+1}: ${p:,.0f}  (thuc te: ${y_test.iloc[i]:,.0f})")

print("""
TAI LIEU PIPELINE
  Buoc 1 - So  : Impute(median) -> IQRClipper -> StandardScaler -> PowerTransform
  Buoc 2 - Cat : Impute(mode)   -> OneHotEncoder(handle_unknown='ignore')
  Buoc 3 - Text: TF-IDF (20 tu, bo stopword tieng Anh)
  Buoc 4 - Date: Trich month + quarter -> StandardScaler
  Buoc 5 - Model: RandomForestRegressor(n_estimators=100)

  Dau vao : DataFrame 8 cot (xem required trong predict_price)
  Dau ra  : mang numpy 1D chua gia du bao

  Rui ro:
    - Unseen category : encode = 0 (an toan, khong crash)
    - Data drift      : phan phoi moi khac train -> nen retrain dinh ky
    - Sai format      : pd.to_numeric(errors='coerce') tu xu ly
    - Thieu cot       : ham raise ValueError ro rang
""")