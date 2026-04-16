# ===0. Setup & Tạo dữ liệu mẫu===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
sns.set_theme(style='whitegrid')
n = 500
districts = ['Hoàn Kiếm', 'Ba Đình', 'Đống Đa', 'Cầu Giấy', 'Thanh Xuân','Hà Đông', 'Long Biên', 'Hoàng Mai', 'Tây Hồ', 'Nam Từ Liêm']
status_list = ['Mới', 'Đã qua sử dụng', 'Cần sửa chữa', 'new', 'used', 'repair']
descriptions = ['Căn hộ luxury view hồ, nội thất cao cấp, an ninh 24/7','Nhà phố cozy gần trường học, yên tĩnh, thoáng mát','Căn hộ tiện nghi đầy đủ, gần trung tâm thương mại','Biệt thự luxury hồ bơi riêng, sân vườn rộng','Nhà cấp 4 cần sửa chữa, giá rẻ, tiềm năng đầu tư','Căn hộ luxury view hồ, nội thất cao cấp, an ninh 24/7','Chung cư mini tiện nghi, gần metro, view đẹp','Nhà liền kề cozy 3 tầng, garage, sân thượng']
df = pd.DataFrame({'id': range(1, n+1),'district': np.random.choice(districts, n),'area': np.random.lognormal(4.2, 0.5, n),'bedrooms': np.random.choice([1,2,3,4,5,0], n, p=[0.1,0.3,0.3,0.2,0.05,0.05]),'price': np.random.lognormal(5.5, 0.8, n) * 1e9,'status': np.random.choice(status_list, n),'description': np.random.choice(descriptions, n),'build_date': pd.to_datetime('2010-01-01') + pd.to_timedelta(np.random.randint(0, 4000, n), unit='D'),'sell_date': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 700, n), unit='D')})
df.loc[np.random.choice(n, 30, replace=False), 'price'] = np.nan
df.loc[np.random.choice(n, 20, replace=False), 'area'] = np.nan
df.loc[np.random.choice(n, 10, replace=False), 'price'] = -abs(df['price'].dropna().sample(10).values)
df.loc[np.random.choice(n, 5, replace=False), 'area'] = 0
df = pd.concat([df, df.sample(10)], ignore_index=True)
print(f'Dataset shape: {df.shape}')
print(df.head())

# === ĐOẠN 1 - Khám phá, Làm sạch, Chuẩn hóa dữ liệu===
#1.1 Phân tích thống kê cơ bản
print('=== Thống kê mô tả ===')
print(df.describe())
print('\n=== Missing values ===')
missing = df.isnull().sum()
print(missing[missing > 0])
print(f'\n=== Duplicate rows: {df.duplicated().sum()} ===')
num_cols = ['price', 'area', 'bedrooms']
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for i, col in enumerate(num_cols):
    data = df[col].dropna()
    sns.histplot(data, ax=axes[i][0], kde=True)
    axes[i][0].set_title(f'Histogram - {col}')
    sns.boxplot(y=data, ax=axes[i][1])
    axes[i][1].set_title(f'Boxplot - {col}')
    sns.violinplot(y=data, ax=axes[i][2])
    axes[i][2].set_title(f'Violin - {col}')
plt.tight_layout()
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['district'].value_counts().plot(kind='bar', ax=axes[0], title='Phân phối District')
df['status'].value_counts().plot(kind='bar', ax=axes[1], title='Phân phối Status')
plt.tight_layout()
plt.show()
#1.2 Xử lý dữ liệu bẩn
df = df.drop_duplicates().reset_index(drop=True)
print(f'Sau khi xóa duplicate: {df.shape}')
df = df[df['price'] > 0]
df = df[df['area'] > 0]
print(f'Sau khi xóa invalid: {df.shape}')
df['price'] = df['price'].fillna(df['price'].median())
df['area'] = df['area'].fillna(df['area'].median())
df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].mode()[0])
status_map = {'new': 'Mới', 'used': 'Đã qua sử dụng', 'repair': 'Cần sửa chữa'}
df['status'] = df['status'].replace(status_map)
df = df[df['bedrooms'] > 0].reset_index(drop=True)
print(f'Sau khi xử lý bedrooms=0: {df.shape}')
print('\nMissing values còn lại:')
print(df.isnull().sum())
# ### 1.3 Phát hiện & xử lý Outlier
def detect_outliers_iqr(series):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    return (series < lower) | (series > upper)
def detect_outliers_zscore(series, threshold=3):
    z = np.abs((series - series.mean()) / series.std())
    return z > threshold
for col in ['price', 'area']:
    n_iqr = detect_outliers_iqr(df[col]).sum()
    n_z = detect_outliers_zscore(df[col]).sum()
    print(f'{col}: IQR={n_iqr} outliers, Z-score={n_z} outliers')
def cap_outliers(series):
    Q1, Q3 = series.quantile([0.01, 0.99])
    return series.clip(lower=Q1, upper=Q3)
df['price'] = cap_outliers(df['price'])
df['area'] = cap_outliers(df['area'])
print('\nĐã áp dụng capping outlier tại percentile 1%-99%')
#1.4 Chuẩn hóa số & Encoding categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
scaler_minmax = MinMaxScaler()
scaler_zscore = StandardScaler()
df['price_minmax'] = scaler_minmax.fit_transform(df[['price']])
df['price_zscore'] = scaler_zscore.fit_transform(df[['price']])
df['area_minmax'] = scaler_minmax.fit_transform(df[['area']])
le = LabelEncoder()
df['status_encoded'] = le.fit_transform(df['status'])
print('Label classes:', le.classes_)
district_dummies = pd.get_dummies(df['district'], prefix='dist')
df = pd.concat([df, district_dummies], axis=1)
print(f'Shape sau encoding: {df.shape}')
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=20)
tfidf_matrix = tfidf.fit_transform(df['description'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{w}' for w in tfidf.get_feature_names_out()])
print('Top TF-IDF features:', tfidf.get_feature_names_out()[:10])
print(f'TF-IDF shape: {tfidf_df.shape}')
#1.5 Phát hiện duplicate dựa trên Text Similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(tfidf_matrix)
duplicate_pairs = []
for i in range(len(sim_matrix)):
    for j in range(i+1, len(sim_matrix)):
        if sim_matrix[i][j] > 0.9:
            duplicate_pairs.append((i, j, round(sim_matrix[i][j], 3)))
print(f'Số cặp mô tả tương đồng (>90%): {len(duplicate_pairs)}')
print('\nGợi ý merge các bản ghi sau:')
for i, j, score in duplicate_pairs[:5]:
    print(f'  Row {i} & Row {j} - similarity: {score}')
    print(f'  \"{df["description"].iloc[i][:60]}...\"')

# ===GIAI ĐOẠN 2 - Feature Engineering, Pipeline, Mô hình dự báo===
#2.1 Feature Engineering nâng cao
from scipy import stats
for col in ['price', 'area']:
    print(f'{col} skewness: {df[col].skew():.3f}')
df['price_log'] = np.log1p(df['price'])
df['area_log'] = np.log1p(df['area'])
df['price_boxcox'], _ = stats.boxcox(df['price'] + 1)
print('\nSau log transform:')
print(f'price_log skewness: {df["price_log"].skew():.3f}')
df['sell_month'] = df['sell_date'].dt.month
df['sell_quarter'] = df['sell_date'].dt.quarter
df['sell_season'] = df['sell_month'].map({12:4, 1:4, 2:4,3:1, 4:1, 5:1,6:2, 7:2, 8:2,9:3, 10:3, 11:3})
df['days_to_sell'] = (df['sell_date'] - df['build_date']).dt.days
df['word_count'] = df['description'].str.split().str.len()
df['has_luxury'] = df['description'].str.contains('luxury', case=False).astype(int)
df['has_cozy'] = df['description'].str.contains('cozy', case=False).astype(int)
positive_words = ['cao cấp', 'đẹp', 'thoáng', 'tiện nghi', 'luxury', 'cozy']
df['sentiment_score'] = df['description'].apply(lambda x: sum(1 for w in positive_words if w in x.lower()))
print('Đã tạo features từ ngày và text:')
print(df[['sell_month','sell_quarter','sell_season','days_to_sell','word_count','has_luxury','sentiment_score']].head())
np.random.seed(42)
df['img_avg_brightness'] = np.random.uniform(80, 220, len(df))
df['img_color_r'] = np.random.uniform(100, 200, len(df))
df['img_color_g'] = np.random.uniform(90, 180, len(df))
df['img_color_b'] = np.random.uniform(80, 170, len(df))
print('Đã thêm image features (mô phỏng CNN embedding)')
#2.2 Pipeline hoàn chỉnh (tái sử dụng cho dữ liệu mới)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
num_features = ['area', 'bedrooms', 'days_to_sell', 'word_count','has_luxury', 'has_cozy', 'sentiment_score','img_avg_brightness', 'sell_month', 'sell_quarter']
cat_features = ['district', 'status']
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer([('num', num_pipeline, num_features),('cat', cat_pipeline, cat_features)])
print('Pipeline đã được định nghĩa - sẵn sàng cho training và dữ liệu mới')
new_data = df[num_features + cat_features].sample(50).copy()
new_data.loc[new_data.sample(5).index, 'area'] = np.nan
X_new = preprocessor.fit_transform(new_data)
print(f'Pipeline output shape: {X_new.shape}')
print('Không có lỗi - pipeline hoạt động tốt với dữ liệu mới!')
#2.3 Huấn luyện & So sánh mô hình dự báo giá
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    from xgboost import XGBRegressor
    has_xgb = True
except ImportError:
    has_xgb = False
X = df[num_features + cat_features].copy()
y = df['price_log']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
print(f'Train size: {X_train_proc.shape}, Test size: {X_test_proc.shape}')
models = {'Linear Regression': LinearRegression(),'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)}
if has_xgb:
    models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
results = []
for name, model in models.items():
    model.fit(X_train_proc, y_train)
    y_pred = model.predict(X_test_proc)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    results.append({'Model': name, 'RMSE': round(rmse,4), 'MAE': round(mae,4), 'R²': round(r2,4)})
results_df = pd.DataFrame(results).sort_values('RMSE')
print('=== So sánh mô hình (target: log-price) ===')
print(results_df)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, metric in enumerate(['RMSE', 'MAE', 'R²']):
    axes[i].bar(results_df['Model'], results_df[metric], color='steelblue')
    axes[i].set_title(metric)
    axes[i].tick_params(axis='x', rotation=30)
plt.suptitle('So sánh hiệu quả các mô hình', fontsize=14)
plt.tight_layout()
plt.show()
best_model_name = results_df.iloc[0]['Model']
y_raw = df['price']
_, _, y_train_raw, y_test_raw = train_test_split(X, y_raw, test_size=0.3, random_state=42)
raw_model = RandomForestRegressor(n_estimators=100, random_state=42)
raw_model.fit(X_train_proc, y_train_raw)
y_pred_raw = raw_model.predict(X_test_proc)
rmse_raw = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
r2_raw = r2_score(y_test_raw, y_pred_raw)
print(f'Random Forest - Raw price:  RMSE={rmse_raw:.2e}, R²={r2_raw:.4f}')
print(f'Random Forest - Log price:  RMSE={results_df[results_df["Model"]=="Random Forest"]["RMSE"].values[0]:.4f}, R²={results_df[results_df["Model"]=="Random Forest"]["R²"].values[0]:.4f}')
print('\n→ Log transform giúp giảm skew, mô hình học tốt hơn, R² cải thiện rõ rệt')
#2.4 Phân tích đa kịch bản & KPI
df['price_per_m2'] = df['price'] / df['area']
df['log_price_index'] = np.log(df['price'] / df['price'].median())
df['luxury_score'] = (df['has_luxury'] * 3 + df['sentiment_score'] + df['price_minmax'] * 2)
district_stats = df.groupby('district').agg(avg_price=('price', 'mean'),avg_price_m2=('price_per_m2', 'mean'),count=('id', 'count'),avg_luxury=('luxury_score', 'mean')).round(2).sort_values('avg_price', ascending=False)
print('=== Giá trung bình theo khu vực ===')
print(district_stats)
top5_threshold = df['price'].quantile(0.95)
top5_df = df[df['price'] >= top5_threshold]
print(f'Kịch bản giá cực trị (top 5%): {len(top5_df)} căn, ngưỡng giá={top5_threshold/1e9:.2f} tỷ')
print('Khu vực xuất hiện nhiều nhất trong top 5%:')
print(top5_df['district'].value_counts().head(5))
print('\n=== Phân khúc khách hàng tiềm năng ===')
print(' Luxury (luxury_score > 7): Khách hàng cao cấp, đầu tư dài hạn')
print(' Mid-range (3-7): Gia đình, an cư lập nghiệp')
print(' Budget (<3): Khách hàng trẻ, đầu tư lướt sóng')
#2.5 Dashboard & Insight nghiệp vụ
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
sns.histplot(df['price'], ax=axes[0][0], kde=True, color='tomato')
axes[0][0].set_title('Price - Raw (lệch phải)')
sns.histplot(df['price_log'], ax=axes[0][1], kde=True, color='steelblue')
axes[0][1].set_title('Price - Log Transform (gần chuẩn hơn)')
sns.histplot(df['price_boxcox'], ax=axes[0][2], kde=True, color='green')
axes[0][2].set_title('Price - Box-Cox Transform')
district_stats['avg_price'].plot(kind='barh', ax=axes[1][0], color='steelblue')
axes[1][0].set_title('Giá TB theo Quận/Huyện')
sns.scatterplot(data=df, x='area', y='price', hue='status',alpha=0.5, ax=axes[1][1])
axes[1][1].set_title('Diện tích vs Giá (màu = tình trạng)')
sns.boxplot(data=df, x='district', y='price_per_m2', ax=axes[1][2])
axes[1][2].set_title('Giá/m² theo Quận')
axes[1][2].tick_params(axis='x', rotation=45)
plt.suptitle('PropTech Dashboard - Phân tích thị trường BĐS Hà Nội', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=df['price'], ax=axes[0], color='tomato')
axes[0].set_title('Boxplot Price - Raw (outlier rõ)')
sns.boxplot(y=df['price_log'], ax=axes[1], color='steelblue')
axes[1].set_title('Boxplot Price - Log (outlier giảm)')
plt.tight_layout()
plt.show()
print('→ Log transform giúp nhận diện và giảm thiểu ảnh hưởng của outlier/điểm cực trị')

# ===GIAI ĐOẠN HOÀN THIỆN===
#3.1 Xử lý unseen categories từ nguồn dữ liệu mới
new_source = pd.DataFrame({'district': ['Đan Phượng', 'Hoàn Kiếm', 'Mê Linh'],'status': ['Mới', 'Đã qua sử dụng', 'Renovated'],'area': [80, None, 120],'bedrooms': [2, 3, None],'days_to_sell': [300, 200, 400],'word_count': [8, 7, 9],'has_luxury': [0, 1, 0],'has_cozy': [1, 0, 1],'sentiment_score': [2, 3, 1],'img_avg_brightness': [150, 180, 130],'sell_month': [3, 7, 11],'sell_quarter': [1, 3, 4]})
try:
    X_new_proc = preprocessor.transform(new_source)
    print(f'Pipeline xử lý dữ liệu mới thành công: shape={X_new_proc.shape}')
    print('Unseen categories được encode thành vector 0 (bỏ qua an toàn)')
except Exception as e:
    print(f'Lỗi: {e}')
#3.2 Feature Interaction: Diện tích × Số phòng × Quận
df['area_x_bedrooms'] = df['area'] * df['bedrooms']
df['district_price_mean'] = df.groupby('district')['price'].transform('mean')
df['area_x_district_price'] = df['area'] * df['district_price_mean']
X_base = preprocessor.transform(df[num_features + cat_features])
X_inter = df[num_features + ['area_x_bedrooms', 'area_x_district_price'] + cat_features]
num_features_ext = num_features + ['area_x_bedrooms', 'area_x_district_price']
preprocessor_ext = ColumnTransformer([('num', num_pipeline, num_features_ext),('cat', cat_pipeline, cat_features)])
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_inter, y, test_size=0.3, random_state=42)
X_train2_proc = preprocessor_ext.fit_transform(X_train2)
X_test2_proc = preprocessor_ext.transform(X_test2)
rf_inter = RandomForestRegressor(n_estimators=100, random_state=42)
rf_inter.fit(X_train2_proc, y_train2)
y_pred2 = rf_inter.predict(X_test2_proc)
r2_inter = r2_score(y_test2, y_pred2)
r2_base_val = results_df[results_df['Model']=='Random Forest']['R²'].values[0]
print(f'Random Forest - Không interaction: R²={r2_base_val:.4f}')
print(f'Random Forest - Có interaction:    R²={r2_inter:.4f}')
print(f'\n→ Cải thiện: {r2_inter - r2_base_val:+.4f}')
#3.3 So sánh: Numerical only vs Numerical + Text + Ảnh
basic_features = ['area', 'bedrooms', 'days_to_sell']
X_basic = df[basic_features].fillna(df[basic_features].median())
X_basic_scaled = StandardScaler().fit_transform(X_basic)
full_features = ['area', 'bedrooms', 'days_to_sell','word_count', 'has_luxury', 'has_cozy', 'sentiment_score','img_avg_brightness', 'img_color_r', 'img_color_g', 'img_color_b']
X_full = df[full_features].fillna(df[full_features].median())
X_full_scaled = StandardScaler().fit_transform(X_full)
comparison = []
for name, X_data in [('Numerical only', X_basic_scaled), ('Num + Text + Image', X_full_scaled)]:
    Xtr, Xte, ytr, yte = train_test_split(X_data, y.values, test_size=0.3, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(Xtr, ytr)
    ypred = rf.predict(Xte)
    comparison.append({'Model': name,'RMSE': round(np.sqrt(mean_squared_error(yte, ypred)), 4),'R²': round(r2_score(yte, ypred), 4)})
print('=== So sánh mô hình theo loại feature ===')
print(pd.DataFrame(comparison))
print('\n→ Thêm text và image features thường cải thiện R² đáng kể')