import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, boxcox
from scipy.stats import skewnorm
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('ITA105_Lab_7.csv')

# BÀI 1: Phân tích phân phối & skewness
num_cols = df.select_dtypes(include='number').columns
skew_series = df[num_cols].skew().abs().sort_values(ascending=False)

print("=== TOP 10 CỘT LỆCH NHẤT ===")
print(skew_series.head(10).to_frame(name='|Skewness|').round(4))

top3 = skew_series.head(3).index.tolist()
print(f"\nTop 3: {top3}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(top3):
    sns.histplot(df[col], kde=True, ax=axes[i], color='steelblue')
    axes[i].set_title(f'{col}\nskew={df[col].skew():.2f}')
plt.suptitle('Bài 1 – Histogram + KDE (3 cột lệch mạnh nhất)', fontsize=13)
plt.tight_layout()
plt.savefig('bai1_histogram_kde.png', dpi=100)
plt.show()

"""
NHẬN XÉT BÀI 1:
- SalePrice & LotArea lệch phải (skew > 0): tập trung giá thấp,
  tồn tại nhiều outlier nhà giá cao → phân phối đuôi dài bên phải.
- NegSkewIncome lệch trái (skew < 0): thu nhập tập trung cao,
  có outlier âm rất nhỏ.
- Outlier làm lệch mean, ảnh hưởng hệ số hồi quy.
- Đề xuất: dùng log/Box-Cox cho cột dương (SalePrice, LotArea),
  Yeo-Johnson cho cột có giá trị âm (NegSkewIncome, MixedFeature).
- Tác động lên mô hình: skewness cao → vi phạm giả định phân phối
  chuẩn của OLS → RMSE tăng, hệ số ước lượng kém ổn định.
"""

# BÀI 2: Biến đổi dữ liệu nâng cao
df['SalePrice_log']    = np.log1p(df['SalePrice'])
df['LotArea_log']      = np.log1p(df['LotArea'])

df['SalePrice_boxcox'], lam1 = boxcox(df['SalePrice'])
df['LotArea_boxcox'],   lam2 = boxcox(df['LotArea'])
print(f"\nλ Box-Cox – SalePrice: {lam1:.4f} | LotArea: {lam2:.4f}")

yj = PowerTransformer(method='yeo-johnson')
df['NegSkewIncome_yeo'] = yj.fit_transform(df[['NegSkewIncome']])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, title in zip(axes,
    ['SalePrice', 'SalePrice_log', 'SalePrice_boxcox'],
    ['Gốc', 'Sau Log', 'Sau Box-Cox']):
    sns.histplot(df[col], kde=True, ax=ax, color='coral')
    ax.set_title(f'SalePrice – {title}\nskew={df[col].skew():.3f}')
plt.suptitle('Bài 2 – SalePrice: Trước & Sau Biến Đổi', fontsize=13)
plt.tight_layout()
plt.savefig('bai2_transform_compare.png', dpi=100)
plt.show()

compare = pd.DataFrame({
    'Cột': ['SalePrice', 'LotArea', 'NegSkewIncome'],
    'Skew gốc': [df['SalePrice'].skew(), df['LotArea'].skew(), df['NegSkewIncome'].skew()],
    'Skew Log':  [df['SalePrice_log'].skew(), df['LotArea_log'].skew(), None],
    'Skew BoxCox': [df['SalePrice_boxcox'].skew(), df['LotArea_boxcox'].skew(), None],
    'Skew YeoJohnson': [None, None, df['NegSkewIncome_yeo'].skew()],
    'Nhận xét': [
        'Box-Cox tốt nhất vì tìm λ tối ưu',
        'Log & Box-Cox tương đương',
        'Yeo-Johnson phù hợp vì có giá trị âm'
    ]
}).round(4)
print("\n=== BẢNG SO SÁNH BÀI 2 ===")
print(compare.to_string(index=False))

"""
Ý nghĩa λ trong Box-Cox:
  λ = 1  → không biến đổi (giữ nguyên)
  λ = 0  → log transform
  λ = 0.5 → căn bậc hai
  λ tối ưu được chọn để maximize log-likelihood → dữ liệu gần chuẩn nhất.
"""

# BÀI 3: Mô hình hóa – 3 phiên bản
features = ['LotArea', 'HouseAge', 'Rooms']

# Version A – dữ liệu gốc
X = df[features]
y_A = df['SalePrice']

# Version B – log biến mục tiêu
y_B = np.log1p(df['SalePrice'])

# Version C – PowerTransformer toàn bộ feature + target
pt = PowerTransformer(method='yeo-johnson')
X_C = pd.DataFrame(pt.fit_transform(df[features]), columns=features)
y_C = np.log1p(df['SalePrice'])

results = {}
for version, X_use, y_use, name in [
    ('A', X,   y_A, 'Gốc'),
    ('B', X,   y_B, 'Log target'),
    ('C', X_C, y_C, 'PowerTransform'),
]:
    Xtr, Xte, ytr, yte = train_test_split(X_use, y_use, test_size=0.2, random_state=42)
    model = LinearRegression().fit(Xtr, ytr)
    pred  = model.predict(Xte)

    if version == 'B':          
        yte_real  = np.expm1(yte)
        pred_real = np.expm1(pred)
        rmse = np.sqrt(mean_squared_error(yte_real, pred_real))
        r2   = r2_score(yte_real, pred_real)
    elif version == 'C':
        yte_real  = np.expm1(yte)
        pred_real = np.expm1(pred)
        rmse = np.sqrt(mean_squared_error(yte_real, pred_real))
        r2   = r2_score(yte_real, pred_real)
    else:
        rmse = np.sqrt(mean_squared_error(yte, pred))
        r2   = r2_score(yte, pred)

    results[name] = {'RMSE': round(rmse, 2), 'R²': round(r2, 4)}

print("\n=== BẢNG KẾT QUẢ BÀI 3 ===")
print(pd.DataFrame(results).T.to_string())

"""
Phân tích:
- Log transform thu nhỏ khoảng cách outlier → RMSE thực tế giảm.
- PowerTransformer giúp feature phân phối đều hơn → giảm nhiễu,
  ổn định hệ số hồi quy.
- Mô hình B/C thường ổn định hơn A trên dữ liệu lệch mạnh.
"""

# BÀI 4: Ứng dụng nghiệp vụ
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Version A – raw
sns.histplot(df['SalePrice'], kde=True, ax=axes[0,0], color='tomato')
axes[0,0].set_title(f'SalePrice – Raw (skew={df["SalePrice"].skew():.2f})')

sns.histplot(df['LotArea'], kde=True, ax=axes[0,1], color='tomato')
axes[0,1].set_title(f'LotArea – Raw (skew={df["LotArea"].skew():.2f})')

# Version B – log transform
sns.histplot(df['SalePrice_log'], kde=True, ax=axes[1,0], color='seagreen')
axes[1,0].set_title(f'SalePrice – Log (skew={df["SalePrice_log"].skew():.2f})')

sns.histplot(df['LotArea_log'], kde=True, ax=axes[1,1], color='seagreen')
axes[1,1].set_title(f'LotArea – Log (skew={df["LotArea_log"].skew():.2f})')

plt.suptitle('Bài 4 – Raw vs. Log Transform', fontsize=13)
plt.tight_layout()
plt.savefig('bai4_raw_vs_log.png', dpi=100)
plt.show()

df['log_price_index'] = df['SalePrice_log'] / df['LotArea_log']
print("\n=== LOG PRICE INDEX (5 mẫu) ===")
print(df[['SalePrice', 'LotArea', 'log_price_index']].head())

"""
INSIGHT (dành cho người không chuyên):
- Raw data: đuôi dài bên phải khiến biểu đồ bị kéo dãn → khó
  nhìn rõ phần lớn nhà có giá phổ thông.
- Sau log: biểu đồ cân đối hơn → dễ thấy khoảng giá phổ biến,
  phát hiện nhà giá bất thường nhanh hơn.

log-price-index = log(SalePrice) / log(LotArea):
  - Cao: giá đắt so với diện tích → phát hiện khu vực "thổi giá".
  - Thấp: diện tích lớn, giá hợp lý → cơ hội đầu tư.

KHUYẾN NGHỊ KINH DOANH:
  1. Dùng log-price-index để phân nhóm BĐS: cao cấp / trung bình / bình dân.
  2. Nhà có index ngoại lệ (rất cao) cần xem xét lại định giá.
  3. Mô hình dự báo nên dùng log(SalePrice) làm target để giảm
     ảnh hưởng của các căn nhà siêu đắt.
"""