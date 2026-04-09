import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

np.random.seed(42)
n = 200

data = {
    'gia_nha':   np.random.choice([np.nan, 500, -100, 1000, 1200, 800, 950, 1100, 5000], n),
    'dien_tich': np.random.choice([np.nan, 50, 80, 100, 120, 0, 60, 200], n),
    'so_phong':  np.random.choice([0, 1, 2, 3, 4, np.nan], n),
    'tinh_trang': np.random.choice(['mới', 'cũ', 'moi', None, 'MOI', 'Mới'], n),
    'vi_tri':    np.random.choice(['Hà Nội', 'HCM', 'Đà Nẵng', None], n),
    'mo_ta':     np.random.choice([
        'Nhà đẹp gần trung tâm',
        'Căn hộ hiện đại tiện nghi',
        'Nhà đẹp gần trung tâm',
        'Biệt thự sang trọng',
        None
    ], n),
    'thoi_gian_giao_dich': pd.date_range('2023-01-01', periods=n, freq='D').astype(str)
}

df = pd.DataFrame(data)

print("=" * 55)
print("  DU LIEU GIA LAP (thay = file that cua ban)")
print("=" * 55)
print(df.head())

print("\n" + "=" * 55)
print("  1. KHAM PHA DU LIEU DA DANG")
print("=" * 55)

print("\n[1.1] Thong ke co ban (mean, median, std, min, max):")
print(df.describe())

print("\n[1.2] So gia tri bi thieu (missing values):")
print(df.isnull().sum())

print(f"\n[1.3] So hang bi trung lap (duplicate): {df.duplicated().sum()}")

print("\n[1.4] Ve bieu do histogram, boxplot, violin plot...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df['gia_nha'].dropna(), bins=20, color='steelblue', edgecolor='white')
axes[0].set_title('Histogram - Gia nha')
axes[0].set_xlabel('Gia (trieu)')

axes[1].boxplot(df['dien_tich'].dropna())
axes[1].set_title('Boxplot - Dien tich')
axes[1].set_ylabel('m2')

axes[2].violinplot(df['so_phong'].dropna())
axes[2].set_title('Violin - So phong')

plt.tight_layout()
plt.savefig('bieu_do_phan_phoi.png', dpi=100)
plt.close()
print("  => Da luu: bieu_do_phan_phoi.png")

print("\n[1.5] Phan tich categorical - tinh_trang:")
print(df['tinh_trang'].value_counts(dropna=False))

print("\n[1.5] Phan tich categorical - vi_tri:")
print(df['vi_tri'].value_counts(dropna=False))

print("\n" + "=" * 55)
print("  2. XU LY DU LIEU BAN")
print("=" * 55)

print("\n[2.1] Dien missing values...")

df['gia_nha']   = df['gia_nha'].fillna(df['gia_nha'].median())
df['dien_tich'] = df['dien_tich'].fillna(df['dien_tich'].mean())
df['so_phong']  = df['so_phong'].fillna(df['so_phong'].mode()[0])
df['tinh_trang'] = df['tinh_trang'].fillna('khong ro')
df['vi_tri']    = df['vi_tri'].fillna('khong ro')
df['mo_ta']     = df['mo_ta'].fillna('')

print("  Missing sau khi dien:")
print(df.isnull().sum())

print("\n[2.2] Xu ly gia tri khong hop le...")

so_gia_am = (df['gia_nha'] < 0).sum()
print(f"  Gia am: {so_gia_am} dong -> thay = median")
df.loc[df['gia_nha'] < 0, 'gia_nha'] = df['gia_nha'].median()

so_phong_0 = (df['so_phong'] == 0).sum()
print(f"  So phong = 0: {so_phong_0} dong -> thay = 1")
df.loc[df['so_phong'] == 0, 'so_phong'] = 1

so_dt_0 = (df['dien_tich'] == 0).sum()
print(f"  Dien tich = 0: {so_dt_0} dong -> thay = mean")
df.loc[df['dien_tich'] == 0, 'dien_tich'] = df['dien_tich'].mean()

df['tinh_trang'] = df['tinh_trang'].str.lower().str.strip()
print("  Tinh_trang sau chuan hoa:")
print(df['tinh_trang'].value_counts())

so_dup_truoc = df.duplicated().sum()
df = df.drop_duplicates()
print(f"\n[2.3] Loai bo {so_dup_truoc} ban ghi trung lap")

print("\n" + "=" * 55)
print("  3. OUTLIERS & SKEW")
print("=" * 55)

def xu_ly_outlier_IQR(df, cot):
    Q1  = df[cot].quantile(0.25)
    Q3  = df[cot].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    so_outlier = ((df[cot] < lower) | (df[cot] > upper)).sum()
    print(f"  [{cot}] IQR=[{lower:.1f}, {upper:.1f}] | Outliers: {so_outlier} dong -> cap lai")
    df[cot] = df[cot].clip(lower, upper)
    return df

print("\n[3.1] Phat hien & xu ly outlier bang IQR (cap):")
for col in ['gia_nha', 'dien_tich', 'so_phong']:
    df = xu_ly_outlier_IQR(df, col)

print("\n[3.2] Kiem tra skew (do lech phan phoi):")
for col in ['gia_nha', 'dien_tich']:
    skew = df[col].skew()
    print(f"  {col}: skew = {skew:.2f}", end="")
    if abs(skew) > 1:
        print("  => Lech nhieu, nen log-transform")
        df[col + '_log'] = np.log1p(df[col])
    else:
        print("  => Phan phoi tuong doi binh thuong")

print("\n" + "=" * 55)
print("  4. CHUAN HOA SO & ENCODING")
print("=" * 55)

print("\n[4.1] Min-Max Scaling cho gia_nha, dien_tich:")
scaler = MinMaxScaler()
df[['gia_nha_scaled', 'dien_tich_scaled']] = scaler.fit_transform(
    df[['gia_nha', 'dien_tich']]
)
print(df[['gia_nha', 'gia_nha_scaled', 'dien_tich', 'dien_tich_scaled']].head(3))

print("\n[4.2] Z-score Scaling cho so_phong:")
df['so_phong_zscore'] = (df['so_phong'] - df['so_phong'].mean()) / df['so_phong'].std()
print(df[['so_phong', 'so_phong_zscore']].head(3))

print("\n[4.3] One-hot encoding cho vi_tri:")
df = pd.get_dummies(df, columns=['vi_tri'], prefix='khu_vuc')
print([c for c in df.columns if c.startswith('khu_vuc')])

print("\n[4.4] Label encoding cho tinh_trang:")
le = LabelEncoder()
df['tinh_trang_encoded'] = le.fit_transform(df['tinh_trang'])
print(dict(zip(le.classes_, le.transform(le.classes_))))

print("\n" + "=" * 55)
print("  5. PHAT HIEN DUPLICATE VAN BAN (mo ta nha)")
print("=" * 55)

df['mo_ta_chuan'] = df['mo_ta'].str.lower().str.strip()

dup_mask = df['mo_ta_chuan'].duplicated(keep=False) & (df['mo_ta_chuan'] != '')
dup_df   = df[dup_mask][['mo_ta', 'gia_nha', 'vi_tri' if 'vi_tri' in df.columns else 'mo_ta_chuan']]

print(f"\n[5.1] So ban ghi mo ta trung lap: {dup_mask.sum()}")
print("  Goi y: Nen review va merge cac ban ghi nay:")
print(df[dup_mask][['mo_ta']].drop_duplicates())

df = df.drop(columns=['mo_ta_chuan'])

print("\n" + "=" * 55)
print("  KET QUA CUOI CUNG")
print("=" * 55)
print(f"So hang:  {df.shape[0]}")
print(f"So cot:   {df.shape[1]}")
print(f"Missing:  {df.isnull().sum().sum()} o trong")
print("\nCac cot sau xu ly:")
print(df.dtypes)

df.to_csv('du_lieu_sach.csv', index=False)
print("\n=> Da luu du lieu sach ra: du_lieu_sach.csv")
print("=> Da luu bieu do ra:       bieu_do_phan_phoi.png")
print("\nHOAN THANH GIAI DOAN 1!")