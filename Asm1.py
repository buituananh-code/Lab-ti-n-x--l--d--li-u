import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Đang đọc dữ liệu từ house_data.csv...\n")
df = pd.read_csv('house_data.csv')

print("--- 1. PHÂN TÍCH THỐNG KÊ ---")
print("- Thống kê cơ bản (mean, min, max...):\n", df.describe())
print("\n- Số lượng dữ liệu bị thiếu (missing values):\n", df.isnull().sum())
print("\n- Số dòng bị trùng lặp (duplicate):", df.duplicated().sum())

print("\n- Phân phối của cột 'vi_tri':\n", df['vi_tri'].value_counts())

print("\n--- Đang vẽ biểu đồ... (Hãy đóng cửa sổ biểu đồ để chạy tiếp) ---")

plt.figure(figsize=(6, 4))
df['gia_nha'].hist()
plt.title('Phân phối Giá nhà (Histogram)')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x=df['dien_tich'])
plt.title('Boxplot Diện tích')
plt.show()

plt.figure(figsize=(6, 4))
sns.violinplot(x=df['so_phong'])
plt.title('Violin plot Số phòng')
plt.show()

print("\n--- 2. BẮT ĐẦU XỬ LÝ DỮ LIỆU BẨN ---")

df['gia_nha'] = df['gia_nha'].fillna(df['gia_nha'].mean())

df['vi_tri'] = df['vi_tri'].fillna(df['vi_tri'].mode()[0])

df = df[df['gia_nha'] > 0]
df = df[df['so_phong'] > 0]

df['vi_tri'] = df['vi_tri'].str.lower()

df = df.drop_duplicates()

print("\nĐã dọn dẹp xong! Kích thước dữ liệu hiện tại (số dòng, số cột):", df.shape)
print("\n5 dòng dữ liệu đầu tiên sau khi làm sạch:")
print(df.head())