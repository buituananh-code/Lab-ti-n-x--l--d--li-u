import pandas as pd
import numpy as np

data = {
    'ma_nha': ['N01', 'N02', 'N03', 'N04', 'N05', 'N06', 'N07', 'N08', 'N01', 'N09'],
    'gia_nha': [3500, 4200, -500, np.nan, 8000, 2100, 5500, 15000, 3500, 4800], # Lỗi: giá âm (-500), thiếu giá (NaN), có outlier (15000)
    'dien_tich': [50, 65, 45, 120, 80, 35, 70, 200, 50, 60],
    'so_phong': [2, 2, 0, 3, 3, 1, 2, 5, 2, 0], # Lỗi: số phòng = 0
    'vi_tri': ['Cầu Giấy', 'cầu giấy', 'Đống Đa', 'Thanh Xuân', np.nan, 'Hoàn Kiếm', 'Cầu Giấy', 'Tây Hồ', 'Cầu Giấy', 'đống đa'], # Lỗi: sai chính tả chữ hoa/thường, thiếu dữ liệu
    'tinh_trang': ['Mới', 'Cũ', 'Mới', 'Đang xây', 'Cũ', 'Mới', 'Cũ', 'Mới', 'Mới', 'Cũ'],
    'mo_ta': [
        'Nhà đẹp ở luôn, gần trung tâm', 
        'Cần bán gấp nhà ngõ rộng', 
        'Nhà mới xây thiết kế hiện đại', 
        'Biệt thự liền kề đang hoàn thiện', 
        'Nhà cũ tiện xây mới', 
        'Chung cư mini giá rẻ', 
        'Nhà lô góc hai mặt thoáng', 
        'Siêu biệt thự view hồ', 
        'Nhà đẹp ở luôn, gần trung tâm',
        'Nhà mới xây thiết kế rất hiện đại'
    ]
}

df = pd.DataFrame(data)

df.to_csv('house_data.csv', index=False, encoding='utf-8-sig')

print("Đã tạo thành công file 'house_data.csv' có chứa đầy đủ dữ liệu bẩn!")
print("Bạn hãy xem file này trong thư mục, sau đó chạy lại file Asm1.py để xem code dọn dẹp nó nhé.")