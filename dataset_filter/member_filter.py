import pandas as pd

# Đọc file members.csv
members = pd.read_csv("kkbox_dataset/members.csv")

# Kiểm tra missing
print("Missing values:")
print(members.isnull().sum())

# Lọc member có đủ thông tin
filtered_members = members.dropna(
    subset=[
        "bd",      # tuổi
        "city"
    ]
)

# Lọc tuổi hợp lệ
filtered_members = filtered_members[
    (filtered_members["bd"] >= 1) &
    (filtered_members["bd"] <= 100)
]

# Nếu muốn chỉ giữ male/female
filtered_members = filtered_members[
    filtered_members["gender"].isin(["male", "female"])
]

# Reset index
filtered_members = filtered_members.reset_index(drop=True)

# Hiển thị
print("\nSố member còn lại:", len(filtered_members))
print(filtered_members.head())

# Lưu file mới
filtered_members.to_csv("dataset_filter/members_filtered.csv", index=False)

print("\nĐã lưu members_filtered.csv")