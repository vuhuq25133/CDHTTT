import pandas as pd

# Đọc train.csv
train = pd.read_csv("kkbox_dataset/train.csv")

# Chỉ lấy cột cần thiết
train = train[["msno", "song_id", "target"]]

# Chỉ giữ bài user thích
train = train[train["target"] == 1]

# Reset index
train = train.reset_index(drop=True)

# Hiển thị
print(train.head())
print("Total interactions:", len(train))

# Lưu file
train.to_csv("train_filtered.csv", index=False)

print("Saved train_filtered.csv")