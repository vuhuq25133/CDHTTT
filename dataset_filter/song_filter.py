import pandas as pd

# Đọc file songs.csv
songs = pd.read_csv("kkbox_dataset/songs.csv")

# Kiểm tra missing values
print("Missing values:")
print(songs.isnull().sum())

# Lọc bài hát có đủ:
# - genre_ids
# - language
filtered_songs = songs.dropna(
    subset=[
        "genre_ids",
        "language"
    ]
)

# Loại bỏ genre rỗng
filtered_songs = filtered_songs[
    filtered_songs["genre_ids"].astype(str).str.strip() != ""
]

# Lấy genre đầu tiên
filtered_songs["main_genre"] = filtered_songs["genre_ids"].apply(
    lambda x: str(x).split("|")[0]
)

# Chỉ giữ các cột cần thiết
filtered_songs = filtered_songs[
    [
        "song_id",
        "artist_name",
        "language",
        "main_genre",
        "composer",
        "lyricist"
    ]
]

# Reset index
filtered_songs = filtered_songs.reset_index(drop=True)

# Hiển thị kết quả
print("\nSố bài hát còn lại:", len(filtered_songs))
print(filtered_songs.head())

# Lưu file mới
filtered_songs.to_csv("dataset_filter/songs_filtered.csv", index=False)

print("\nĐã lưu: songs_filtered.csv")