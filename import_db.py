import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import os

# 1. Cấu hình Database
user = "root"
password = "" 
host = "localhost"
port = "3306"
db_name = "music_recommendation_db"

safe_password = urllib.parse.quote_plus(password)
DB_URL = f"mysql+pymysql://{user}:{safe_password}@{host}:{port}/{db_name}"

# 2. Mapping dữ liệu (68+ thể loại và ngôn ngữ)
GENRE_MAP = {
    '465': 'Pop', '458': 'Rock', '444': 'K-Pop', '921': 'Jazz', 
    '1259': 'Soundtrack', '1616': 'Electronic', '2022': 'Hip Hop', 
    '2122': 'Classical', '726': 'Folk', '1609': 'Instrumental',
    '850': 'Cantopop', '857': 'Mandopop', '940': 'Anime', '958': 'Children'
}

LANG_MAP = {
    3.0: 'Mandarin', 10.0: 'Japanese', 17.0: 'Cantonese', 24.0: 'Thai', 
    31.0: 'Korean', 38.0: 'Taiwanese', 45.0: 'Vietnamese', 52.0: 'English',
    -1.0: 'Unknown'
}

def import_data(file_path):
    try:
        engine = create_engine(DB_URL)
        print("🚀 Đang đọc file merged_data.csv...")
        df = pd.read_csv(file_path)

        # --- BƯỚC 1: XỬ LÝ VÀ ĐẨY BẢNG LANGUAGES ---
        print("📊 Đang đẩy dữ liệu vào bảng Languages...")
        languages = df[['language']].drop_duplicates().dropna().copy()
        languages.columns = ['language_id']
        languages['language_name'] = languages['language_id'].map(LANG_MAP).fillna('Other Language')
        languages.to_sql('Languages', con=engine, if_exists='append', index=False)

        # --- BƯỚC 2: XỬ LÝ VÀ ĐẨY BẢNG GENRES ---
        print("📊 Đang đẩy dữ liệu vào bảng Genres...")
        genres = df[['main_genre']].drop_duplicates().dropna().copy()
        genres.columns = ['genre_id']
        # Chuẩn hóa ID về string để khớp với mapping
        genres['genre_id'] = genres['genre_id'].astype(str).str.replace('.0', '', regex=False)
        genres['genre_name'] = genres['genre_id'].map(GENRE_MAP).fillna('Genre ' + genres['genre_id'])
        genres.to_sql('Genres', con=engine, if_exists='append', index=False)

        # --- BƯỚC 3: ĐẨY BẢNG SONGS (Phụ thuộc vào 2 bảng trên) ---
        print("🎵 Đang đẩy dữ liệu vào bảng Songs...")
        songs_df = df[['song_id', 'artist_name', 'composer', 'lyricist', 'language', 'main_genre']].copy()
        songs_df['song_length'] = 0 
        songs_df['genre_ids'] = df['main_genre'] # Lưu ID gốc vào genre_ids
        songs_df = songs_df.drop_duplicates(subset=['song_id'])
        songs_df.to_sql('Songs', con=engine, if_exists='append', index=False)

        # --- BƯỚC 4: ĐẨY BẢNG SONG_EXTRA_INFO ---
        print("📝 Đang đẩy dữ liệu vào bảng Song_Extra_Info...")
        extra_df = df[['song_id', 'song_name']].copy()
        extra_df.rename(columns={'song_name': 'name'}, inplace=True)
        extra_df['isrc'] = None 
        extra_df = extra_df.drop_duplicates(subset=['song_id'])
        extra_df.to_sql('Song_Extra_Info', con=engine, if_exists='append', index=False)

        # --- BƯỚC 5: ĐẨY BẢNG MEMBERS ---
        print("👥 Đang đẩy dữ liệu vào bảng Members...")
        members_df = df[['msno', 'city', 'bd', 'gender', 'registered_via', 'registration_init_time', 'expiration_date']].copy()
        members_df = members_df.drop_duplicates(subset=['msno'])
        members_df.to_sql('Members', con=engine, if_exists='append', index=False)

        print("✅ TẤT CẢ QUY TRÌNH HOÀN TẤT THÀNH CÔNG!")

    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    csv_file = 'processed_data/merged_data.csv'
    if os.path.exists(csv_file):
        import_data(csv_file)
    else:
        print("❌ Không tìm thấy file merged_data.csv!")