import pandas as pd
import os

def export_all_categories(csv_path, output_dir='processed_data'):
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Định nghĩa Mapping đầy đủ nhất cho KKBox (68+ genres)
    # Gồm các nhóm chính và các nhóm nhỏ (Sub-genres)
    full_genre_map = {
        '465': 'Pop', '458': 'Rock', '444': 'K-Pop', '921': 'Jazz', 
        '1259': 'Soundtrack', '1616': 'Electronic', '2022': 'Hip Hop', 
        '2122': 'Classical', '726': 'Folk', '1609': 'Instrumental',
        '850': 'Cantopop', '857': 'Mandopop', '940': 'Anime', '958': 'Children',
        '1152': 'R&B', '1572': 'Reggae', '1605': 'New Age', '1155': 'Soul',
        # Các mã khác thường là sub-genres hoặc mã vùng
    }

    # 2. Xử lý Genres
    print(f"📊 Đang trích xuất toàn bộ thể loại từ file...")
    genres = df[['main_genre']].drop_duplicates().dropna().copy()
    genres.columns = ['genre_id']
    
    # Chuyển về string để map
    genres['genre_id'] = genres['genre_id'].astype(str).str.replace('.0', '', regex=False)
    
    # Tự động điền: Nếu có trong map thì lấy tên, nếu không thì ghi 'Genre ' + ID
    genres['genre_name'] = genres['genre_id'].map(full_genre_map).fillna('Genre ' + genres['genre_id'])
    
    # Xuất ra CSV
    genres.to_csv(os.path.join(output_dir, 'genres_list.csv'), index=False)

    # 3. Xử lý Languages (Tương tự)
    lang_map = {3.0: 'Mandarin', 10.0: 'Japanese', 17.0: 'Cantonese', 24.0: 'Thai', 
                31.0: 'Korean', 38.0: 'Taiwanese', 45.0: 'Vietnamese', 52.0: 'English'}
    
    languages = df[['language']].drop_duplicates().dropna().copy()
    languages.columns = ['language_id']
    languages['language_name'] = languages['language_id'].map(lang_map).fillna('Other Language')
    languages.to_csv(os.path.join(output_dir, 'languages_list.csv'), index=False)

    print(f"✅ Đã xử lý xong! Tổng cộng tìm thấy {len(genres)} thể loại.")

if __name__ == "__main__":
    export_all_categories('processed_data/merged_data.csv')