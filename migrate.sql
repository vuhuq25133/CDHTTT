-- 1. Tạo Database hỗ trợ đa ngôn ngữ (Tiếng Việt, Trung, Hàn, Nhật)
CREATE DATABASE IF NOT EXISTS music_recommendation_db
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE music_recommendation_db;

-- 2. Bảng danh mục Ngôn ngữ (Chứa ID và Tên đã map)
CREATE TABLE IF NOT EXISTS Languages (
    language_id DOUBLE PRIMARY KEY,
    language_name VARCHAR(100) DEFAULT 'Other Language'
) ENGINE=InnoDB;

-- 3. Bảng danh mục Thể loại (Chứa ID và Tên cho toàn bộ 68+ thể loại)
CREATE TABLE IF NOT EXISTS Genres (
    genre_id VARCHAR(50) PRIMARY KEY,
    genre_name VARCHAR(100) DEFAULT 'Unknown Genre'
) ENGINE=InnoDB;

-- 4. Bảng lưu trữ thông tin gốc của bài hát
CREATE TABLE IF NOT EXISTS Songs (
    song_id VARCHAR(50) NOT NULL,
    song_length INT DEFAULT 0,
    genre_ids VARCHAR(255) DEFAULT NULL,    -- Chuỗi gốc (ví dụ: "465|458")
    main_genre VARCHAR(50) DEFAULT NULL,    -- ID thể loại chính (Khóa ngoại)
    artist_name TEXT DEFAULT NULL,
    composer TEXT DEFAULT NULL,
    lyricist TEXT DEFAULT NULL,
    language DOUBLE DEFAULT NULL,           -- ID ngôn ngữ (Khóa ngoại)
    PRIMARY KEY (song_id),
    -- Thiết lập liên kết để đảm bảo tính toàn vẹn
    CONSTRAINT fk_song_language FOREIGN KEY (language) 
        REFERENCES Languages(language_id) ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT fk_song_genre FOREIGN KEY (main_genre) 
        REFERENCES Genres(genre_id) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB;

-- 5. Bảng lưu tên bài hát và thông tin chuẩn quốc tế
CREATE TABLE IF NOT EXISTS Song_Extra_Info (
    song_id VARCHAR(50) NOT NULL,
    name TEXT DEFAULT NULL,                 -- Tên thực tế của bài hát
    isrc VARCHAR(50) DEFAULT NULL,
    PRIMARY KEY (song_id),
    CONSTRAINT fk_extra_song_id FOREIGN KEY (song_id) 
        REFERENCES Songs(song_id) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;

-- 6. Bảng lưu trữ thông tin người dùng (Members)
CREATE TABLE IF NOT EXISTS Members (
    msno VARCHAR(50) NOT NULL,
    city INT DEFAULT NULL,
    bd INT DEFAULT NULL,                    -- Tuổi
    gender VARCHAR(20) DEFAULT NULL,
    registered_via INT DEFAULT NULL,
    registration_init_time INT DEFAULT NULL, -- Định dạng YYYYMMDD
    expiration_date INT DEFAULT NULL,        -- Định dạng YYYYMMDD
    PRIMARY KEY (msno)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS user_preferences (
    id       INT AUTO_INCREMENT PRIMARY KEY,
    msno     VARCHAR(255) NOT NULL,
    pref_type ENUM('genre', 'language') NOT NULL,
    ref_id   INT NOT NULL,
    UNIQUE KEY uq_user_pref (msno, pref_type, ref_id),
    FOREIGN KEY (msno) REFERENCES Members(msno) ON DELETE CASCADE
);

-- 7. Chỉ mục (Index) để tăng tốc độ gợi ý và tìm kiếm
CREATE INDEX idx_songs_artist ON Songs(artist_name(50));
CREATE INDEX idx_members_city_age ON Members(city, bd);
-- Note: Các cột Primary Key và Foreign Key đã tự động có Index.