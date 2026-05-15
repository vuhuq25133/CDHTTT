import urllib.parse
import datetime
import os
import pickle
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, flash
from sqlalchemy import create_engine, text

# =========================================================
# FLASK CONFIG
# =========================================================

app = Flask(__name__)
app.secret_key = "music_recommendation_secret_2026"

# =========================================================
# DATABASE CONFIG
# =========================================================

USERNAME = "root"
PASSWORD = ""
HOST = "localhost"
PORT = "3306"
DB_NAME = "music_recommendation_db"

safe_password = urllib.parse.quote_plus(PASSWORD)
connection_url = f"mysql+pymysql://{USERNAME}:{safe_password}@{HOST}:{PORT}/{DB_NAME}?charset=utf8mb4"
engine = create_engine(connection_url, pool_size=10, max_overflow=20)

# =========================================================
# LOAD MODELS
# =========================================================

MODEL_DIR = "models/"

# -------------------------------------------------------
# [MODEL 1] Content-Based (TF-IDF + KNN)
# File: content_based_model.pkl
# Keys: tfidf, tfidf_matrix, knn_model, songs_df, song_idx
# -------------------------------------------------------
cb_model = None
try:
    with open("content_based_model.pkl", "rb") as f:
        cb_model = pickle.load(f)
    print("✅ Load Content-Based thành công!")
except Exception as e:
    print(f"❌ Lỗi Content-Based: {e}")

# -------------------------------------------------------
# [MODEL 2] Hybrid SVD (Matrix Factorization + User Taste)
# Files: svd_U.pkl, svd_sigma.pkl, svd_Vt.pkl,
#        svd_user_to_idx.pkl, svd_idx_to_song.pkl,
#        svd_user_history.pkl, user_profiles.pkl
# NOTE: song_details (index=song_id) cần load riêng
# -------------------------------------------------------
svd_model = {}
try:
    svd_model['U']        = joblib.load(os.path.join(MODEL_DIR, 'svd_U.pkl'))
    svd_model['sigma']    = joblib.load(os.path.join(MODEL_DIR, 'svd_sigma.pkl'))
    svd_model['Vt']       = joblib.load(os.path.join(MODEL_DIR, 'svd_Vt.pkl'))
    svd_model['u2idx']    = joblib.load(os.path.join(MODEL_DIR, 'svd_user_to_idx.pkl'))
    svd_model['idx2s']    = joblib.load(os.path.join(MODEL_DIR, 'svd_idx_to_song.pkl'))
    svd_model['history']  = joblib.load(os.path.join(MODEL_DIR, 'svd_user_history.pkl'))
    # user_profiles: DataFrame index=msno, columns=[top_genre, top_language]
    svd_model['profiles'] = joblib.load(os.path.join(MODEL_DIR, 'user_profiles.pkl'))
    print("✅ Load Hybrid SVD thành công!")
except Exception as e:
    print(f"❌ Lỗi SVD: {e}")

# -------------------------------------------------------
# [MODEL 3] Hybrid KNN (Item-Based + User Personal Info)
# File: models/knn_final_model.pkl
# Keys: model, song_to_idx, idx_to_song,
#       user_profiles, song_details, user_history
#
# NOTE: item_user_matrix, user_knn, X_personal,
#       user_to_idx_personal KHÔNG được lưu vào pkl,
#       nên cold-start bằng user_knn không khả dụng.
#       Thay vào đó dùng popular songs fallback.
# -------------------------------------------------------
knn_model = None
try:
    knn_model = joblib.load(os.path.join(MODEL_DIR, 'knn_final_model.pkl'))
    print("✅ Load Hybrid KNN thành công!")
    # Kiểm tra các key bắt buộc
    required_keys = ['model', 'song_to_idx', 'idx_to_song', 'user_profiles', 'song_details', 'user_history']
    missing = [k for k in required_keys if k not in knn_model]
    if missing:
        print(f"⚠️  KNN model thiếu key: {missing}")
        knn_model = None
except Exception as e:
    print(f"❌ Lỗi KNN: {e}")

# =========================================================
# RECOMMENDATION FUNCTIONS
# =========================================================

def get_content_based_recs(song_id, top_n=6):
    """
    Gợi ý bài hát tương tự dựa trên nội dung (TF-IDF + KNN cosine).
    Dùng cho trang chi tiết bài hát.
    Trả về list of dict: [{song_id, song_name, artist_name}, ...]
    """
    if not cb_model:
        return []
    if song_id not in cb_model['song_idx']:
        return []
    try:
        idx = cb_model['song_idx'][song_id]
        distances, indices = cb_model['knn_model'].kneighbors(
            cb_model['tfidf_matrix'][idx], n_neighbors=top_n + 1
        )
        rec_indices = indices.flatten()[1:]
        results = cb_model['songs_df'].iloc[rec_indices]
        return results[['song_id', 'song_name', 'artist_name']].to_dict('records')
    except Exception as e:
        print(f"Lỗi Content-Based recs: {e}")
        return []


def _svd_score_and_rank(preds, listened, user_pref_genre, user_pref_lang, top_n):
    """
    Hàm dùng chung: boost score theo genre/language rồi trả top_n song_id.
    """
    genre_boost = 0.5
    lang_boost  = 0.8
    max_pred    = preds.max() if preds.max() > 0 else 1.0

    scored = []
    for i, score in enumerate(preds):
        sid = svd_model['idx2s'][i]
        if sid in listened:
            continue
        final_score = float(score)
        if user_pref_genre is not None and knn_model:
            if sid in knn_model['song_details'].index:
                info = knn_model['song_details'].loc[sid]
                if str(info.get('primary_genre', '')) == str(user_pref_genre):
                    final_score += genre_boost * max_pred
                if info.get('language') == user_pref_lang:
                    final_score += lang_boost * max_pred
        scored.append((sid, final_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:top_n]]


def get_svd_recs(user_id, top_n=12, pref_genre=None, pref_language=None):
    """
    Gợi ý cá nhân hóa bằng SVD + User Taste Profile.

    - User cũ (có trong u2idx): dùng vector U đã train → dự đoán điểm chuẩn.
    - User mới (cold-start): tổng hợp vector giả từ trung bình các bài hát
      thuộc genre/language đã chọn lúc đăng ký, rồi chiếu lên không gian SVD.

    Trả về list of song_id.
    """
    if not svd_model:
        return []

    # ── Xác định genre/language sẽ dùng ───────────────────────────────
    user_pref_genre = None
    user_pref_lang  = None
    profiles = svd_model.get('profiles')
    if profiles is not None and user_id in profiles.index:
        user_pref_genre = profiles.loc[user_id, 'top_genre']
        user_pref_lang  = profiles.loc[user_id, 'top_language']
    else:
        user_pref_genre = pref_genre
        user_pref_lang  = pref_language

    listened = svd_model['history'].get(user_id, set())

    # ── CASE 1: User đã có trong ma trận SVD ──────────────────────────
    if user_id in svd_model.get('u2idx', {}):
        try:
            u_idx = svd_model['u2idx'][user_id]
            preds = np.dot(
                np.dot(svd_model['U'][u_idx, :], svd_model['sigma']),
                svd_model['Vt']
            )
            return _svd_score_and_rank(preds, listened, user_pref_genre, user_pref_lang, top_n)
        except Exception as e:
            print(f"Lỗi SVD recs (existing user): {e}")
            return []

    # ── CASE 2: Cold-start — user mới chưa có trong SVD ───────────────
    # Thử 3 tầng fallback để luôn trả về kết quả.
    if knn_model is None:
        print("SVD cold-start: thiếu song_details.")
        return []

    try:
        song_details = knn_model['song_details']
        Vt           = svd_model['Vt']                        # (k, n_songs)
        idx2s        = svd_model['idx2s']                     # int → song_id
        s2idx_svd    = {sid: i for i, sid in idx2s.items()}   # song_id → int

        def build_preds_from_indices(indices):
            """Tính pseudo-vector từ danh sách song matrix indices, nhân với Vt."""
            vecs = Vt[:, indices]           # (k, m)
            pseudo_u = vecs.mean(axis=1)   # (k,)
            return np.dot(pseudo_u, Vt)    # (n_songs,)

        # ── Tầng 1: match language (so sánh chuỗi thô, không cần genre) ──
        match_indices = []
        if user_pref_lang:
            for sid, row in song_details.iterrows():
                if str(row.get('language', '')).strip().lower() == str(user_pref_lang).strip().lower():
                    if sid in s2idx_svd:
                        match_indices.append(s2idx_svd[sid])

        if match_indices:
            print(f"SVD cold-start tầng 1 (language match): {len(match_indices)} bài seed.")
            preds = build_preds_from_indices(match_indices)
            return _svd_score_and_rank(preds, listened, user_pref_genre, user_pref_lang, top_n)

        # ── Tầng 2: dùng trung bình toàn bộ Vt (popular/central songs) ──
        # Không cần song_details — chỉ cần Vt có trong pkl.
        print("SVD cold-start tầng 2: không khớp language, dùng trung bình Vt toàn cục.")
        try:
            pseudo_u = Vt.mean(axis=1)          # (k,)
            preds    = np.dot(pseudo_u, Vt)     # (n_songs,)
            result   = _svd_score_and_rank(preds, listened, None, None, top_n)
            if result:
                return result
        except Exception as e2:
            print(f"SVD cold-start tầng 2 lỗi: {e2}")

        # ── Tầng 3: trả thẳng top song_details (luôn có kết quả) ────────
        print("SVD cold-start tầng 3: fallback top song_details.")
        fallback = []
        for sid in song_details.head(top_n * 2).index:
            if sid not in listened:
                row = song_details.loc[sid]
                fallback.append(sid)
            if len(fallback) >= top_n:
                break
        return fallback

    except Exception as e:
        print(f"Lỗi SVD cold-start: {e}")
        return []


def get_knn_recs(user_id, top_n=12, pref_genre=None, pref_language=None):
    """
    Gợi ý Hybrid KNN Item-Based + User Taste Profile.
    Dùng cho Dashboard (section thứ 2 hoặc bổ trợ cho SVD).
    pref_genre / pref_language: fallback từ DB preferences nếu user chưa có trong pkl.
    Trả về list of dict: [{song_id, song_name, artist_name}, ...]
    """
    if knn_model is None:
        return []

    song_to_idx   = knn_model['song_to_idx']
    idx_to_song   = knn_model['idx_to_song']
    user_history  = knn_model['user_history']
    song_details  = knn_model['song_details']
    user_profiles = knn_model['user_profiles']
    model         = knn_model['model']

    try:
        candidate_songs = {}

        if user_id in user_history and len(user_history[user_id]) > 0:
            listened_songs = list(user_history[user_id])
            # Giới hạn 50 bài để không quá chậm
            if len(listened_songs) > 50:
                listened_songs = listened_songs[:50]

            for song_id in listened_songs:
                if song_id not in song_to_idx:
                    continue
                s_idx = song_to_idx[song_id]
                if 'item_user_matrix' in knn_model:
                    song_vector = knn_model['item_user_matrix'][s_idx]
                    distances, indices = model.kneighbors(song_vector, n_neighbors=101)
                    for sim_idx in indices.flatten()[1:]:
                        sim_sid = idx_to_song[sim_idx]
                        if sim_sid not in user_history[user_id]:
                            candidate_songs[sim_sid] = candidate_songs.get(sim_sid, 0) + 1
                else:
                    pass
        else:
            # Cold start: lấy top bài hát phổ biến từ song_details
            popular = song_details.head(top_n * 2).index.tolist()
            for sid in popular:
                candidate_songs[sid] = candidate_songs.get(sid, 0) + 1

        if not candidate_songs:
            # Fallback: popular songs
            popular = song_details.head(top_n * 2).index.tolist()
            for sid in popular:
                candidate_songs[sid] = candidate_songs.get(sid, 0) + 1

        # Boost theo User Taste Profile
        user_pref_genre = None
        user_pref_lang  = None
        if user_profiles is not None and user_id in user_profiles.index:
            user_pref_genre = user_profiles.loc[user_id, 'top_genre']
            user_pref_lang  = user_profiles.loc[user_id, 'top_language']
        else:
            # Fallback: dùng preferences từ DB (user mới đăng ký)
            user_pref_genre = pref_genre
            user_pref_lang  = pref_language

        genre_boost = 2.0
        lang_boost  = 1.5

        scored = []
        for sid, base_score in candidate_songs.items():
            if sid not in song_details.index:
                continue
            info = song_details.loc[sid]
            final_score = base_score
            if user_pref_genre is not None:
                if str(info.get('primary_genre', '')) == str(user_pref_genre):
                    final_score += genre_boost
                if info.get('language') == user_pref_lang:
                    final_score += lang_boost
            scored.append((sid, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [s[0] for s in scored[:top_n]]

        # Trả về list dict có song_name và artist_name từ song_details
        results = []
        for sid in top_ids:
            if sid in song_details.index:
                row = song_details.loc[sid]
                results.append({
                    'song_id':     sid,
                    'song_name':   row.get('song_name', ''),
                    'artist_name': row.get('artist_name', ''),
                })
        return results

    except Exception as e:
        print(f"Lỗi KNN recs: {e}")
        return []


def fetch_songs_by_ids(conn, song_ids):
    """
    Truy vấn DB để lấy thông tin bài hát theo danh sách song_id.
    Trả về list of RowMapping, giữ nguyên thứ tự song_ids.
    """
    if not song_ids:
        return []
    try:
        rows = conn.execute(
            text("""
                SELECT s.song_id, e.name AS song_name, s.artist_name
                FROM Songs s
                JOIN Song_Extra_Info e ON s.song_id = e.song_id
                WHERE s.song_id IN :ids
            """),
            {"ids": tuple(song_ids)}
        ).fetchall()
        # Sắp xếp lại theo thứ tự gợi ý
        row_map = {r.song_id: r for r in rows}
        return [row_map[sid] for sid in song_ids if sid in row_map]
    except Exception as e:
        print(f"Lỗi fetch_songs_by_ids: {e}")
        return []


def songs_from_model_details(song_ids):
    """
    Lấy thông tin bài từ knn_model['song_details'] thay vì DB
    khi DB không có hoặc để fallback.
    Trả về list of dict.
    """
    if knn_model is None:
        return []
    song_details = knn_model['song_details']
    results = []
    for sid in song_ids:
        if sid in song_details.index:
            row = song_details.loc[sid]
            results.append({
                'song_id':     sid,
                'song_name':   row.get('song_name', ''),
                'artist_name': row.get('artist_name', ''),
            })
    return results


def get_user_preferences(user_id):
    """
    Lấy genre và language preferences từ bảng user_preferences trong DB.
    Dùng cho cold-start khi user mới chưa có trong pkl profiles.
    Trả về dict: {'genres': ['Pop', ...], 'language': 'Tiếng Việt'}
    """
    result = {'genres': [], 'language': None}
    try:
        with engine.connect() as conn:
            genres = conn.execute(
                text("""
                    SELECT g.genre_name
                    FROM user_preferences up
                    JOIN genres g ON up.ref_id = g.genre_id
                    WHERE up.msno = :m AND up.pref_type = 'genre'
                    ORDER BY up.id
                """),
                {"m": user_id}
            ).fetchall()
            result['genres'] = [r.genre_name for r in genres]

            lang = conn.execute(
                text("""
                    SELECT l.language_name
                    FROM user_preferences up
                    JOIN languages l ON up.ref_id = l.language_id
                    WHERE up.msno = :m AND up.pref_type = 'language'
                    LIMIT 1
                """),
                {"m": user_id}
            ).fetchone()
            if lang:
                result['language'] = lang.language_name
    except Exception as e:
        print(f"Lỗi get_user_preferences: {e}")
    return result


# =========================================================
# URL HELPER
# =========================================================

def song_url(song_id):
    """
    Tạo URL an toàn cho song_id có thể chứa ký tự đặc biệt như '/', '+', '='.
    Encode toàn bộ song_id bằng percent-encoding rồi ghép vào /song/.
    """
    encoded = urllib.parse.quote(song_id, safe="")
    return f"/song/{encoded}"

# Đăng ký để dùng được trong mọi template Jinja2
app.jinja_env.globals["song_url"] = song_url


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        msno = request.form.get("msno")
        try:
            with engine.connect() as conn:
                user = conn.execute(
                    text("SELECT * FROM Members WHERE msno = :m"),
                    {"m": msno}
                ).fetchone()
                if user:
                    session["user_id"] = str(user.msno)
                    flash("Đăng nhập thành công!", "success")
                    return redirect(url_for("dashboard"))
                else:
                    flash("User ID không tồn tại!", "danger")
        except Exception as e:
            flash(f"Lỗi database: {e}", "danger")
    return render_template("login.html")


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    # ── GET: load genres & languages từ DB để render form ──────────────
    if request.method == "GET":
        genres    = []
        languages = []
        try:
            with engine.connect() as conn:
                genres = [
                    r.genre_name for r in conn.execute(
                        text("SELECT genre_name FROM genres ORDER BY genre_name")
                    ).fetchall()
                ]
                languages = [
                    r.language_name for r in conn.execute(
                        text("SELECT language_name FROM languages ORDER BY language_name")
                    ).fetchall()
                ]
        except Exception as e:
            print(f"Lỗi load genres/languages: {e}")
            flash("Không thể tải danh sách thể loại/ngôn ngữ.", "warning")
        return render_template("register.html", genres=genres, languages=languages)

    # ── POST: xử lý đăng ký ────────────────────────────────────────────
    msno          = request.form.get("msno", "").strip()
    city          = request.form.get("city", 1)
    bd            = request.form.get("bd", 25)
    gender        = request.form.get("gender", "unknown")
    pref_genres   = request.form.getlist("pref_genres")   # list, tối đa 3
    pref_language = request.form.get("pref_language", "").strip()
    today         = int(datetime.datetime.now().strftime("%Y%m%d"))
    expire        = today + 10000

    if not msno:
        flash("User ID không được để trống!", "danger")
        return redirect(url_for("register"))

    try:
        with engine.begin() as conn:
            # Kiểm tra trùng
            if conn.execute(
                text("SELECT msno FROM Members WHERE msno = :m"), {"m": msno}
            ).fetchone():
                flash("User ID đã tồn tại!", "warning")
                return redirect(url_for("register"))

            # Insert member
            conn.execute(
                text("""
                    INSERT INTO Members
                        (msno, city, bd, gender, registered_via,
                         registration_init_time, expiration_date)
                    VALUES (:msno, :city, :bd, :gender, 7, :init, :exp)
                """),
                {"msno": msno, "city": city, "bd": bd,
                 "gender": gender, "init": today, "exp": expire}
            )

            # Lưu genre preferences (tối đa 3)
            for gname in pref_genres[:3]:
                row = conn.execute(
                    text("SELECT genre_id FROM genres WHERE genre_name = :g"),
                    {"g": gname}
                ).fetchone()
                if row:
                    conn.execute(
                        text("""
                            INSERT IGNORE INTO user_preferences
                                (msno, pref_type, ref_id)
                            VALUES (:msno, 'genre', :ref_id)
                        """),
                        {"msno": msno, "ref_id": row.genre_id}
                    )

            # Lưu language preference (1 ngôn ngữ)
            if pref_language:
                row = conn.execute(
                    text("SELECT language_id FROM languages WHERE language_name = :l"),
                    {"l": pref_language}
                ).fetchone()
                if row:
                    conn.execute(
                        text("""
                            INSERT IGNORE INTO user_preferences
                                (msno, pref_type, ref_id)
                            VALUES (:msno, 'language', :ref_id)
                        """),
                        {"msno": msno, "ref_id": row.language_id}
                    )

        flash("Đăng ký thành công!", "success")
        return redirect(url_for("login"))

    except Exception as e:
        flash(f"Lỗi: {e}", "danger")
        return redirect(url_for("register"))


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    try:
        with engine.connect() as conn:
            user = conn.execute(
                text("SELECT * FROM Members WHERE msno = :m"),
                {"m": user_id}
            ).fetchone()

        # Lấy preferences từ DB (dùng cho cold-start user mới)
        db_prefs     = get_user_preferences(user_id)
        pref_genre   = db_prefs['genres'][0] if db_prefs['genres'] else None
        pref_language = db_prefs['language']

        with engine.connect() as conn:
            # --- SVD: gợi ý cá nhân hóa (section chính) ---
            svd_ids   = get_svd_recs(user_id, top_n=12,
                                     pref_genre=pref_genre,
                                     pref_language=pref_language)
            svd_songs = fetch_songs_by_ids(conn, svd_ids)
            # Fallback: dùng song_details trong pkl nếu DB không trả về đủ
            if not svd_songs and svd_ids:
                svd_songs = songs_from_model_details(svd_ids)

            # --- KNN: "Người nghe tương tự cũng thích" (section phụ) ---
            knn_songs = get_knn_recs(user_id, top_n=8,
                                     pref_genre=pref_genre,
                                     pref_language=pref_language)
            if knn_songs and knn_songs[0].get('song_name'):
                pass  # Đã có thông tin từ song_details trong pkl
            else:
                knn_ids   = [s['song_id'] for s in knn_songs]
                knn_songs = fetch_songs_by_ids(conn, knn_ids)

        # --- User Taste Profile để hiển thị trên UI ---
        taste_profile = {}

        # Ưu tiên 1: pkl profiles (user có lịch sử nghe)
        if knn_model and user_id in knn_model['user_profiles'].index:
            tp = knn_model['user_profiles'].loc[user_id]
            taste_profile = {
                'top_genre':    tp.get('top_genre', 'N/A'),
                'top_language': tp.get('top_language', 'N/A'),
            }

        # Ưu tiên 2: DB preferences (user mới đăng ký)
        if not taste_profile and (db_prefs['genres'] or db_prefs['language']):
            taste_profile = {
                'top_genre':    db_prefs['genres'][0] if db_prefs['genres'] else 'N/A',
                'top_language': db_prefs['language'] or 'N/A',
            }

        return render_template(
            "dashboard.html",
            user=user,
            svd_songs=svd_songs,
            knn_songs=knn_songs,
            taste_profile=taste_profile,
        )

    except Exception as e:
        flash(f"Lỗi: {e}", "danger")
        return redirect(url_for("login"))


# =========================================================
# SONG DETAIL
# =========================================================

@app.route("/song/<path:song_id>")
def song_detail(song_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    # Decode percent-encoding (vd: %2F -> /, %2B -> +, %3D -> =)
    song_id = urllib.parse.unquote(song_id)
    try:
        with engine.connect() as conn:
            song = conn.execute(
                text("""
                    SELECT s.*, e.name AS song_name
                    FROM Songs s
                    JOIN Song_Extra_Info e ON s.song_id = e.song_id
                    WHERE s.song_id = :sid
                """),
                {"sid": song_id}
            ).fetchone()

        # Content-Based: bài hát tương tự về nội dung
        content_rec = get_content_based_recs(song_id, top_n=6)

        # KNN Hybrid: bài hát "người nghe tương tự cũng nghe"
        knn_rec = []
        if knn_model and song_id in knn_model['song_to_idx'] and 'item_user_matrix' in knn_model:
            try:
                s_idx = knn_model['song_to_idx'][song_id]
                distances, indices = knn_model['model'].kneighbors(
                    knn_model['item_user_matrix'][s_idx], n_neighbors=7
                )
                for sim_idx in indices.flatten()[1:]:
                    sid = knn_model['idx_to_song'][sim_idx]
                    if sid in knn_model['song_details'].index:
                        row = knn_model['song_details'].loc[sid]
                        knn_rec.append({
                            'song_id':     sid,
                            'song_name':   row.get('song_name', ''),
                            'artist_name': row.get('artist_name', ''),
                        })
            except Exception as e:
                print(f"Lỗi KNN detail: {e}")

        return render_template(
            "detail.html",
            song=song,
            content_rec=content_rec,
            knn_rec=knn_rec,
        )
    except Exception as e:
        print(f"Lỗi song_detail: {e}")
        return redirect(url_for("dashboard"))


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():
    session.clear()
    flash("Đã đăng xuất!", "success")
    return redirect(url_for("login"))


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)