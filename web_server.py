from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import re
import glob

app = Flask(__name__)
CORS(app)

BOARD_DIR = "static/board"


def load_all_posts():
    """게시판 글 전체 로드 (최신순 정렬)"""
    os.makedirs(BOARD_DIR, exist_ok=True)
    posts = []
    for path in glob.glob(os.path.join(BOARD_DIR, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                post = json.load(f)
                post["_file"] = os.path.basename(path)
                posts.append(post)
        except Exception:
            continue
    # 최신순 정렬
    posts.sort(key=lambda p: p.get("time", ""), reverse=True)
    return posts


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/posts")
def get_posts():
    """게시판 글 목록 반환"""
    return jsonify(load_all_posts())


@app.route("/api/posts/<post_id>", methods=["DELETE"])
def delete_post(post_id):
    """게시글 삭제 (찾아간 물건)"""
    if not re.match(r'^[a-zA-Z0-9_\-]+$', post_id):
        return jsonify({"error": "invalid post_id"}), 400
    json_path = os.path.join(BOARD_DIR, f"{post_id}.json")
    img_path = os.path.join(BOARD_DIR, f"{post_id}.jpg")
    if os.path.exists(json_path):
        os.remove(json_path)
    if os.path.exists(img_path):
        os.remove(img_path)
    return jsonify({"deleted": post_id})


@app.route("/api/status")
def get_status():
    posts = load_all_posts()
    return jsonify({
        "server": "running",
        "total_posts": len(posts),
        "warning": sum(1 for p in posts if p.get("status") == "WARNING"),
        "danger": sum(1 for p in posts if p.get("status") == "DANGER"),
    })


@app.route("/board/<path:filename>")
def board_files(filename):
    return send_from_directory(BOARD_DIR, filename)


if __name__ == "__main__":
    os.makedirs(BOARD_DIR, exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    print("=" * 50)
    print("  AI 분실물 게시판 서버 시작")
    print("  http://localhost:5000")
    print("=" * 50)
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, port=5000, use_reloader=False)