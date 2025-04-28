import sqlite3
from werkzeug.security import generate_password_hash

def init_db():
    conn = sqlite3.connect('medichat.db')
    c = conn.cursor()

    # 1. 备份现有数据
    try:
        c.execute('SELECT * FROM users')
        users = c.fetchall()
        c.execute('SELECT * FROM chat_history')
        messages = c.fetchall()
    except:
        users = []
        messages = []

    # 2. 删除旧表
    c.execute('DROP TABLE IF EXISTS chat_history')
    c.execute('DROP TABLE IF EXISTS sessions')
    c.execute('DROP TABLE IF EXISTS users')

    # 3. 创建新表结构
    c.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'doctor'
        )
    ''')

    c.execute('''
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title VARCHAR(50) NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    c.execute('''
        CREATE TABLE chat_history (
            id INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (id),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # 4. 恢复用户数据
    for user in users:
        c.execute('''
            INSERT INTO users (id, username, password_hash, role)
            VALUES (?, ?, ?, ?)
        ''', user)

    # 5. 创建默认会话并迁移消息
    if users:
        c.execute('''
            INSERT INTO sessions (id, user_id, title)
            VALUES (1, (SELECT id FROM users WHERE username = 'doctor1'), '默认会话')
        ''')

        for msg in messages:
            if len(msg) >= 5:  # 检查消息格式
                c.execute('''
                    INSERT INTO chat_history (id, session_id, user_id, user_message, bot_response, timestamp)
                    VALUES (?, 1, ?, ?, ?, ?)
                ''', (msg[0], msg[1], msg[2], msg[3], msg[4]))

    # 6. 添加测试用户(如果不存在)
    try:
        c.execute('''
            INSERT OR IGNORE INTO users (username, password_hash)
            VALUES ('doctor1', ?)
        ''', (generate_password_hash('mypassword'),))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()

if __name__ == '__main__':
    init_db()
