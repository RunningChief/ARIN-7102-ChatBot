import sqlite3
from datetime import datetime

def init_session_db():
    """独立初始化会话表，不影响原有表"""
    conn = sqlite3.connect('medichat.db')
    c = conn.cursor()

    # 创建独立的会话表（不修改原有表）
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # 创建会话消息表
    c.execute('''
        CREATE TABLE IF NOT EXISTS session_messages (
            id INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL,
            message_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            is_user INTEGER NOT NULL,
            FOREIGN KEY (session_id) REFERENCES user_sessions (id)
        )
    ''')

    # 为现有用户创建默认会话
    c.execute('''
        INSERT OR IGNORE INTO user_sessions (id, user_id, title)
        SELECT 1, id, '默认会话' FROM users WHERE username = 'doctor1'
    ''')

    conn.commit()
    conn.close()

def get_user_sessions(user_id):
    """获取用户会话列表，包含最后一条消息作为摘要"""
    conn = sqlite3.connect('medichat.db')
    c = conn.cursor()
    c.execute('''
        SELECT 
            us.id, 
            us.title, 
            us.created_at,
            (SELECT sm.content 
             FROM session_messages sm 
             WHERE sm.session_id = us.id 
             ORDER BY sm.message_index DESC 
             LIMIT 1) as last_message
        FROM user_sessions us
        WHERE us.user_id = ?
        ORDER BY us.created_at DESC
    ''', (user_id,))
    
    sessions = []
    for row in c.fetchall():
        session = {
            'id': row[0],
            'created_at': row[2]
        }
        # 如果有最后一条消息，使用前20个字符作为标题
        if row[3]:
            session['title'] = row[3][:20] + ('...' if len(row[3]) > 20 else '')
        else:
            # 没有消息则使用原标题
            session['title'] = row[1] if row[1] else '空会话'
            
        sessions.append(session)
    
    conn.close()
    return sessions

def create_user_session(user_id, title=None):
    """创建新会话"""
    title = title or f"Session {datetime.now().strftime('%m-%d %H:%M')}"
    conn = sqlite3.connect('medichat.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO user_sessions (user_id, title)
        VALUES (?, ?)
    ''', (user_id, title))
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id
