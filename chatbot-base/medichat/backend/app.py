import datetime
import sqlite3
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask import abort
import jwt
from auth import token_required, login as auth_login, register as auth_register
from werkzeug.security import generate_password_hash, check_password_hash

# 在文件顶部添加导入（第一行之后）
from flask import Flask, request, jsonify
from flask_cors import CORS
from auth import token_required, login as auth_login, register as auth_register
# 新增会话工具导入
from session_utils import init_session_db, get_user_sessions, create_user_session

# 初始化会话数据库（在创建app之后）
app = Flask(__name__)
CORS(app)
init_session_db()


# 初始化时创建测试用户
def init_db():
    conn = sqlite3.connect('medichat.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'doctor'
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # 添加测试用户(仅开发环境)
    test_hash = generate_password_hash('mypassword', method='pbkdf2:sha256')

    c.execute('''
        INSERT OR IGNORE INTO users (username, password_hash)
        VALUES (?, ?)
    ''', ('doctor1', test_hash))
    # 确保哈希格式正确
    c.execute('SELECT password_hash FROM users WHERE username = ?', ('doctor1',))
    db_hash = c.fetchone()[0]

    conn.commit()
    conn.close()


app = Flask(__name__, static_folder='../static')  # 设置静态文件目录
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
CORS(app)

# 初始化数据库
init_db()

# 模拟医疗知识库
medical_knowledge = {
    "头痛": "建议休息并服用适量止痛药，如持续不缓解请就医",
    "发烧": "建议测量体温，如超过38.5℃可服用退烧药",
    "咳嗽": "多喝水保持喉咙湿润，如持续超过一周请就医",
    "default": "请描述更详细的症状以便提供准确建议"
}


@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'login.html')


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


from flask import abort
from werkzeug.security import check_password_hash


@app.route('/api/login', methods=['POST'])
def login():
    return auth_login()


@app.route('/api/register', methods=['POST'])
def register():
    return auth_register()

@app.route('/api/change_password', methods=['POST'])
@token_required
def change_password(current_user):
    """修改用户密码"""
    data = request.get_json()
    
    if not data or not data.get('current_password') or not data.get('new_password'):
        return jsonify({
            'status': 'error',
            'message': '缺少必要参数'
        }), 400

    # 验证密码长度
    if len(data['new_password']) < 8:
        return jsonify({
            'status': 'error',
            'message': '新密码长度至少为8个字符'
        }), 400

    conn = None
    try:
        conn = sqlite3.connect('medichat.db')
        c = conn.cursor()
        
        # 获取用户当前密码哈希
        c.execute('SELECT password_hash FROM users WHERE username = ?', (current_user,))
        result = c.fetchone()

        
        if not result:
            return jsonify({
                'status': 'error',
                'message': '用户不存在'
            }), 404

        # 调试密码验证过程
        is_valid = check_password_hash(result[0], data['current_password'])
            
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': '当前密码不正确，请检查后重试',
                'code': 'INCORRECT_PASSWORD'  # 添加错误代码
            }), 401

        # 生成并更新新密码
        hashed_password = generate_password_hash(data['new_password'])
        
        c.execute('UPDATE users SET password_hash = ? WHERE username = ?', 
                 (hashed_password, current_user))
        conn.commit()

        return jsonify({
            'status': 'success',
            'message': '密码修改成功'
        })

    except sqlite3.Error as e:
        print(f'数据库错误: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': '数据库操作失败'
        }), 500
    except Exception as e:
        print(f'服务器错误: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': '服务器内部错误'
        }), 500
    finally:
        if conn:
            conn.close()


@app.route('/static-chat')
@token_required
def static_chat():
    return send_from_directory('..', 'static_index.html')


@app.route('/api/debug/db')
def debug_db():
    try:
        conn = sqlite3.connect('medichat.db')
        c = conn.cursor()
        c.execute('SELECT name FROM sqlite_master WHERE type="table"')
        tables = c.fetchall()
        users = c.execute('SELECT * FROM users').fetchall()
        conn.close()
        return jsonify({
            'tables': tables,
            'users': users,
            'db_file': os.path.abspath('medichat.db'),
            'exists': os.path.exists('medichat.db')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/debug/check_password', methods=['POST'])
def debug_check_password():
    data = request.json
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Missing username or password'}), 400

    try:
        conn = sqlite3.connect('medichat.db')
        c = conn.cursor()
        c.execute('SELECT password_hash FROM users WHERE username = ?', (data['username'],))
        result = c.fetchone()
        conn.close()

        if not result:
            return jsonify({'error': 'User not found'}), 404

        is_valid = check_password_hash(result[0], data['password'])
        return jsonify({
            'password_valid': is_valid,
            'hash_in_db': result[0]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/debug/reset_test_user', methods=['POST'])
def debug_reset_test_user():
    try:
        conn = sqlite3.connect('medichat.db')
        c = conn.cursor()
        new_hash = generate_password_hash('mypassword')
        c.execute('''
            UPDATE users SET password_hash = ?
            WHERE username = 'doctor1'
        ''', (new_hash,))
        conn.commit()
        conn.close()
        return jsonify({
            'message': 'Test user password reset',
            'new_hash': new_hash
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
@token_required
def chat(current_user):
    """处理用户医疗问题并返回模型响应
    1. 接收用户问题
    2. 记录问题到数据库
    3. 调用医疗问答模型(预留接口)
    4. 返回模型响应
    """
    data = request.get_json()
    if not data or not data.get('message'):
        return jsonify({
            'status': 'error',
            'message': '缺少消息内容'
        }), 400

    user_message = data['message']
    timestamp = datetime.datetime.now()

    # 记录用户问题到数据库
    try:
        conn = sqlite3.connect('medichat.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO chat_history (user_id, user_message, timestamp)
            VALUES ((SELECT id FROM users WHERE username = ?), ?, ?)
        ''', (current_user, user_message, timestamp))
        conn.commit()
    except Exception as e:
        print(f'保存聊天记录失败: {str(e)}')
        # 不中断流程，继续处理
    
    # 预留模型调用接口
    # TODO: 替换为实际模型调用代码
    # TODO: 在这里实例化一个对象，之后将user_message传入模型，将模型返回的字符串赋值给model_response
    model_response = user_message
        # "这是医疗模型的模拟响应。实际使用时将调用您训练的医疗问答模型。"
    
    # 更新数据库记录模型响应
    try:
        c.execute('''
            UPDATE chat_history SET bot_response = ?
            WHERE user_id = (SELECT id FROM users WHERE username = ?) 
            AND timestamp = ?
        ''', (model_response, current_user, timestamp))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f'更新模型响应失败: {str(e)}')
        # 不中断流程，继续处理

    return jsonify({
        'status': 'success',
        'response': model_response,
        'timestamp': timestamp.isoformat()
    })



@app.route('/api/user/sessions', methods=['POST'])
@token_required
def handle_create_session(current_user):
    """创建新会话"""
    try:
        user_id = get_user_id(current_user)
        session_id = create_user_session(user_id)
        return jsonify({'session_id': session_id}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/sessions/<session_id>', methods=['PUT'])
@token_required
def handle_update_session(current_user, session_id):
    """更新会话消息"""
    try:
        data = request.get_json()
        if not data or 'messages' not in data:
            return jsonify({'error': 'Missing messages data'}), 400
            
        user_id = get_user_id(current_user)
        
        # 连接到数据库
        conn = sqlite3.connect('medichat.db')
        c = conn.cursor()
        
        # 删除旧的会话消息
        c.execute('DELETE FROM session_messages WHERE session_id = ?', (session_id,))
        
        # 插入新的消息
        for idx, msg in enumerate(data['messages']):
            c.execute('''
                INSERT INTO session_messages (session_id, message_index, content, is_user)
                VALUES (?, ?, ?, ?)
            ''', (session_id, idx, msg['content'], int(msg['isUser'])))
        
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/sessions', methods=['GET'])
@token_required
def handle_get_sessions(current_user):
    """获取用户所有会话"""
    try:
        user_id = get_user_id(current_user)
        conn = sqlite3.connect('medichat.db')
        c = conn.cursor()
        
        # 获取会话列表
        c.execute('''
            SELECT id, title, created_at FROM user_sessions 
            WHERE user_id = ? ORDER BY created_at DESC
        ''', (user_id,))
        
        sessions = []
        for row in c.fetchall():
            session_id, title, created_at = row
            # 获取每个会话的最后一条消息作为预览
            c.execute('''
                SELECT content FROM session_messages 
                WHERE session_id = ? 
                ORDER BY message_index DESC 
                LIMIT 1
            ''', (session_id,))
            last_message = c.fetchone()
            preview = last_message[0] if last_message else "null session"
            
            sessions.append({
                'id': session_id,
                'title': title,
                'preview': preview,
                'created_at': created_at
            })
        
        conn.close()
        return jsonify({'sessions': sessions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/sessions/<session_id>/messages', methods=['GET'])
@token_required
def handle_get_session_messages(current_user, session_id):
    """获取特定会话的所有消息"""
    try:
        user_id = get_user_id(current_user)
        conn = sqlite3.connect('medichat.db')
        c = conn.cursor()
        
        # 验证会话属于当前用户
        c.execute('''
            SELECT 1 FROM user_sessions 
            WHERE id = ? AND user_id = ?
        ''', (session_id, user_id))
        
        if not c.fetchone():
            return jsonify({'error': 'Session not found'}), 404
        
        # 获取会话消息
        c.execute('''
            SELECT content, is_user FROM session_messages
            WHERE session_id = ?
            ORDER BY message_index ASC
        ''', (session_id,))
        
        messages = [{
            'content': row[0],
            'is_user': bool(row[1])
        } for row in c.fetchall()]
        
        conn.close()
        return jsonify({'messages': messages})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 辅助函数
def get_user_id(username):
    """根据用户名获取用户ID"""
    conn = sqlite3.connect('medichat.db')
    c = conn.cursor()
    c.execute('SELECT id FROM users WHERE username = ?', (username,))
    user_id = c.fetchone()[0]
    conn.close()
    return user_id

if __name__ == '__main__':
    app.run(debug=True)
