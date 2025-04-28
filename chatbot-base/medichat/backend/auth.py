import datetime
import jwt
from functools import wraps
from flask import request, jsonify, current_app
import sqlite3
from werkzeug.security import check_password_hash

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(" ")[1]

        if not token:
            return jsonify({
                'status': 'error',
                'message': 'Token is missing!'
            }), 401

        try:
            data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['username']
        except jwt.ExpiredSignatureError:
            return jsonify({
                'status': 'error',
                'message': 'Token已过期!'
            }), 401
        except jwt.InvalidTokenError:
            return jsonify({
                'status': 'error',
                'message': '无效的Token!'
            }), 401
        except Exception as e:
            current_app.logger.error(f'Token验证错误: {str(e)}')
            return jsonify({
                'status': 'error',
                'message': '无法验证Token!'
            }), 401

        return f(current_user, *args, **kwargs)

    return decorated

def register():
    """用户注册函数"""
    auth = request.get_json()
    if not auth or not auth.get('username') or not auth.get('password'):
        return jsonify({
            'status': 'error',
            'message': 'Username and password are required'
        }), 400

    if len(auth['username']) < 4:
        return jsonify({
            'status': 'error',
            'message': 'The username needs to be at least 4 characters'
        }), 400

    if len(auth['password']) < 6:
        return jsonify({
            'status': 'error',
            'message': 'The password needs to be at least 6 characters long'
        }), 400

    conn = None
    try:
        conn = sqlite3.connect('medichat.db')
        c = conn.cursor()
        
        # 检查用户名是否已存在
        c.execute('SELECT username FROM users WHERE username = ?', (auth['username'],))
        if c.fetchone():
            return jsonify({
                'status': 'error',
                'message': 'The username already exists!'
            }), 400

        # 创建用户
        from werkzeug.security import generate_password_hash
        password_hash = generate_password_hash(auth['password'])
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                 (auth['username'], password_hash))
        conn.commit()

        return jsonify({
            'status': 'success',
            'message': 'Registered Successfully',
            'username': auth['username']
        }), 201

    except sqlite3.Error as e:
        conn.rollback()
        current_app.logger.error(f'数据库错误: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': 'Registration failed, please try again'
        }), 500
    except Exception as e:
        conn.rollback()
        current_app.logger.error(f'注册处理错误: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': 'Registration processing failed'
        }), 500
    finally:
        if conn:
            conn.close()

def login():
    auth = request.get_json()
    if not auth or not auth.get('username') or not auth.get('password'):
        return jsonify({
            'status': 'error',
            'message': 'Username and password are required'
        }), 401

    conn = None
    try:
        conn = sqlite3.connect('medichat.db')
        c = conn.cursor()

        c.execute('SELECT password_hash FROM users WHERE username = ?', (auth['username'],))
        user = c.fetchone()

        if not user:
            return jsonify({
                'status': 'error',
                'message': 'User name does not exist'
            }), 401

        if not check_password_hash(user[0], auth['password']):
            return jsonify({
                'status': 'error',
                'message': 'Incorrect password'
            }), 401

        token = jwt.encode({
            'username': auth['username'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, current_app.config['SECRET_KEY'])

        # 确保token是字符串格式
        if isinstance(token, bytes):
            token = token.decode('utf-8')

        return jsonify({
            'status': 'success',
            'token': token,
            'message': 'Login succeeded',
            'username': auth['username']
        })

    except sqlite3.Error as e:
        current_app.logger.error(f'数据库错误: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': 'Server Internal Error'
        }), 500
    except Exception as e:
        current_app.logger.error(f'登录处理错误: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': 'Login processing failed'
        }), 500
    finally:
        if conn:
            conn.close()
