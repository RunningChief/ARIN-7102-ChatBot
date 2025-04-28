class SessionManager {
    constructor(autoInit = true) {
        this.currentSessionId = null;
        if (autoInit) {
            this.initElements();
            this.loadSessions();
        }
    }

    initElements() {
        // 创建会话侧边栏
        this.sidebar = document.createElement('div');
        this.sidebar.className = 'session-sidebar';

        this.newSessionBtn = document.createElement('button');
        this.newSessionBtn.className = 'new-session-btn';
        this.newSessionBtn.innerHTML = '<span>+</span> New Session';

        this.sessionList = document.createElement('div');
        this.sessionList.className = 'session-list';

        this.sidebar.appendChild(this.newSessionBtn);
        this.sidebar.appendChild(this.sessionList);
        const mainContent = document.querySelector('.main-content');
        mainContent.insertBefore(this.sidebar, mainContent.firstChild);

        // 事件绑定
        this.newSessionBtn.addEventListener('click', () => this.createSession());
    }

    async loadSessions() {
        try {
            console.log('正在加载会话列表...');
            const response = await fetch('/api/user/sessions', {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('authToken')}`
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('获取到会话数据:', data);
            
            this.sessionList.innerHTML = '';
            
            if (!data.sessions || data.sessions.length === 0) {
                const emptyMsg = document.createElement('div');
                emptyMsg.className = 'empty-session';
                emptyMsg.textContent = '';
                this.sessionList.appendChild(emptyMsg);
                return;
            }
            
            data.sessions.forEach(session => {
                const sessionEl = document.createElement('div');
                sessionEl.className = 'session-item';
                sessionEl.innerHTML = `
                    <div class="session-title">${session.title || '未命名会话'}</div>
                    <div class="session-time">${new Date(session.created_at).toLocaleString()}</div>
                `;
                // 高亮当前会话
                if (this.currentSessionId === session.id) {
                    sessionEl.classList.add('active-session');
                }
                sessionEl.addEventListener('click', async () => {
                    console.log('切换到会话:', session.id);
                    await this.switchSession(session.id);
                });
                this.sessionList.appendChild(sessionEl);
            });
            
            console.log('会话列表加载完成');
        } catch (error) {
            console.error('加载会话失败:', error);
            this.sessionList.innerHTML = '<div class="error-message">加载会话失败，请刷新重试</div>';
        }
    }

    async saveCurrentSession() {
        if (!this.currentSessionId) return;
        
        const messages = Array.from(document.querySelectorAll('#chat-messages .message'))
            .map(msg => ({
                content: msg.querySelector('.message-content').textContent,
                isUser: msg.classList.contains('user-message')
            }));
            
        try {
            const response = await fetch(`/api/user/sessions/${this.currentSessionId}`, {
                method: 'PUT',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    messages,
                    title: messages.length > 0 ? 
                        messages[messages.length-1].content.substring(0, 30) : 
                        '新会话'
                })
            });
            return response.ok;
        } catch (error) {
            console.error('保存会话失败:', error);
            return false;
        }
    }

    async createSession() {
        try {
            console.log('开始创建新会话...');
            
            // 保存当前会话
            if (this.currentSessionId) {
                console.log('正在保存当前会话...');
                const saved = await this.saveCurrentSession();
                if (!saved) {
                    console.warn('当前会话保存失败');
                }
            }
            
            // 创建新会话
            console.log('正在创建新会话...');
            const response = await fetch('/api/user/sessions', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    title: 'Session-' + new Date().toLocaleString()
                })
            });
            
            if (!response.ok) {
                throw new Error(`创建会话失败: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('新会话创建成功:', data);
            this.currentSessionId = data.sessionId;
            
            // 清空聊天区域并添加欢迎消息
            const chatMessages = document.getElementById('chat-messages');
            chatMessages.innerHTML = '';
            const welcomeMsg = document.createElement('div');
            welcomeMsg.className = 'message bot-message';
            welcomeMsg.textContent = 'Hello, I am your medical assistant. May I ask if you have any symptoms';
            chatMessages.appendChild(welcomeMsg);
            
            // 确保输入框可见
            document.querySelector('.input-container').style.display = 'flex';
            
            // 刷新会话列表
            console.log('正在刷新会话列表...');
            await this.loadSessions();
            console.log('新会话流程完成');
            
        } catch (error) {
            console.error('创建会话失败:', error);
            alert('创建新会话失败，请重试');
        }
    }

    async switchSession(sessionId) {
        try {
            console.log(`正在切换到会话 ${sessionId}...`);
            
            // 保存当前会话
            if (this.currentSessionId && this.currentSessionId !== sessionId) {
                console.log('正在保存当前会话...');
                await this.saveCurrentSession();
            }
            
            // 清空聊天区域
            const chatMessages = document.getElementById('chat-messages');
            chatMessages.innerHTML = '';
            
            // 加载新会话消息
            console.log('正在加载会话消息...');
            const response = await fetch(`/api/user/sessions/${sessionId}/messages`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('authToken')}`
                }
            });
            
            if (!response.ok) {
                throw new Error(`加载会话失败: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('获取到会话消息:', data);
            
            if (data.messages && data.messages.length > 0) {
                data.messages.forEach(msg => {
                    this.appendMessage(msg.content, msg.is_user);
                });
            } else {
                // 空会话时显示欢迎消息
                console.log('当前会话为空，显示欢迎消息');
                const welcomeMsg = document.createElement('div');
                welcomeMsg.className = 'message bot-message';
                welcomeMsg.textContent = 'Hello, I am your medical assistant. May I ask if you have any symptoms';
                chatMessages.appendChild(welcomeMsg);
            }
            
            this.currentSessionId = sessionId;
            // 确保输入框可见
            document.querySelector('.input-container').style.display = 'flex';
            
            console.log('会话切换完成');
        } catch (error) {
            console.error('切换会话失败:', error);
            alert('加载会话失败，请重试');
        }
    }

    appendMessage(content, isUser) {
        const chatContainer = document.querySelector('.chat-container');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        messageDiv.appendChild(contentDiv);
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}


