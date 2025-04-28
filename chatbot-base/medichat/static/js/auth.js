// 显示注册成功提示
function showRegisterSuccess() {
    const successAlert = document.getElementById('register-success');
    if (successAlert) {
        successAlert.style.display = 'block';
        setTimeout(() => {
            successAlert.style.display = 'none';
        }, 5000); // 5秒后自动隐藏
    }
}

// 登录表单处理
function setupLogin() {
    // 检查URL参数
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('registered')) {
        showRegisterSuccess();
    }

    const loginForm = document.getElementById('loginForm');
    const loginButton = document.querySelector('#loginForm button[type="submit"]');

    if (!loginForm) {
        console.error('错误: 未找到登录表单');
        return;
    }

    loginForm.addEventListener('submit', function(e) {
        console.log('表单提交事件触发');
        e.preventDefault();
        handleLogin(e);
    });

    if (loginButton) {
        loginButton.addEventListener('click', function(e) {
            console.log('登录按钮点击事件触发');
            e.preventDefault();
            const event = new Event('submit');
            loginForm.dispatchEvent(event);
        });
    }
}

// 登录处理函数
async function handleLogin(e) {
    console.log('开始处理登录');
    const form = e.target;
    const username = form.username.value.trim();
    const password = form.password.value.trim();

    if (!username || !password) {
        alert('Please enter your username and password');
        return;
    }

    try {
        console.log('发送登录请求...');
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password })
        });

        console.log('收到响应:', response.status);

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.message || '登录失败，请检查凭证');
        }

        const data = await response.json();
        console.log('登录响应数据:', data);

        if (data.token) {
            localStorage.setItem('authToken', data.token);
            localStorage.setItem('username', username);
            window.location.href = '/static/chat.html';
        } else {
            throw new Error('服务器未返回认证令牌');
        }
    } catch (error) {
        console.error('登录错误:', error);
        alert(error.message || '登录过程中发生错误');
    }
}

// 注册表单处理
function setupRegister() {
    const registerForm = document.getElementById('registerForm');
    const registerButton = document.querySelector('#registerForm button[type="submit"]');

    if (!registerForm) {
        console.error('错误: 未找到注册表单');
        return;
    }

    registerForm.addEventListener('submit', function(e) {
        console.log('注册表单提交事件触发');
        e.preventDefault();
        handleRegister(e);
    });

    if (registerButton) {
        registerButton.addEventListener('click', function(e) {
            console.log('注册按钮点击事件触发');
            e.preventDefault();
            const event = new Event('submit');
            registerForm.dispatchEvent(event);
        });
    }

    // 添加密码实时验证
    const password = document.getElementById('password');
    const confirmPassword = document.getElementById('confirmPassword');

    if (password && confirmPassword) {
        [password, confirmPassword].forEach(input => {
            input.addEventListener('input', validatePasswordMatch);
        });
    }
}

// 密码匹配验证
function validatePasswordMatch() {
    const password = document.getElementById('password');
    const confirmPassword = document.getElementById('confirmPassword');
    const errorElement = document.getElementById('password-error') ||
        document.createElement('div');

    if (!password || !confirmPassword) return;

    errorElement.id = 'password-error';
    errorElement.className = 'error-message';

    if (password.value && confirmPassword.value &&
        password.value !== confirmPassword.value) {
        errorElement.textContent = 'Entered passwords differ!';
        if (!document.getElementById('password-error')) {
            confirmPassword.parentNode.appendChild(errorElement);
        }
    } else {
        errorElement.textContent = '';
    }
}

// 注册处理函数
async function handleRegister(e) {
    console.log('开始处理注册');
    const form = e.target;
    const username = form.username.value.trim();
    const password = form.password.value.trim();
    const confirmPassword = form.confirmPassword.value.trim();

    // 输入验证
    if (!username || !password || !confirmPassword) {
        alert('请填写所有字段');
        return;
    }

    if (password !== confirmPassword) {
        alert('Entered passwords differ!');
        return;
    }

    if (username.length < 4) {
        alert('Username needs to be at least 4 characters');
        return;
    }

    if (password.length < 6) {
        alert('Password needs to be at least 6 characters');
        return;
    }

    try {
        console.log('发送注册请求...');
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password })
        });

        console.log('收到响应:', response.status);

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.message || '注册失败，请重试');
        }

        const data = await response.json();
        console.log('注册响应数据:', data);

        if (data.status === 'success') {
            // 直接跳转不显示提示
            window.location.href = '/static/login.html?registered=true';
        } else {
            throw new Error(data.message || '注册处理失败');
        }
    } catch (error) {
        console.error('注册错误:', error);
        alert(error.message || '注册过程中发生错误');
    }
}

// 页面加载初始化
if (document.readyState === 'complete') {
    setupLogin();
    setupRegister();
} else {
    window.addEventListener('load', function() {
        setupLogin();
        setupRegister();
    });
}