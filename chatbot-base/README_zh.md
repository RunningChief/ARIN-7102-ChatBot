# 医疗问答助手系统

## 项目概述
这是一个基于Web的医疗问答系统，为用户提供医疗咨询和问答服务。系统包含用户认证、密码管理、实时聊天等功能。

## 功能特点
- 用户注册与登录
- 密码修改功能
- 实时医疗问答聊天
- 响应式界面设计

## 技术架构
### 前端
- HTML5, CSS3, JavaScript
- 响应式设计 (Flexbox/Grid)
- 国际化UI组件

### 后端
- Python 3 + Flask框架
- RESTful API设计
- JWT认证

### 数据库
- SQLite (开发环境)
- SQLAlchemy ORM

## 快速开始
### 环境要求
- Python 3.8+
- pip包管理器

### 安装步骤


1. 安装依赖:
```bash
pip install -r medichat/backend/requirements.txt
```

2. 初始化数据库:
```bash
python medichat/backend/db_init.py
```

3.启动服务:
```bash
python medichat/backend/app.py
```

4. 访问应用:
```
http://localhost:5000
```


## 已知问题
- 未实现会话历史导航功能
- 聊天记录仅限当前会话
- 移动端UI需要优化
- 医疗知识库有限



## 贡献指南
请阅读[贡献指南](CONTRIBUTING.md)

## 开源协议
基于MIT协议开源，详见[LICENSE](LICENSE)
>
