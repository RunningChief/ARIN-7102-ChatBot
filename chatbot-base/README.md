# Medical Q&A Assistant System

## Project Overview
A web-based medical Q&A system providing medical consultation services. Includes user authentication, password management, and real-time chat features.

![System Architecture](docs/architecture.png)
>

## Key Features
- User registration and login
- Password change functionality
- Real-time medical Q&A chat
- Responsive UI design

## Technical Stack
### Frontend
- HTML5, CSS3, JavaScript
- Responsive design with Flexbox/Grid
- Internationalized UI components

### Backend 
- Python 3 with Flask framework
- RESTful API design  
- JWT authentication

### Database
- SQLite for development
- SQLAlchemy ORM
>

## Getting Started
### Prerequisites
- Python 3.8+
- pip package manager

### Installation


1. Install dependencies:
```bash
pip install -r medichat/backend/requirements.txt
```

2. Initialize database:
```bash
python medichat/backend/db_init.py
```

3. Start development server:
```bash
python medichat/backend/app.py
```

4. Access the application:
```
http://localhost:5000
```
>

## Known Issues
- Session history navigation not implemented
- Chat history persistence limited to current session  
- Mobile UI needs refinement
- Limited medical knowledge base

