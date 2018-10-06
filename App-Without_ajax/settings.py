"""
Application Settings
"""

# Celery configuration
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
UPLOAD_FOLDER = '../files'
ALLOWED_EXTENSIONS = {'csv'}
DEBUG = True
# CELERY_ALWAYS_EAGER = True
