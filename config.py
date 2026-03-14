# ✅ REFACTORED: Configuration & Security
# Location: config.py
# This file fixes Issue 6.1 (Credentials in Code)

import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Get root directory
PROJECT_ROOT = Path(__file__).parent


# ============================================================================
# ✅ CRITICAL FIX: Load from environment, not hardcoded
# ============================================================================
@dataclass
class DatabaseConfig:
    \"\"\"Database configuration (from environment variables).\"\"\"
    
    # ✅ FIXED Issue 6.1: Moved from database_ops.py
    supabase_url: str = os.getenv(
        \"SUPABASE_URL\",
        \"https://orxtfxdernqmpkfmsijj.supabase.co\"  # Placeholder
    )
    supabase_key: str = os.getenv(\"SUPABASE_KEY\", \"\")  # Must be set in .env
    
    # These are fallback values (Streamlit secrets are preferred)
    db_host: str = os.getenv(\"DB_HOST\", \"localhost\")
    db_port: int = int(os.getenv(\"DB_PORT\", \"5432\"))
    db_name: str = os.getenv(\"DB_NAME\", \"fakenews\")
    db_user: str = os.getenv(\"DB_USER\", \"\")
    db_password: str = os.getenv(\"DB_PASSWORD\", \"\")
    
    def validate(self) -> bool:
        \"\"\"Check if critical credentials are set.\"\"\"
        warnings = []
        
        if not self.supabase_key or len(self.supabase_key) < 20:
            warnings.append(\"⚠️  SUPABASE_KEY not properly configured\")
        
        if not self.db_password:
            warnings.append(\"⚠️  DB_PASSWORD not set\")
        
        if warnings:
            logging.warning(f\"Configuration warnings: {', '.join(warnings)}\")
            return False
        
        return True


@dataclass
class EmailConfig:
    \"\"\"Email configuration for notifications and OTP.\"\"\"
    
    # ✅ FIXED Issue 6.1: Gmail app password (not account password!)
    # Never use your main Gmail password — use app password instead:
    # 1. Go to https://myaccount.google.com/apppasswords
    # 2. Generate new app password for \"Mail\"
    # 3. Copy the 16-character password to .env as GMAIL_APP_PASSWORD
    
    sender_email: str = os.getenv(\"GMAIL_EMAIL\", \"\")
    sender_password: str = os.getenv(\"GMAIL_APP_PASSWORD\", \"\")
    
    smtp_host: str = os.getenv(\"SMTP_HOST\", \"smtp.gmail.com\")
    smtp_port: int = int(os.getenv(\"SMTP_PORT\", \"587\"))
    
    def validate(self) -> bool:
        \"\"\"Check if email is configured.\"\"\"
        if not self.sender_email or not self.sender_password:
            logging.warning(\"Email configuration incomplete\")
            return False
        return True


@dataclass
class ModelConfig:
    \"\"\"AI model configuration.\"\"\"
    
    model_path: Path = Path(os.getenv(
        \"MODEL_PATH\",
        PROJECT_ROOT / \"best_model.pth\"
    ))
    artifacts_path: Path = Path(os.getenv(
        \"ARTIFACTS_PATH\",
        PROJECT_ROOT / \"artifacts.pkl\"
    ))
    
    bert_model_name: str = os.getenv(
        \"BERT_MODEL\",
        \"airesearch/wangchanberta-base-att-spm-uncased\"
    )
    
    device: str = os.getenv(\"DEVICE\", \"auto\")  # auto, cuda, cpu
    knn_neighbors: int = int(os.getenv(\"KNN_NEIGHBORS\", \"10\"))
    
    # Inference settings
    max_text_length: int = int(os.getenv(\"MAX_TEXT_LENGTH\", \"5000\"))
    max_tokens: int = int(os.getenv(\"MAX_TOKENS\", \"256\"))
    confidence_threshold: float = float(os.getenv(\"CONFIDENCE_THRESHOLD\", \"0.5\"))
    
    # Performance
    cache_model: bool = os.getenv(\"CACHE_MODEL\", \"true\").lower() == \"true\"
    use_gpu: bool = os.getenv(\"USE_GPU\", \"true\").lower() == \"true\"


@dataclass  
class AppConfig:
    \"\"\"Application configuration.\"\"\"
    
    environment: str = os.getenv(\"ENVIRONMENT\", \"development\")
    debug: bool = os.getenv(\"DEBUG\", \"False\").lower() == \"true\"
    log_level: str = os.getenv(\"LOG_LEVEL\", \"INFO\")
    
    # Feature flags
    enable_admin: bool = os.getenv(\"ENABLE_ADMIN\", \"true\").lower() == \"true\"
    enable_feedback: bool = os.getenv(\"ENABLE_FEEDBACK\", \"true\").lower() == \"true\"
    enable_url_scraping: bool = os.getenv(\"ENABLE_URL_SCRAPING\", \"true\").lower() == \"true\"
    
    # Rate limiting
    max_predictions_per_minute: int = int(os.getenv(\"MAX_PREDICTIONS_PER_MINUTE\", \"30\"))
    max_predictions_per_hour: int = int(os.getenv(\"MAX_PREDICTIONS_PER_HOUR\", \"500\"))
    
    # Session configuration
    session_timeout_minutes: int = int(os.getenv(\"SESSION_TIMEOUT_MINUTES\", \"60\"))
    
    @property
    def is_production(self) -> bool:
        \"\"\"Check if running in production.\"\"\"
        return self.environment.lower() == \"production\"
    
    @property
    def is_development(self) -> bool:
        \"\"\"Check if running in development.\"\"\"
        return self.environment.lower() == \"development\"


# ============================================================================
# COMBINED CONFIGURATION
# ============================================================================
class Config:
    \"\"\"Master configuration class.\"\"\"
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.email = EmailConfig()
        self.model = ModelConfig()
        self.app = AppConfig()
        self._validate()
    
    def _validate(self) -> None:
        \"\"\"Validate all configurations on startup.\"\"\"
        logging.info(f\"🔧 Initializing Config (Environment: {self.app.environment})\")
        
        # Database validation
        if not self.database.validate():
            if self.app.is_production:
                raise RuntimeError(\"Database configuration missing in production\")
        
        # Email validation
        if self.app.enable_feedback and not self.email.validate():
            logging.warning(\"Email not configured - feedback emails disabled\")
        
        # Model validation
        if not self.model.model_path.exists():
            logging.warning(f\"Model path not found: {self.model.model_path}\")
        
        if not self.model.artifacts_path.exists():
            logging.warning(f\"Artifacts path not found: {self.model.artifacts_path}\")
        
        logging.info(\"✅ Configuration validated\")
    
    def get_db_connection_params(self) -> dict:
        \"\"\"Get database connection parameters.\"\"\"
        return {
            'host': self.database.db_host,
            'port': self.database.db_port,
            'database': self.database.db_name,
            'user': self.database.db_user,
            'password': self.database.db_password,
        }
    
    def get_email_params(self) -> dict:
        \"\"\"Get email configuration parameters.\"\"\"
        return {
            'sender_email': self.email.sender_email,
            'sender_password': self.email.sender_password,
            'smtp_host': self.email.smtp_host,
            'smtp_port': self.email.smtp_port,
        }


# ============================================================================
# GLOBAL CONFIG INSTANCE
# ============================================================================
config = Config()

# Log configuration (for debugging)
if config.app.debug:
    logging.info(f\"Database: {config.database.db_host}:{config.database.db_port}/{config.database.db_name}\")
    logging.info(f\"Model: {config.model.model_path}\")
    logging.info(f\"Artifacts: {config.model.artifacts_path}\")


# ============================================================================
# USAGE IN YOUR APPLICATION
# ============================================================================
\"\"\"
In your database_ops.py, replace hardcoded credentials with:

    from config import config
    
    SUPABASE_URL = config.database.supabase_url
    SUPABASE_KEY = config.database.supabase_key
    SENDER_EMAIL = config.email.sender_email
    SENDER_PASSWORD = config.email.sender_password

In your ai_engine.py:

    from config import config
    
    MODEL_PATH = str(config.model.model_path)
    ARTIFACTS_PATH = str(config.model.artifacts_path)
    BERT_MODEL_NAME = config.model.bert_model_name

In your rate limiter:

    from config import config
    
    MAX_REQUESTS = config.app.max_predictions_per_minute
\"\"\"
