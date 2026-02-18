import os
import yaml
import re
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def expand_env_vars(config):
    """
    Recursively expand environment variables in string values of the config dictionary.
    Supports ${VAR} and $VAR syntax.
    """
    env_pattern = re.compile(r"\$\{?(\w+)\}?")

    def replace_env(match):
        env_var = match.group(1)
        return os.getenv(env_var, match.group(0))  # Return original if not found

    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(i) for i in config]
    elif isinstance(config, str):
        return env_pattern.sub(replace_env, config)
    else:
        return config


def validate_config(config):
    """
    Validate that required configuration sections and keys exist.
    Prints warnings for missing optional keys.
    """
    required_sections = ['app', 'model']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    if 'name' not in config.get('model', {}):
        raise ValueError("Missing required config key: 'model.name'")

    if 'title' not in config.get('app', {}):
        raise ValueError("Missing required config key: 'app.title'")

    # Warn about optional but important keys
    qdrant = config.get('qdrant', {})
    if not qdrant.get('url') or qdrant.get('url', '').startswith('$'):
        print("Warning: 'qdrant.url' is not set or not resolved. Check your .env file.")
    if not qdrant.get('api_key') or qdrant.get('api_key', '').startswith('$'):
        print("Warning: 'qdrant.api_key' is not set or not resolved. Check your .env file.")


def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file and expand environment variables.
    """
    if not os.path.exists(config_path):
        # Try finding it in the parent directory if running from app/
        if os.path.exists(os.path.join("..", config_path)):
            config_path = os.path.join("..", config_path)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path}: {e}")

    # Expand environment variables
    config = expand_env_vars(config)

    # Validate
    validate_config(config)

    return config


# Expose a singleton config object
try:
    settings = load_config()
except (FileNotFoundError, ValueError) as e:
    print(f"FATAL: Configuration error: {e}")
    sys.exit(1)
