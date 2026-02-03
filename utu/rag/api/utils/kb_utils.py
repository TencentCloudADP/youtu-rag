"""Utils related to knowledge base."""
import logging
import os
import re
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


def load_yaml_config(kb_name: str) -> dict:
    """Load YAML configuration for a knowledge base.
    
    Args:
        kb_name: Knowledge base name.
        
    Returns:
        Configuration dictionary.
    """
    try:
        # Get project root directory (assuming kb_utils.py is in utu/rag/api/utils/)
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent.parent
        config_dir = project_root / "configs" / "rag"
        
        # Try loading specific KB config first, fallback to default config
        kb_config_path = config_dir / f"{kb_name}.yaml"
        default_config_path = config_dir / "default.yaml"
        
        config_path = kb_config_path if kb_config_path.exists() else default_config_path
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found for KB '{kb_name}'")
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Resolve environment variables (e.g., ${UTU_LLM_MODEL} -> actual value)
        def resolve_env_var(value):
            if isinstance(value, str):
                pattern = re.compile(r'\$\{([^}]+)\}')
                matches = pattern.findall(value)
                for var_name in matches:
                    env_value = os.getenv(var_name, '')
                    value = value.replace(f'${{{var_name}}}', env_value)
                return value
            elif isinstance(value, dict):
                return {k: resolve_env_var(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_env_var(item) for item in value]
            return value
        
        config = resolve_env_var(config)
        logger.info(f"Loaded YAML config for KB '{kb_name}' from {config_path}")
        
        return config
    
    except Exception as e:
        logger.error(f"Error loading YAML config for KB '{kb_name}': {e}")
        return {}
