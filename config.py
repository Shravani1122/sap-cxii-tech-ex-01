"""
Configuration for Product Similarity Search API

Set USE_ENHANCED=True to enable PCA dimensionality reduction and color extraction.
"""

import os

# Toggle between base and enhanced similarity search
USE_ENHANCED = os.getenv('USE_ENHANCED', 'false').lower() == 'true'

# PCA configuration (only used if USE_ENHANCED=True)
PCA_COMPONENTS = int(os.getenv('PCA_COMPONENTS', '50'))

# Cache configuration
CACHE_SIZE = int(os.getenv('CACHE_SIZE', '128'))

# Data path
DATA_PATH = os.getenv(
    'DATA_PATH',
    'data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson'
)

# API configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

def get_config_summary():
    """Get configuration summary for logging."""
    return {
        'use_enhanced': USE_ENHANCED,
        'pca_components': PCA_COMPONENTS if USE_ENHANCED else 'N/A',
        'cache_size': CACHE_SIZE,
        'data_path': DATA_PATH,
        'api_host': API_HOST,
        'api_port': API_PORT,
        'log_level': LOG_LEVEL
    }

# Made with Bob
