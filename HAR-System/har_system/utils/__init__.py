"""
HAR-System Utilities
====================
Utility functions for CLI and helpers
"""

from .cli import (
    parse_arguments,
    setup_output_directory,
    print_configuration,
    save_final_data,
    print_final_summary
)

# Re-export commonly used helpers to keep imports short in app modules.
__all__ = [
    'parse_arguments',
    'setup_output_directory',
    'print_configuration',
    'save_final_data',
    'print_final_summary'
]
