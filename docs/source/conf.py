from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

project = 'PyDI'
copyright = '2025, Peeters, Steiner, Bizer'
author = 'Peeters, Steiner, Bizer'
release = '2025'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Mock imports that may not be available during doc build
autodoc_mock_imports = [
    'pandas',
    'numpy',
    'pint',
    'pycountry',
    'phonenumbers',
    'email_validator',
    'textdistance',
    'langchain_core',
    'langchain_openai',
    'sklearn',
    'scipy',
    'networkx',
    'stdnum',
    'babel',
]

templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True
autodoc_typehints = 'description'
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Fallback to alabaster when the RTD theme is unavailable locally.
try:
    import sphinx_rtd_theme

    html_theme = 'sphinx_rtd_theme'
except ModuleNotFoundError:
    html_theme = 'alabaster'

html_static_path = ['_static']
