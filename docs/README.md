# Building the Documentation

The API documentation is hosted at: **https://wbsg-uni-mannheim.github.io/PyDI/**

Documentation is automatically built and deployed via GitHub Actions when changes are pushed to `main`.

## Building Locally

### Prerequisites

Install the documentation dependencies:

```bash
pip install sphinx sphinx-rtd-theme myst-parser
```

Or install the package with the docs extra:

```bash
pip install -e ".[docs]"
```

### Build Commands

From the `docs/` directory:

```bash
# Build HTML documentation
make html
```

This will:
1. Generate API documentation from source code using `sphinx-apidoc`
2. Build HTML output into `API_docs/`

Open `API_docs/index.html` in your browser to view the docs.

### Other Build Targets

```bash
# Show all available targets
make help

# Generate only the API stubs (without building HTML)
make apidoc
```

## Directory Structure

```
docs/
├── API_docs/        # Generated HTML output
├── source/          # Sphinx source files
│   ├── conf.py      # Sphinx configuration
│   ├── index.rst    # Documentation root
│   └── api/         # Auto-generated API reference (gitignored)
├── wiki/            # Module guides in Markdown
├── tutorial/        # Jupyter notebook tutorials
├── examples/        # Example scripts
├── Makefile         # Build commands
└── make.bat         # Windows build script
```

## Editing Documentation

- **API Reference**: Docstrings in the source code are automatically extracted. Edit the Python files in `PyDI/` to update API docs.
- **Module Guides**: Edit Markdown files in `wiki/`
- **Tutorials**: Edit Jupyter notebooks in `tutorial/`
- **Structure/Theme**: Edit `source/conf.py` for Sphinx settings
