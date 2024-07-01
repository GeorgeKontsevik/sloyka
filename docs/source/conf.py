# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('./sloyka'))

project = 'sloyka'
copyright = '2024, itmo_idu'
author = 'itmo_idu'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]

templates_path = ['_templates']
exclude_patterns = []


language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_show_sphinx = False

html_static_path = ['_static']
html_logo = '../logo/logo.png'
html_theme_options = {

}

html_css_files = ['_static/custom.css']

# html_js_files = [
#     'https://cdnjs.cloudflare.com/ajax/libs/medium-zoom/1.0.6/medium-zoom.min.js',
#     'https://p.trellocdn.com/embed.min.js',
#     'custom.js',
# ]
