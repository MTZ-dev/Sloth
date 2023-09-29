# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "SlothPy"
copyright = "2023, Mikołaj Żychowicz"
author = "Mikołaj Żychowicz"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
    "sphinx.ext.ifconfig",
    "matplotlib.sphinxext.plot_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "numpydoc",
    "sphinx_togglebutton",
    "jupyter_sphinx",
    "matplotlib.sphinxext.plot_directive",
    "sphinxcontrib.youtube",
]

templates_path = ["_templates"]
exclude_patterns = []
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
toc_object_entries_show_parents = "hide"
html_theme_options = {
    "show_nav_level": 6,
    "navigation_depth": 6,
    "logo": {
        "text": "SlothPy",
        "image_dark": "_static/slothpy.png",
        "alt_text": "SlothPy",
    },
    "show_toc_level": 1,
    "secondary_sidebar_items": [
        "edit-this-page",
        "sourcelink",
    ],  # Here add "page-toc" for table of contents on the right
    "navbar_align": "content",
    "navbar_center": ["version-switcher", "navbar-nav"],
    "icon_links": [
        {
            "name": "Twitter",
            "url": "https://twitter.com/PyData",
            "icon": "fa-brands fa-twitter",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/MTZ-dev/Sloth",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pydata-sphinx-theme",
            "icon": "fa-custom fa-pypi",
        },
        {
            "name": "MultiLumiMater",
            "url": "https://multilumimater.pl/",
            "icon": "",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
    ],
    "use_edit_page_button": True,
    "show_version_warning_banner": True,
}
html_context = {
    "github_user": "MTZ-dev",
    "github_repo": "Sloth",
    "github_version": "dev-doc",
    "doc_path": "doc",
}
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_sidebars = {
    "**": [
        "search-field.html",
        "sidebar-nav-bs.html",
        "globaltoc.html",
    ],
}

html_logo = "_static/slothpy.png"
html_favicon = "_static/slothpy.png"
