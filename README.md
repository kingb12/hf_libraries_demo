# NLP 244 Section: Huggingface Library Demos

This repo collects code examples and demos used for demonstrating different aspects of the Huggingface library set
code 

### Preamble: Python packaging

Not required for anything in this course, but I often personally find it useful to organize my work as a Python package. 
This allows others to **import** elements from your work without using or extracting individual files from your repo. 
This also allows a simpler workflow for importing your code into a [Google Colab notebook](https://colab.research.google.com/)

This is accomplished in three main steps:

1. Create a `pyproject.toml` file [like this](./pyproject.toml). For typical use-cases, you can copy the one linked 
without edits. 
2. Create a `setup.cfg` file [like this](setup.cfg). You'll need to edit everything under `[metadata]`, 
`install_requires` for your dependencies, and possibly other attributes as you customize your package organization and 
contents.
3. Add all your code under a sources directory linked from `setup.cfg`. In this case, I have everything under 
`src/hf_libraries_demo` since my `package_dir` includes `=src`. You'll want to rename appropriately.

Given this, you can install an editable version of your package with `pip install -e .` from its root directory, and
import functions you've defined in different modules in `src`. You can also install it from the git link directly.
See an example of doing this [in Colab here](./ImportingAGithubPyPackage.ipynb).

## Using Huggingface Datasets
- Loading a dataset from Huggingface
