# pip install -e . --config-settings editable_mode=compat

from setuptools import setup, find_packages

setup(
    name="project",
    version="0.1",
    packages=find_packages()
)

