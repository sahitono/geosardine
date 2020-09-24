@echo off
del /f pyproject.toml
%PYTHON% -m pip install . -vv
Rem %PYTHON% setup.py install