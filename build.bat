@echo off
set PATH=C:\msys64\mingw64\bin;%PATH%
C:\msys64\mingw64\bin\g++.exe -g "%~1" -o "%~dp1%~n1.exe" 2>&1
exit /b %errorlevel%
