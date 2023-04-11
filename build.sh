#!/bin/bash
dotnet tool restore
dotnet fable --lang py 
echo "now run main.py with python"