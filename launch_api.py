"""
Script simple pour lancer l'API
"""
import subprocess
import os

# Changer vers le répertoire api
os.chdir('api')

# Lancer l'API
print("Lancement de l'API FastAPI...")
print("L'API sera accessible sur: http://localhost:8000")
print("Documentation: http://localhost:8000/docs")
print("Arrêtez avec Ctrl+C")

subprocess.run([
    '../.venv/Scripts/uvicorn.exe', 
    'main:app', 
    '--reload',
    '--host', '0.0.0.0',
    '--port', '8000'
])
