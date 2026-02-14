import os
import sys
import platform
import subprocess
import shutil
import time
import ssl
import tempfile
from pathlib import Path
from datetime import datetime

class Diagnostics:

    FILENAME = 'report.txt'
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        if os.path.exists(self.FILENAME):
            os.remove(self.FILENAME)

    def log(self, message):
        print(message)
        with open(self.FILENAME, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

    def start(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log(f"Démarrage des diagnostics à {now}\n")

    def end(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log(f"\n\nDiagnostics complétés à {now}\n")
        print("\nVeuillez m'envoyer ces diagnostics à ed@edwarddonner.com")
        print(f"Soit copiez et collez la sortie ci-dessus dans un e-mail, soit joignez le fichier {self.FILENAME} qui a été créé dans ce répertoire.")
    

    def _log_error(self, message):
        self.log(f"ERREUR : {message}")
        self.errors.append(message)

    def _log_warning(self, message):
        self.log(f"AVERTISSEMENT : {message}")
        self.warnings.append(message)

    def run(self):
        self.start()
        self._step1_system_info()
        self._step2_check_files()
        self._step3_git_repo()
