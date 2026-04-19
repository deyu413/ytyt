import glob
import subprocess
import os
import sys

def main():
    wavs = glob.glob("assets/music_generada/*.wav")
    if not wavs:
        print("No se encontraron archivos WAV en assets/music_generada/")
        sys.exit(1)
        
    # Paso 1: Extender a 3 horas
    # Opcional: Para evitar que muera la RAM si el entorno es pequeño, generaremos 1 hora primero 
    # (El usuario puede cambiarlo luego, pero 3 horas pesa 8GB en RAM)
    target_hours = "3"
    
    cmd1 = ["python", "src/audio/extender.py"] + wavs + ["--hours", target_hours, "--output", "master_3h_combinado"]
    print("==========================================")
    print(" PASO 1: Ensamblando piezas a 3 horas...")
    print("==========================================")
    result = subprocess.run(cmd1)
    if result.returncode != 0:
        print("Error en Paso 1")
        sys.exit(1)

    # Paso 2: Orquestador (Texturas + Ensamblado de Video)
    cmd2 = ["python", "src/orchestrator.py", "--full", "--audio", "assets/audio_raw/master_3h_combinado.wav", "--hours", target_hours, "--dry-run"]
    print("\n==========================================")
    print(" PASO 2: Post-Masterización y Renders de Video...")
    print("==========================================")
    subprocess.run(cmd2)

if __name__ == "__main__":
    main()
