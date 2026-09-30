"""Build Windows nativo e verificação do pacote que será publicado em Releases."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import windows_installer as app

DEST = ROOT / "dist/windows"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pe(exe):
    blob = exe.read_bytes()
    assert blob[:2] == b"MZ"
    offset = struct.unpack_from("<I", blob, 0x3C)[0]
    assert blob[offset:offset+4] == b"PE\x00\x00"
    assert struct.unpack_from("<H", blob, offset+4)[0] == 0x8664, "O alvo deve ser Windows x64"
    assert struct.unpack_from("<H", blob, offset+24+68)[0] == 2, "O aplicativo deve abrir sem console"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-runtime", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Windows":
        sys.exit("O .exe precisa ser compilado no Windows; use o workflow Windows no GitHub Actions.")
    DEST.mkdir(parents=True, exist_ok=True)
    exe = DEST / "MosquitoWingbeat-Windows.exe"
    audit_path = DEST / "windows_build_audit.json"
    if args.verify_runtime:
        runtime = json.loads((DEST / "windows_runtime_audit.json").read_text(encoding="utf-8"))
        assert runtime["success"] and runtime["frozen_executable"] and runtime["gui_created"]
        assert runtime["compile_succeeded"] and runtime["cli_version"] == "1.5.1"
        assert runtime["core_version"] == "4.6.0" and not runtime["flashed"]
        assert runtime["serial_module_version"] == "3.5" and runtime["result_parser_checked"]
        for name, digest in runtime["firmware_sha256"].items():
            assert sha(ROOT / "firmware/MosquitoSpecies" / name) == digest
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["native_windows_executable_smoke_test_passed"] = True
        audit["native_windows_firmware_compile_passed"] = True
        audit["runtime_audit"] = runtime
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
                        "--name", "MosquitoWingbeat-Windows", "--distpath", str(DEST),
                        "--workpath", str(ROOT / "tmp/pyinstaller-build"), "--specpath", str(ROOT / "tmp"),
                        "--add-data", str(ROOT / "firmware/MosquitoSpecies") + ":firmware/MosquitoSpecies",
                        str(ROOT / "windows_installer.py")], cwd=ROOT, check=True)
        verify_pe(exe)
        audit = dict(version=app.VERSION, exe_sha256=sha(exe), exe_bytes=exe.stat().st_size,
                     build_platform=platform.platform(), python=platform.python_version(),
                     windows_x64_gui_subsystem_verified=True, requires_python_on_user_machine=False,
                     requires_admin=False, signed=False, physical_board_tested=False, flashed=False,
                     firmware_sha256={name:sha(ROOT/"firmware/MosquitoSpecies"/name) for name in app.PAYLOAD_FILES})
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (DEST / "COMO-INSTALAR.txt").write_text(
            "MOSQUITO WINGBEAT - WINDOWS 10/11 64 BITS\n\n"
            "1. Abra MosquitoWingbeat-Windows.exe (dois cliques).\n"
            "2. Conecte sua Nano 33 BLE Sense / Sense Rev2 com cabo USB de dados.\n"
            "3. Clique em Buscar minha placa. A primeira preparacao usa internet.\n"
            "4. Clique em Instalar no Arduino e mantenha o cabo conectado.\n"
            "5. Aguarde os resultados na janela.\n\n"
            "Nao precisa instalar Python ou Arduino IDE. Reserve cerca de 1 GB livre.\n"
            "O modelo e experimental e pode errar. Nao houve validacao fisica no Arduino.\n"
            "O executavel ainda nao tem assinatura digital. Se o Windows pedir confirmacao,\n"
            "confira que baixou de github.com/Lciarallo/mosquito-wingbeat/releases.\n"
            "Ajuda: https://github.com/Lciarallo/mosquito-wingbeat/blob/main/WINDOWS.md\n",
            encoding="utf-8-sig")
    assert sha(exe) == json.loads(audit_path.read_text(encoding="utf-8"))["exe_sha256"]
    (DEST / "SHA256SUMS.txt").write_text("\n".join(sha(path) + "  " + path.name for path in sorted(DEST.iterdir())
                                               if path.is_file() and path.name != "SHA256SUMS.txt") + "\n", encoding="utf-8")
    print("Pacote Windows verificado:", exe)


if __name__ == "__main__":
    main()
