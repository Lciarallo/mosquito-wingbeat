#!/usr/bin/env python3
"""Instalador portátil do Arduino; usa somente a biblioteca padrão do Python."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
CLI_VERSION = "1.5.1"
CORE_ID = "arduino:mbed_nano"
CORE_VERSION = "4.6.0"
FQBN = CORE_ID + ":nano33ble"
# SHA-256 publicados na release oficial v1.5.1; nunca executar um download sem conferir.
# https://github.com/arduino/arduino-cli/releases/tag/v1.5.1
CLI_SHA256 = {
    "Linux_32bit.tar.gz": "85ed48978e7553b16f187971f8202d380421b352696ab28327a36cc6d8f11c6a",
    "Linux_64bit.tar.gz": "28a8e119c498a25607821c36cb2dc49e8463941b261a0d99091baa7bc692dd2b",
    "Linux_ARM64.tar.gz": "1e69e077479f300614d4551334e0a33f08ee40b04315d83b8e7e0e94f0d0ee62",
    "Linux_ARMv6.tar.gz": "168aa0c632d7079fea0ccd30b3f1e928e89e2f59a339404f1d2f4a07ed6cc566",
    "Linux_ARMv7.tar.gz": "890af36e9873606e4dfa743534846186621b9f3a339175ea3ec481adecf07143",
    "macOS_64bit.tar.gz": "c982e940027996bea9901050e95fae99c59c1dcfee54beedecaf28141e7bf2e7",
    "macOS_ARM64.tar.gz": "cb952e8c1621c95ef5f1d17831c945e3d0ec5973f89c557a7ec8feb9c4f7d4c9",
    "Windows_32bit.zip": "885e491c7c7fb8b396151c09daa5c4c56d8b60697d172a5cfe72c939eed50fe3",
    "Windows_64bit.zip": "fabe42e0eb04d00e776a66178299ff95a46c623dbc260f997e58fd514853dd40",
}


class InstallError(Exception):
    pass


def say(message):
    print(message, flush=True)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_asset(system=None, machine=None):
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if system == "Windows":
        # A release 1.5.1 não possui Windows ARM nativo; use a IDE nesse caso.
        arch = "64bit" if machine in ("amd64", "x86_64") else "32bit" if machine in ("x86", "i386", "i686") else None
        suffix = f"Windows_{arch}.zip" if arch else None
    elif system == "Darwin":
        arch = "ARM64" if machine in ("arm64", "aarch64") else "64bit" if machine in ("amd64", "x86_64") else None
        suffix = f"macOS_{arch}.tar.gz" if arch else None
    elif system == "Linux":
        arch = {"x86_64": "64bit", "amd64": "64bit", "i386": "32bit", "i686": "32bit",
                "aarch64": "ARM64", "arm64": "ARM64", "armv7l": "ARMv7", "armv6l": "ARMv6"}.get(machine)
        suffix = f"Linux_{arch}.tar.gz" if arch else None
    else:
        suffix = None
    if suffix not in CLI_SHA256:
        raise InstallError(f"Sem download testado da CLI para {system}/{machine}. Use a Arduino IDE ou --cli com a CLI {CLI_VERSION}.")
    return f"arduino-cli_{CLI_VERSION}_{suffix}", CLI_SHA256[suffix]


def download_cli(tools):
    asset, expected_hash = release_asset()
    archive = tools / "downloads" / asset
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or file_hash(archive) != expected_hash:
        url = f"https://github.com/arduino/arduino-cli/releases/download/v{CLI_VERSION}/{asset}"
        say(f"Baixando Arduino CLI {CLI_VERSION} de uma release oficial ({asset})...")
        partial = archive.with_name(archive.name + ".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "MosquitoWingbeat-Installer"})
            with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as stream:
                size, last_update = 0, time.monotonic()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    size += len(chunk)
                    if time.monotonic() - last_update > 2:
                        say(f"  {size / 1e6:.1f} MB recebidos...")
                        last_update = time.monotonic()
            if file_hash(partial) != expected_hash:
                raise InstallError("O SHA-256 do download não confere. A CLI não foi instalada; execute novamente.")
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    exe_name = "arduino-cli.exe" if os.name == "nt" else "arduino-cli"
    target = tools / "bin" / exe_name
    target.parent.mkdir(parents=True, exist_ok=True)
    # Extrair só o executável regular, evitando caminhos/links contidos no arquivo.
    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive) as handle:
            matches = [p for p in handle.infolist() if Path(p.filename).name == exe_name and not p.is_dir()]
            if len(matches) != 1:
                raise InstallError("A release não contém um único executável Arduino CLI.")
            with handle.open(matches[0]) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
    else:
        with tarfile.open(archive) as handle:
            matches = [p for p in handle.getmembers() if Path(p.name).name == exe_name and p.isfile()]
            if len(matches) != 1:
                raise InstallError("A release não contém um único executável Arduino CLI.")
            with handle.extractfile(matches[0]) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
    target.chmod(0o755)
    say("Download conferido por SHA-256; CLI instalada na pasta do projeto.")
    return target


def cli_version(executable):
    try:
        result = subprocess.run([str(executable), "--json", "version"], capture_output=True, text=True, timeout=15)
        return json.loads(result.stdout).get("VersionString") if result.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def find_cli(explicit, tools):
    if explicit:
        found = shutil.which(str(explicit))
        path = Path(found or explicit).expanduser().resolve()
        if cli_version(path) != CLI_VERSION:
            raise InstallError(f"--cli deve apontar para Arduino CLI {CLI_VERSION}; executável ausente ou versão diferente: {path}")
        return path
    name = "arduino-cli.exe" if os.name == "nt" else "arduino-cli"
    candidates = [tools / "bin" / name, ROOT / name]
    if shutil.which(name):
        candidates.append(Path(shutil.which(name)))
    for candidate in candidates:
        if candidate.is_file() and cli_version(candidate) == CLI_VERSION:
            say(f"Reutilizando Arduino CLI {CLI_VERSION}: {candidate}")
            return candidate.resolve()
    downloaded = download_cli(tools)
    if cli_version(downloaded) != CLI_VERSION:
        raise InstallError("A CLI baixada não executa neste computador. Use a Arduino IDE.")
    return downloaded


def make_config(tools, explicit):
    if explicit:
        config = Path(explicit).expanduser().resolve()
        if not config.is_file():
            raise InstallError(f"Configuração não encontrada: {config}")
        return config
    config = tools / "arduino-cli.yaml"
    lines = ["directories:"]
    for key in ("data", "downloads", "user"):
        folder = tools / key
        folder.mkdir(parents=True, exist_ok=True)
        lines.append(f"  {key}: {json.dumps(str(folder), ensure_ascii=False)}")
    lines += ["build_cache:", f"  path: {json.dumps(str(tools / 'cache'), ensure_ascii=False)}",
              "updater:", "  enable_notification: false",
              "network:", "  cloud_api:", "    skip_board_detection_calls: true"]
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config


class ArduinoCLI:
    def __init__(self, executable, config, tools):
        self.command = [str(executable), "--config-file", str(config), "--no-color"]
        self.tools = tools
        # Estas variáveis precederiam o YAML e poderiam escrever na instalação da IDE.
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith("ARDUINO_DIRECTORIES_") and k != "ARDUINO_BUILD_CACHE_PATH"}

    def json(self, *args):
        result = subprocess.run(self.command + ["--json", *args], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", env=self.env)
        if result.returncode:
            raise InstallError(f"Arduino CLI falhou em {' '.join(args)}:\n{(result.stderr or result.stdout)[-2000:]}")
        try:
            return json.loads(result.stdout)
        except ValueError as exc:
            raise InstallError("A CLI não retornou JSON válido. Confira a versão da CLI.") from exc

    def run(self, *args, log_name=None):
        command = self.command + list(args)
        if log_name is None:
            result = subprocess.run(command, env=self.env)
            code = result.returncode
        else:
            log = self.tools / log_name
            with log.open("w", encoding="utf-8") as stream, subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", env=self.env,
            ) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                code = process.wait()
        if code:
            raise InstallError(f"Falha em {args[0]} (código {code})." +
                               (f" Veja {self.tools / log_name}." if log_name else ""))


def ensure_core(cli, external_config):
    try:
        installed = {p["id"]: p.get("installed_version") for p in cli.json("core", "list").get("platforms", [])}
    except InstallError:
        if external_config:
            raise
        installed = {}
    if installed.get(CORE_ID) == CORE_VERSION:
        say(f"Core {CORE_ID}@{CORE_VERSION} já instalado; mantendo o cache.")
        return
    if external_config and installed.get(CORE_ID):
        raise InstallError("O --config-file informado tem outro core instalado. Use a configuração local padrão para instalar 4.6.0 sem substituir esse core.")
    say(f"Instalando {CORE_ID}@{CORE_VERSION}. A primeira preparação baixa ferramentas grandes; aguarde.")
    cli.run("core", "update-index", log_name="last_setup.log")
    cli.run("core", "install", f"{CORE_ID}@{CORE_VERSION}", log_name="last_core_install.log")
    installed = {p["id"]: p.get("installed_version") for p in cli.json("core", "list").get("platforms", [])}
    if installed.get(CORE_ID) != CORE_VERSION:
        raise InstallError("O core esperado não foi encontrado após a instalação.")


def serial_ports(cli):
    return [p for p in cli.json("board", "list", "--discovery-timeout", "3s").get("detected_ports", [])
            if p.get("port", {}).get("protocol") == "serial"]


def is_nano(entry):
    return any(b.get("fqbn", "").split(":")[:3] == FQBN.split(":")
               for b in entry.get("matching_boards", []))


def show_ports(ports):
    if not ports:
        say("Nenhuma porta serial detectada.")
    for index, entry in enumerate(ports, 1):
        names = ", ".join(b.get("name", "?") for b in entry.get("matching_boards", []))
        say(f"  {index}. {entry['port']['address']} - {names or 'placa não identificada'}")


PORT_ADVICE = ("Conecte a Nano 33 BLE Sense/Sense Rev2 com cabo USB de dados, feche o monitor da IDE "
               "e tente novamente. Se necessário, dê dois toques rápidos em RESET e use a nova porta.")


def select_port(ports, explicit, non_interactive):
    if explicit:
        entries = [p for p in ports if p["port"]["address"] == explicit]
        if not entries:
            show_ports(ports)
            raise InstallError(f"Porta {explicit} não detectada. {PORT_ADVICE}")
        selected = entries[0]
        if selected.get("matching_boards") and not is_nano(selected):
            raise InstallError(f"A porta {explicit} foi identificada como outra placa; escolha a porta da Nano 33 BLE Sense.")
    else:
        candidates = [p for p in ports if is_nano(p)]
        if len(candidates) == 1:
            selected = candidates[0]
        else:
            # Uma porta desconhecida requer escolha explícita, mesmo quando é a única.
            candidates = candidates or [p for p in ports if not p.get("matching_boards")]
            show_ports(candidates or ports)
            if not candidates:
                raise InstallError(PORT_ADVICE)
            if non_interactive or not sys.stdin.isatty():
                raise InstallError("Informe --port PORTA para escolher a placa. Nenhum upload foi iniciado.")
            choice = input("Escolha o número da sua Nano 33 BLE Sense (Enter cancela): ").strip()
            if not choice.isdigit() or not 1 <= int(choice) <= len(candidates):
                raise InstallError("Seleção cancelada. Nenhum upload foi iniciado.")
            selected = candidates[int(choice) - 1]
    port = selected["port"]["address"]
    if platform.system() == "Linux" and Path(port).exists() and not os.access(port, os.R_OK | os.W_OK):
        import grp
        group = grp.getgrgid(Path(port).stat().st_gid).gr_name
        advice = (f"Se aplicável: sudo usermod -aG {shlex.quote(group)} \"$USER\"; depois saia e entre na sessão."
                  if group in ("dialout", "uucp", "plugdev") else
                  "Confira as regras de acesso USB/serial da sua distribuição; o instalador não altera grupos ou permissões.")
        raise InstallError(f"Sem permissão de leitura/escrita em {port}. Confira o grupo com ls -l {port} (grupo atual: {group}). "
                           f"{advice} Não execute o instalador com sudo.")
    return selected


def monitor_port(cli, selected):
    properties = selected["port"].get("properties", {})
    serial = properties.get("serialNumber")
    for attempt in range(3):
        ports = serial_ports(cli)
        if serial:
            matches = [p for p in ports if p["port"].get("properties", {}).get("serialNumber") == serial]
        else:
            matches = [p for p in ports if p["port"]["address"] == selected["port"]["address"]]
        if len(matches) == 1:
            return matches[0]["port"]["address"]
        if attempt < 2:
            say("Aguardando a porta serial da placa reaparecer...")
            time.sleep(1)
    raise InstallError("A porta mudou ou ainda não reapareceu. Execute --monitor-only --port NOVA_PORTA; consulte --list-ports.")


def parser():
    args = argparse.ArgumentParser(description="Instalar o classificador de espécies no Nano 33 BLE Sense/Sense Rev2.",
                                   epilog="Exemplo: bash flash_arduino.sh --monitor | ZIP: python3 install_arduino.py --monitor")
    args.add_argument("legacy_port", nargs="?", help="porta; compatível com o script anterior")
    args.add_argument("legacy_sketch", nargs="?", help="pasta do sketch; compatível com o script anterior")
    args.add_argument("--port", help="porta serial, como /dev/ttyACM0 ou COM3")
    args.add_argument("--sketch", type=Path, help="pasta alternativa do sketch")
    modes = args.add_mutually_exclusive_group()
    modes.add_argument("--compile-only", action="store_true", help="preparar e compilar, sem placa e sem upload")
    modes.add_argument("--list-ports", action="store_true", help="preparar ferramentas e listar portas, sem upload")
    modes.add_argument("--monitor-only", action="store_true", help="abrir serial a 115200 baud, sem compilar/gravar")
    args.add_argument("--monitor", action="store_true", help="abrir monitor depois de gravar (Ctrl+C para sair)")
    args.add_argument("--non-interactive", action="store_true", help="falhar se a porta precisar de escolha manual")
    args.add_argument("--tools-dir", type=Path, default=ROOT / ".arduino-tools", help="cache local; padrão: .arduino-tools")
    args.add_argument("--cli", help="executável existente da Arduino CLI 1.5.1")
    args.add_argument("--config-file", type=Path, help="configuração existente da CLI (uso avançado)")
    return args


def main(argv=None):
    arg_parser = parser()
    options = arg_parser.parse_args(argv)
    if options.port and options.legacy_port:
        arg_parser.error("use --port ou a porta posicional, não ambos")
    if options.sketch and options.legacy_sketch:
        arg_parser.error("use --sketch ou a pasta posicional, não ambos")
    if options.monitor and (options.compile_only or options.list_ports):
        arg_parser.error("--monitor não pode acompanhar --compile-only ou --list-ports")
    default_sketch = ROOT if (ROOT / "MosquitoSpecies.ino").is_file() else ROOT / "firmware/MosquitoSpecies"
    specified_sketch = options.sketch or options.legacy_sketch
    sketch = Path(specified_sketch).expanduser() if specified_sketch else default_sketch
    if not sketch.is_absolute():
        sketch = ROOT / sketch
    sketch = sketch.resolve()
    if not (sketch / (sketch.name + ".ino")).is_file():
        raise InstallError(f"Sketch não encontrado: {sketch}. Extraia o ZIP inteiro, mantendo o .ino e os .h juntos.")
    tools = options.tools_dir.expanduser().resolve()
    tools.mkdir(parents=True, exist_ok=True)
    config = make_config(tools, options.config_file)
    say("Mosquito Wingbeat - Arduino Nano 33 BLE Sense / Sense Rev2")
    executable = find_cli(options.cli, tools)
    cli = ArduinoCLI(executable, config, tools)
    ensure_core(cli, bool(options.config_file))
    if options.list_ports:
        show_ports(serial_ports(cli))
        return 0
    selected = None
    if not options.compile_only:
        selected = select_port(serial_ports(cli), options.port or options.legacy_port, options.non_interactive)
        say(f"Porta selecionada: {selected['port']['address']}")
    if not options.monitor_only:
        build = tools / "build" / sketch.name
        say(f"Compilando {sketch.name} com core {CORE_VERSION}...")
        cli.run("compile", "--fqbn", FQBN, "--jobs", "2", "--build-path", str(build), str(sketch), log_name="last_compile.log")
        if options.compile_only:
            say(f"Compilação concluída. Nenhuma placa foi gravada. Binários: {build}")
            return 0
        say("Gravando o mesmo binário compilado; aguarde o término do upload...")
        try:
            cli.run("upload", "--fqbn", FQBN, "--port", selected["port"]["address"],
                    "--build-path", str(build), str(sketch), log_name="last_upload.log")
        except InstallError as exc:
            raise InstallError(f"{exc}\n{PORT_ADVICE}") from exc
        say("Upload concluído. O firmware emitirá estados e identificação provisória pelo serial.")
    if options.monitor or options.monitor_only:
        port = monitor_port(cli, selected)
        say(f"Monitor serial: {port}, 115200 baud. Ctrl+C para sair.")
        cli.run("monitor", "--port", port, "--config", "baudrate=115200")
    else:
        command = [sys.executable, str(Path(__file__).resolve()), "--monitor-only",
                   "--port", selected["port"]["address"], "--tools-dir", str(tools)]
        if options.cli:
            command += ["--cli", str(executable)]
        if options.config_file:
            command += ["--config-file", str(config)]
        if specified_sketch:
            command += ["--sketch", str(sketch)]
        say("Para ver as previsões:\n  " + (subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)))
    return 0


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        sys.exit("Este instalador requer Python 3.9 ou mais recente, ou use a Arduino IDE.")
    try:
        sys.exit(main())
    except (InstallError, OSError, urllib.error.URLError, EOFError) as exc:
        say(f"ERRO: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        say("\nOperação interrompida. Se interrompeu um upload, conecte novamente e repita a instalação.")
        sys.exit(130)
