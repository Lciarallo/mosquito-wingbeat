"""Verifica o fluxo do instalador sem conectar/gravar hardware.

O processo externo arduino-cli é simulado; a compilação real está na auditoria
installation_audit.json e usa --compile-only. Nenhum teste faz upload físico.
"""
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("installer", PROJECT / "install_arduino.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)

FAKE_CLI = r'''
import json, os
from pathlib import Path
import sys
args = sys.argv[1:]
if '--config-file' in args:
    index = args.index('--config-file')
    del args[index:index+2]
args = [a for a in args if a not in ('--json','--no-color')]
with open(os.environ['TASK_FAKE_TRACE'],'a',encoding='utf-8') as stream:
    stream.write(json.dumps(args)+'\n')
state = Path(os.environ['TASK_FAKE_STATE'])
installed = state.read_text() if state.exists() else os.environ.get('TASK_FAKE_CORE','4.6.0')
if args == ['version']:
    print(json.dumps({'VersionString':os.environ.get('TASK_FAKE_VERSION','1.5.1')}))
elif args[:2] == ['core','list']:
    print(json.dumps({'platforms':[{'id':'arduino:mbed_nano','installed_version':installed}] if installed else []}))
elif args[:2] == ['core','install']:
    assert args[2] == 'arduino:mbed_nano@4.6.0'
    state.write_text('4.6.0')
elif args[:2] == ['core','update-index']:
    pass
elif args[:2] == ['board','list']:
    uploaded = Path(os.environ['TASK_FAKE_STATE']+'.uploaded').exists()
    value = os.environ.get('TASK_FAKE_PORTS_AFTER_UPLOAD') if uploaded else None
    print(json.dumps({'detected_ports':json.loads(value or os.environ['TASK_FAKE_PORTS'])}))
elif args[0] in ('compile','upload','monitor'):
    if os.environ.get('TASK_FAKE_FAIL') == args[0]:
        print('Falha controlada em '+args[0],file=sys.stderr)
        sys.exit(7)
    if args[0] == 'compile':
        assert args[args.index('--fqbn')+1] == 'arduino:mbed_nano:nano33ble'
        build = Path(args[args.index('--build-path')+1])
        build.mkdir(parents=True,exist_ok=True)
        (build/'compiled.fixture').write_text('binario de teste')
    elif args[0] == 'upload':
        assert (Path(args[args.index('--build-path')+1])/'compiled.fixture').is_file()
        Path(os.environ['TASK_FAKE_STATE']+'.uploaded').touch()
    elif args[0] == 'monitor':
        assert args[args.index('--config')+1] == 'baudrate=115200'
else:
    sys.exit('Comando inesperado: '+repr(args))
'''


def board(address="/dev/ttyACM0", fqbn=installer.FQBN, serial="SENSE-A"):
    return {"port": {"address": address, "protocol": "serial", "properties": {"serialNumber": serial}},
            "matching_boards": [{"name": "Arduino Nano 33 BLE", "fqbn": fqbn}] if fqbn else []}


@unittest.skipIf(os.name == "nt", "O executável CLI simulado usa um shebang POSIX; não é teste nativo Windows.")
class InstallerProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="Mosquito instalação com espaços ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sketch = self.root / "firmware/MosquitoSpecies"
        self.sketch.mkdir(parents=True)
        (self.sketch / "MosquitoSpecies.ino").write_text("void setup(){} void loop(){}")
        for name in ("install_arduino.py", "flash_arduino.sh"):
            shutil.copy2(PROJECT / name, self.root / name)
        self.cli = self.root / "cli simulado"
        self.cli.write_text("#!" + sys.executable + "\n" + FAKE_CLI, encoding="utf-8")
        self.cli.chmod(0o755)
        self.trace = self.root / "trace.jsonl"
        self.state = self.root / "state"

    def invoke(self, *args, ports=None, wrapper=False, **environment):
        env = dict(os.environ, TASK_FAKE_TRACE=str(self.trace), TASK_FAKE_STATE=str(self.state),
                   TASK_FAKE_PORTS=json.dumps(ports if ports is not None else [board()]))
        env.update(environment)
        command = ["bash", str(self.root / "flash_arduino.sh")] if wrapper else [sys.executable, str(self.root / "install_arduino.py")]
        result = subprocess.run(command + ["--cli", str(self.cli), *args],
                                cwd=self.root.parent, env=env, capture_output=True, text=True, timeout=20)
        self.calls = [json.loads(line) for line in self.trace.read_text().splitlines()] if self.trace.exists() else []
        return result

    def commands(self):
        return [call[0] for call in self.calls]

    def test_compile_only_does_not_discover_or_upload(self):
        result = self.invoke("--compile-only", ports=[])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Nenhuma placa foi gravada", result.stdout)
        self.assertNotIn("board", self.commands())
        self.assertNotIn("upload", self.commands())

    def test_bash_wrapper_from_other_directory_and_legacy_port(self):
        result = self.invoke("/dev/ttyACM0", "firmware/MosquitoSpecies", wrapper=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        compile_call = next(p for p in self.calls if p[0] == "compile")
        upload_call = next(p for p in self.calls if p[0] == "upload")
        self.assertEqual(compile_call[-1], str(self.sketch))
        self.assertEqual(compile_call[compile_call.index("--build-path")+1], upload_call[upload_call.index("--build-path")+1])

    def test_single_matching_nano_autoselects_and_monitors(self):
        result = self.invoke("--monitor", ports=[board("COM17"), board("COM2", "arduino:avr:uno", serial="UNO-B")])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertLess(self.commands().index("compile"), self.commands().index("upload"))
        self.assertLess(self.commands().index("upload"), self.commands().index("monitor"))
        self.assertIn("COM17", next(p for p in self.calls if p[0] == "upload"))

    def test_multiple_nanos_require_explicit_choice(self):
        result = self.invoke("--non-interactive", ports=[board(), board("/dev/ttyACM1", serial="SENSE-B")])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Informe --port", result.stdout)
        self.assertNotIn("compile", self.commands())
        self.assertNotIn("upload", self.commands())

    def test_explicit_port_selects_among_multiple_nanos(self):
        result = self.invoke("--port", "COM8", ports=[board("COM7"), board("COM8", serial="SENSE-B")])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("COM8", next(p for p in self.calls if p[0] == "upload"))

    def test_wrong_board_is_refused(self):
        result = self.invoke("--port", "COM3", ports=[board("COM3", "arduino:avr:uno")])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outra placa", result.stdout)
        self.assertNotIn("upload", self.commands())

    def test_unknown_board_is_not_automatically_selected(self):
        result = self.invoke(ports=[board(fqbn=None)])
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("upload", self.commands())

    def test_unknown_board_accepts_explicit_port(self):
        result = self.invoke("--port", "COM3", ports=[board("COM3", fqbn=None)])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("upload", self.commands())

    def test_absent_port_stops_before_compile(self):
        result = self.invoke("--port", "COM99", ports=[])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cabo USB de dados", result.stdout)
        self.assertNotIn("compile", self.commands())

    def test_failed_compile_never_uploads(self):
        result = self.invoke("--monitor", TASK_FAKE_FAIL="compile")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("upload", self.commands())
        self.assertNotIn("monitor", self.commands())
        self.assertIn("last_compile.log", result.stdout)

    def test_failed_upload_does_not_claim_success_or_monitor(self):
        result = self.invoke("--monitor", TASK_FAKE_FAIL="upload")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Upload concluído", result.stdout)
        self.assertNotIn("monitor", self.commands())
        self.assertIn("last_upload.log", result.stdout)

    def test_monitor_follows_board_serial_after_port_change(self):
        result = self.invoke("--monitor", ports=[board("COM3")],
                             TASK_FAKE_PORTS_AFTER_UPLOAD=json.dumps([board("COM4"), board("COM3", serial="SENSE-B")]))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("COM4", next(p for p in self.calls if p[0] == "monitor"))

    def test_monitor_only_does_not_flash(self):
        result = self.invoke("--monitor-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("compile", self.commands())
        self.assertNotIn("upload", self.commands())
        self.assertIn("monitor", self.commands())

    def test_missing_core_is_installed_at_pinned_version(self):
        result = self.invoke("--compile-only", TASK_FAKE_CORE="")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(["core", "install", "arduino:mbed_nano@4.6.0"], self.calls)

    def test_ready_core_does_not_update_indexes(self):
        result = self.invoke("--compile-only")
        self.assertEqual(result.returncode, 0)
        self.assertNotIn(["core", "update-index"], self.calls)
        self.assertNotIn(["core", "install", "arduino:mbed_nano@4.6.0"], self.calls)

    def test_extracted_zip_layout_needs_no_repository(self):
        extracted = self.root / "MosquitoSpecies"
        extracted.mkdir()
        for path in list(self.sketch.iterdir()):
            path.replace(extracted / path.name)
        for name in ("install_arduino.py", "flash_arduino.sh"):
            (self.root / name).replace(extracted / name)
        self.sketch.rmdir()
        self.sketch.parent.rmdir()
        self.root = extracted
        result = self.invoke("--compile-only", ports=[])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(next(p for p in self.calls if p[0] == "compile")[-1], str(self.root))

    def test_invalid_sketch_and_flags_do_not_call_cli(self):
        result = self.invoke("--compile-only", "--monitor")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.calls, [])
        result = self.invoke("--sketch", "pasta ausente")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls, [])

    def test_wrong_cli_version_fails_before_setup(self):
        result = self.invoke("--compile-only", TASK_FAKE_VERSION="0.35.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.commands(), ["version"])

    def test_list_ports_does_not_compile_or_flash(self):
        result = self.invoke("--list-ports", ports=[board("COM3")])
        self.assertEqual(result.returncode, 0)
        self.assertIn("COM3", result.stdout)
        self.assertNotIn("compile", self.commands())
        self.assertNotIn("upload", self.commands())


class DownloadIntegrityTests(unittest.TestCase):
    def test_platform_assets_have_pinned_hashes(self):
        for digest in installer.CLI_SHA256.values():
            self.assertEqual(len(digest), 64)
        for system, machine in [("Linux", "x86_64"), ("Linux", "aarch64"), ("Darwin", "arm64"),
                                ("Darwin", "x86_64"), ("Windows", "AMD64"), ("Windows", "x86")]:
            asset, digest = installer.release_asset(system, machine)
            self.assertIn("1.5.1", asset)
            self.assertEqual(len(digest), 64)
        with self.assertRaises(installer.InstallError):
            installer.release_asset("Windows", "ARM64")

    def test_bad_download_is_not_extracted_or_executed(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(installer.urllib.request, "urlopen", return_value=io.BytesIO(b"corrompido")):
            target = Path(temp)
            with self.assertRaisesRegex(installer.InstallError, "SHA-256"):
                installer.download_cli(target)
            self.assertFalse((target / "bin").exists())
            self.assertEqual(list(target.rglob("*.part")), [])


@unittest.skipIf(os.name == "nt", "Diagnóstico de permissões POSIX.")
class PermissionDiagnosticsTests(unittest.TestCase):
    def test_denied_serial_uses_actual_serial_group(self):
        import grp
        with tempfile.NamedTemporaryFile() as port, mock.patch.object(installer.platform, "system", return_value="Linux"), \
             mock.patch.object(installer.os, "access", return_value=False), \
             mock.patch.object(grp, "getgrgid", return_value=SimpleNamespace(gr_name="dialout")):
            with self.assertRaisesRegex(installer.InstallError, "usermod -aG dialout"):
                installer.select_port([board(port.name)], port.name, True)

    def test_root_group_does_not_suggest_adding_user_to_root(self):
        import grp
        with tempfile.NamedTemporaryFile() as port, mock.patch.object(installer.platform, "system", return_value="Linux"), \
             mock.patch.object(installer.os, "access", return_value=False), \
             mock.patch.object(grp, "getgrgid", return_value=SimpleNamespace(gr_name="root")):
            with self.assertRaises(installer.InstallError) as result:
                installer.select_port([board(port.name)], port.name, True)
            self.assertNotIn("usermod", str(result.exception))
            self.assertIn("regras de acesso USB/serial", str(result.exception))


if __name__ == "__main__":
    unittest.main()
