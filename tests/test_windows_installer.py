"""Regras do aplicativo gráfico; uploads são simulados, sem hardware."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import install_arduino as backend
import windows_installer as app


def board(port="COM3", serial="SENSE-A", fqbn=backend.FQBN):
    return {"port": {"address": port, "protocol": "serial", "properties": {"serialNumber": serial}},
            "matching_boards": [{"name": "Nano 33 BLE", "fqbn": fqbn}]}


class ResultInterpretationTests(unittest.TestCase):
    def test_uncertain_candidate_is_never_displayed_as_identification(self):
        result = app.result_from_line("1,INCERTO,Aedes aegypti,-,0.7,0.48,0.8,1,0,20")
        self.assertEqual(result[0], "Espécie incerta")
        self.assertNotIn("aegypti", " ".join(result))

    def test_identification_is_explicitly_provisional(self):
        result = app.result_from_line("1,IDENTIFICACAO_PROVISORIA,Aedes aegypti,Aedes aegypti,0.7,0.48,0.8,1,0,20")
        self.assertEqual(result[0], "Possível Aedes aegypti")
        self.assertIn("provisória", result[1])

    def test_rejected_window_does_not_repeat_old_identification(self):
        for state in ("SEM_EVIDENCIA", "AUDIO_INVALIDO", "INCERTO"):
            result = app.result_from_line(f"1,{state},Aedes aegypti,-,0.7,0.48,0.8,1,0,20")
            self.assertNotIn("Possível", result[0])

    def test_malformed_lines_and_headers_are_ignored(self):
        for line in ("", "tempo_ms,estado,candidata", "# modelo", "1,IDENTIFICACAO_PROVISORIA,Aedes", "1,ESTADO_NOVO,-,-,0,0,0,0,0,0"):
            self.assertIsNone(app.result_from_line(line))

    def test_capture_loss_and_microphone_error_clear_result(self):
        self.assertEqual(app.result_from_line("# AUDIO_INTERROMPIDO: 20")[0], "Áudio interrompido")
        self.assertEqual(app.result_from_line("ERRO: microfone PDM indisponivel.")[2], "error")


class DesktopBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="Mosquito GUI com espaços ")
        self.addCleanup(self.temp.cleanup)
        self.logs = []
        self.client = app.DesktopBackend(Path(self.temp.name), self.logs.append, self.logs.append)
        self.client.cli = mock.Mock()

    def test_payload_is_complete_and_matches_checked_sources(self):
        destination = Path(self.temp.name) / "MosquitoSpecies"
        hashes = app.prepare_payload(destination)
        self.assertEqual(set(hashes), set(app.PAYLOAD_FILES))
        for name, digest in hashes.items():
            self.assertEqual(digest, backend.file_hash(ROOT / "firmware/MosquitoSpecies" / name))
        (destination / "SpeciesModel.h").write_text("arquivo alterado")
        self.assertEqual(app.prepare_payload(destination), hashes)

    def test_missing_embedded_payload_fails_before_install(self):
        with mock.patch.object(app, "RESOURCES", Path(self.temp.name)):
            with self.assertRaises(backend.InstallError):
                app.prepare_payload(Path(self.temp.name) / "MosquitoSpecies")

    def test_disconnected_board_does_not_compile_or_upload(self):
        with mock.patch.object(backend, "serial_ports", return_value=[]):
            with self.assertRaises(backend.InstallError):
                self.client.install(board())
        self.client.cli.run.assert_not_called()

    def test_wrong_board_is_refused(self):
        with mock.patch.object(backend, "serial_ports", return_value=[board(fqbn="arduino:avr:uno")]):
            with self.assertRaises(backend.InstallError):
                self.client.install(board())
        self.client.cli.run.assert_not_called()

    def test_failed_compile_does_not_upload(self):
        self.client.cli.run.side_effect = backend.InstallError("falha de compilação")
        with mock.patch.object(backend, "serial_ports", return_value=[board()]):
            with self.assertRaises(backend.InstallError):
                self.client.install(board())
        self.assertEqual([c.args[0] for c in self.client.cli.run.call_args_list], ["compile"])
        self.assertFalse(self.client.upload_succeeded)

    def test_upload_failure_does_not_set_success(self):
        self.client.cli.run.side_effect = [None, backend.InstallError("falha de gravação")]
        with mock.patch.object(backend, "serial_ports", return_value=[board()]):
            with self.assertRaisesRegex(backend.InstallError, "Não foi possível gravar"):
                self.client.install(board())
        self.assertFalse(self.client.upload_succeeded)

    def test_same_board_is_followed_when_com_address_changes(self):
        with mock.patch.object(backend, "serial_ports", return_value=[board("COM17"), board("COM3", serial="OTHER")]):
            port = self.client.install(board("COM3"))
        self.assertEqual(port, "COM17")
        calls = self.client.cli.run.call_args_list
        self.assertEqual([c.args[0] for c in calls], ["compile", "upload"])
        self.assertIn("COM17", calls[1].args)
        self.assertTrue(self.client.upload_succeeded)

    def test_missing_board_after_compile_prevents_upload(self):
        with mock.patch.object(backend, "serial_ports", side_effect=[[board()], []]):
            with self.assertRaises(backend.InstallError):
                self.client.install(board())
        self.assertEqual([c.args[0] for c in self.client.cli.run.call_args_list], ["compile"])


if __name__ == "__main__":
    unittest.main()
