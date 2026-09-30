"""Aplicativo gráfico portátil; Python e firmware são incorporados ao .exe."""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import sys
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, scrolledtext, ttk
import webbrowser

import install_arduino as backend

VERSION = "1.0.0"
DOWNLOAD_PAGE = "https://github.com/Lciarallo/mosquito-wingbeat/blob/main/WINDOWS.md"
PAYLOAD_FILES = ("MosquitoSpecies.ino", "StreamingFeatures.h", "PresenceModel.h", "SpeciesModel.h", "SpeciesDecision.h")
RESOURCES = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def data_directory():
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "MosquitoWingbeat"
    return Path.home() / ".local/share/MosquitoWingbeat"


def prepare_payload(destination):
    source = RESOURCES / "firmware/MosquitoSpecies"
    for name in PAYLOAD_FILES:
        if not (source / name).is_file():
            raise backend.InstallError("O aplicativo está incompleto. Baixe novamente o .exe pelo GitHub.")
    destination.mkdir(parents=True, exist_ok=True)
    for name in PAYLOAD_FILES:
        target = destination / name
        if not target.exists() or backend.file_hash(target) != backend.file_hash(source / name):
            shutil.copyfile(source / name, target)
    return {name: backend.file_hash(destination / name) for name in PAYLOAD_FILES}


def result_from_line(line):
    """Uma candidata incerta nunca aparece como espécie identificada."""
    if line.startswith("ERRO:"):
        return ("Microfone indisponível", "Confira se sua placa é Nano 33 BLE Sense ou Sense Rev2.", "error")
    if line.startswith("# AUDIO_INTERROMPIDO"):
        return ("Áudio interrompido", "A análise está sendo reiniciada. Aguarde novas amostras.", "warning")
    try:
        row = next(csv.reader([line]))
    except (csv.Error, StopIteration):
        return None
    if len(row) != 10 or not row[0].isdigit():
        return None
    state = row[1]
    if state == "IDENTIFICACAO_PROVISORIA" and row[3] not in ("", "-"):
        return ("Possível " + row[3], "Identificação provisória. O modelo pode errar.", "success")
    if state == "INCERTO":
        return ("Espécie incerta", "Este trecho não teve evidência suficiente para receber um nome.", "warning")
    if state == "SEM_EVIDENCIA":
        return ("Sem evidência suficiente", "O detector não identificou presença de mosquito neste trecho.", "neutral")
    if state == "AUDIO_INVALIDO":
        return ("Áudio insuficiente", "O som está muito baixo ou saturado. Confira a posição do microfone.", "warning")
    return None


class DesktopBackend:
    def __init__(self, folder, output, status):
        self.folder, self.output, self.status = Path(folder), output, status
        self.tools = self.folder / "tools"
        self.sketch = self.folder / "firmware" / VERSION / "MosquitoSpecies"
        self.cli = None
        self.upload_succeeded = False

    def prepare(self):
        self.status("Preparando o computador. A primeira vez pode levar alguns minutos.")
        self.tools.mkdir(parents=True, exist_ok=True)
        self.payload_hashes = prepare_payload(self.sketch)
        backend.say = self.output
        config = backend.make_config(self.tools, None)
        executable = backend.find_cli(None, self.tools)
        self.cli = backend.ArduinoCLI(executable, config, self.tools, on_output=self.output)
        backend.ensure_core(self.cli, False)

    def scan(self):
        self.prepare()
        self.status("Procurando sua placa Arduino...")
        return [p for p in backend.serial_ports(self.cli)
                if backend.is_nano(p) or not p.get("matching_boards")]

    def compile(self):
        self.status("Preparando o programa para sua placa...")
        self.build = self.tools / "build/MosquitoSpecies"
        self.cli.run("compile", "--fqbn", backend.FQBN, "--jobs", "2", "--build-path", str(self.build),
                     str(self.sketch), log_name="last_compile.log")

    def current_board(self, selected):
        ports = backend.serial_ports(self.cli)
        serial_number = selected["port"].get("properties", {}).get("serialNumber")
        matches = [p for p in ports if (p["port"].get("properties", {}).get("serialNumber") == serial_number
                   if serial_number else p["port"]["address"] == selected["port"]["address"])]
        if len(matches) != 1:
            raise backend.InstallError("Sua placa foi desconectada ou a porta mudou. Clique em Buscar minha placa e tente novamente.")
        return backend.select_port(ports, matches[0]["port"]["address"], True)

    def install(self, selected):
        self.upload_succeeded = False
        self.current_board(selected)
        self.compile()
        selected = self.current_board(selected)
        self.status("Gravando na placa. Mantenha o cabo conectado até terminar.")
        try:
            self.cli.run("upload", "--fqbn", backend.FQBN, "--port", selected["port"]["address"],
                         "--build-path", str(self.build), str(self.sketch), log_name="last_upload.log")
        except backend.InstallError as exc:
            raise backend.InstallError("Não foi possível gravar. Feche o monitor da Arduino IDE. Se necessário, toque duas vezes em RESET, busque a placa novamente e repita.\n\n" + str(exc)) from exc
        self.upload_succeeded = True
        self.status("Programa gravado. Aguardando a placa reiniciar...")
        return backend.monitor_port(self.cli, selected)


class InstallerWindow:
    def __init__(self, root, folder=None):
        self.root = root
        self.folder = Path(folder) if folder else data_directory()
        self.events = queue.Queue()
        self.device = DesktopBackend(self.folder, lambda text: self.events.put(("log", text)),
                                     lambda text: self.events.put(("status", text)))
        self.busy = False
        self.ports = []
        self.monitor_stop = threading.Event()
        self.monitor_thread = None
        self.monitor_port = None
        self.root.title("Mosquito Wingbeat | Instalar no Arduino")
        left, top, right, bottom = 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()
        if os.name == "nt":
            from ctypes import wintypes
            area = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(area), 0):
                left, top, right, bottom = area.left, area.top, area.right, area.bottom
        # Fontes em pixels e dimensões na mesma escala evitam cortes em monitores
        # com DPI alto. Reservar espaço para a barra de tarefas e bordas da janela.
        self.scale = min(max(float(root.tk.call("tk", "scaling")) / (96 / 72), 1),
                         (right - left - 60) / 860, (bottom - top - 60) / 660)
        width, height = self.px(860), self.px(660)
        x, y = left + max(0, (right - left - width) // 2), top + max(0, (bottom - top - height - 40) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(self.px(790), self.px(600))
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            tkfont.nametofont(name, root=root).configure(family="Segoe UI", size=-self.px(15))
        self.root.configure(bg="#edf2f8")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_widgets()
        # PyInstaller altera a busca de DLLs. A CLI/GCC devem usar as DLLs do sistema.
        # Tk/SSL/Pillow (diagnóstico) já estão carregados antes desta mudança.
        if os.name == "nt" and getattr(sys, "frozen", False):
            ctypes.windll.kernel32.SetDllDirectoryW(None)
        self.root.after(80, self._poll)

    def px(self, value):
        return max(1, round(value * self.scale))

    def _build_widgets(self):
        p = self.px
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#edf2f8")
        style.configure("TLabel", background="#edf2f8", foreground="#16314d", font=("Segoe UI", -p(15)))
        style.configure("Title.TLabel", font=("Segoe UI", -p(30), "bold"))
        style.configure("Small.TLabel", font=("Segoe UI", -p(13)), foreground="#52677e")
        style.configure("TButton", font=("Segoe UI", -p(15)), padding=(p(12), p(10)))
        style.configure("Install.TButton", background="#2463dc", foreground="white", font=("Segoe UI", -p(15), "bold"))
        style.map("Install.TButton", background=[("disabled", "#becadb"), ("active", "#1d4eb3")])
        style.configure("TCombobox", padding=p(8), font=("Segoe UI", -p(15)))
        style.configure("TNotebook.Tab", font=("Segoe UI", -p(13)), padding=(p(10), p(6)))
        body = ttk.Frame(self.root, padding=(p(28), p(22)))
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Instalar detector no Arduino", style="Title.TLabel").pack(anchor="w")
        ttk.Label(body, text="Nano 33 BLE Sense ou Sense Rev2 • Versão " + VERSION, style="Small.TLabel").pack(anchor="w", pady=(p(4), p(17)))
        ttk.Label(body, text="1. Conecte a placa ao computador com um cabo USB de dados.").pack(anchor="w")
        ttk.Label(body, text="2. Busque sua placa e depois clique em Instalar no Arduino.").pack(anchor="w", pady=(p(3), p(14)))
        row = ttk.Frame(body)
        row.pack(fill="x")
        self.scan_button = ttk.Button(row, text="Buscar minha placa", command=self.scan)
        self.scan_button.pack(side="left", padx=(0, p(12)))
        self.port_box = ttk.Combobox(row, state="disabled", font=("Segoe UI", -p(15)))
        self.port_box.pack(side="left", fill="x", expand=True)
        self.port_box.bind("<<ComboboxSelected>>", lambda _: self._controls())
        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=(p(14), p(12)))
        self.install_button = ttk.Button(actions, text="Instalar no Arduino", style="Install.TButton", command=self.install, state="disabled")
        self.install_button.pack(side="left", padx=(0, p(10)))
        self.monitor_button = ttk.Button(actions, text="Ver resultados", command=self.toggle_monitor, state="disabled")
        self.monitor_button.pack(side="left")
        ttk.Button(actions, text="Passo a passo", command=lambda: webbrowser.open(DOWNLOAD_PAGE)).pack(side="right")
        self.status = tk.StringVar(value="Pronto para começar. A primeira preparação usa internet e cerca de 1 GB de espaço.")
        self.status_label = ttk.Label(body, textvariable=self.status, wraplength=p(735), justify="left")
        self.status_label.pack(anchor="w", fill="x", pady=(p(3), p(8)))
        self.progress = ttk.Progressbar(body, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, p(16)))
        tabs = ttk.Notebook(body)
        tabs.pack(fill="both", expand=True)
        result_panel = tk.Frame(tabs, bg="white", padx=p(22), pady=p(22))
        tabs.add(result_panel, text=" Resultado ")
        self.result_title = tk.Label(result_panel, text="Aguardando a placa", bg="white", fg="#16314d", font=("Segoe UI", -p(28), "bold"), anchor="w", justify="left", wraplength=p(660))
        self.result_title.pack(fill="x", pady=(0, p(9)))
        self.result_detail = tk.Label(result_panel, text="Após a instalação, os resultados aparecem aqui automaticamente.", bg="white", fg="#52677e", font=("Segoe UI", -p(15)), anchor="w", justify="left", wraplength=p(660))
        self.result_detail.pack(fill="x")
        self.log = scrolledtext.ScrolledText(tabs, height=9, font=("Consolas", -p(12)), bg="#102339", fg="#dfeafa", state="disabled")
        tabs.add(self.log, text=" Detalhes ")
        self.footer = ttk.Label(body, text="Modelo experimental: pode errar e frequentemente responde “espécie incerta”.", style="Small.TLabel")
        self.footer.pack(anchor="w", pady=(p(14), 0))
        self._controls()

    def _controls(self):
        monitoring = bool(self.monitor_thread and self.monitor_thread.is_alive())
        self.scan_button.configure(state="disabled" if self.busy else "normal")
        self.port_box.configure(state="disabled" if self.busy or not self.ports else "readonly")
        chosen = self.port_box.current() >= 0
        self.install_button.configure(state="normal" if chosen and not self.busy else "disabled")
        self.monitor_button.configure(state="normal" if (self.monitor_port or monitoring or chosen) and not self.busy else "disabled",
                                      text="Parar resultados" if monitoring else "Ver resultados")

    def _start_work(self, function, success):
        self.stop_monitor()
        self.busy = True
        self.progress.start(12)
        self._controls()
        def work():
            try:
                self.events.put((success, function()))
            except Exception as exc:
                self.events.put(("error", str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def scan(self):
        self.device.upload_succeeded = False
        self.monitor_port = None
        self.port_box.set("")
        self.ports = []
        self._start_work(self.device.scan, "ports")

    def install(self):
        index = self.port_box.current()
        if index < 0 or self.busy:
            return
        selected = self.ports[index]
        self._set_result(("Preparando a instalação", "Aguarde o término e mantenha o cabo conectado.", "neutral"))
        self._start_work(lambda: self.device.install(selected), "installed")

    def _finish_work(self):
        self.busy = False
        self.progress.stop()
        self._controls()

    def _poll(self):
        for _ in range(100):
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status.set(value)
            elif kind == "log":
                self._append_log(value)
            elif kind == "ports":
                self.ports = value
                labels = [("Arduino Nano 33 BLE" if backend.is_nano(p) else "Porta não identificada") + " — " + p["port"]["address"] for p in value]
                self.port_box.configure(values=labels)
                known = [i for i, p in enumerate(value) if backend.is_nano(p)]
                if len(known) == 1:
                    self.port_box.current(known[0])
                    self.status.set("Placa encontrada. Instale no primeiro uso ou abra os resultados se já instalou.")
                elif value:
                    self.status.set("Escolha a porta da sua placa na lista. Se houver dúvida, desconecte as outras placas e busque novamente.")
                else:
                    self.status.set("Nenhuma placa encontrada. Confira o cabo USB de dados; se necessário, toque duas vezes em RESET e busque novamente.")
                self._finish_work()
            elif kind == "installed":
                self.monitor_port = value
                self.status.set("Instalação concluída. Os resultados serão exibidos abaixo.")
                self._finish_work()
                self.start_monitor()
            elif kind == "error":
                self._append_log(value)
                if self.device.upload_succeeded:
                    self.status.set("Programa gravado. O monitor não abriu; busque a placa e abra os resultados novamente.")
                else:
                    self.status.set("A operação não foi concluída. Confira a mensagem abaixo e tente novamente.")
                self._set_result(("Precisamos conferir a conexão", value, "error"))
                self._finish_work()
            elif kind == "line":
                self._append_log(value)
                result = result_from_line(value)
                if result:
                    self._set_result(result)
            elif kind == "monitor_error":
                self.status.set("O monitor foi desconectado. Busque a placa e tente novamente.")
                self._set_result(("Conexão interrompida", value, "error"))
                self._controls()
            elif kind == "diagnostic_done":
                self.diagnostic_result = value
                self.root.after(200, self._save_diagnostic)
        self.root.after(80, self._poll)

    def _append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", str(text) + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 1000:
            self.log.delete("1.0", "200.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_result(self, result):
        title, detail, tone = result
        color = {"success": "#116149", "warning": "#916000", "error": "#ab2b36"}.get(tone, "#16314d")
        self.result_title.configure(text=title, fg=color)
        self.result_detail.configure(text=detail)

    def start_monitor(self):
        if not self.monitor_port or self.busy or (self.monitor_thread and self.monitor_thread.is_alive()):
            return
        self.monitor_stop = threading.Event()
        stop = self.monitor_stop
        port = self.monitor_port
        self._set_result(("Analisando o som", "Aguarde alguns segundos pelas primeiras leituras.", "neutral"))
        def read():
            try:
                import serial
                with serial.Serial(port, baudrate=115200, timeout=0.5) as connection:
                    while not stop.is_set():
                        data = connection.readline(4096)
                        if data:
                            self.events.put(("line", data.decode("utf-8", errors="replace").strip()))
            except Exception as exc:
                if not stop.is_set():
                    self.events.put(("monitor_error", str(exc)))
        self.monitor_thread = threading.Thread(target=read, daemon=True)
        self.monitor_thread.start()
        self._controls()

    def stop_monitor(self):
        self.monitor_stop.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        self.monitor_thread = None
        self._controls()

    def toggle_monitor(self):
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_monitor()
            self._set_result(("Resultados pausados", "Clique em Ver resultados para retomar a leitura.", "neutral"))
        else:
            if not self.monitor_port and self.port_box.current() >= 0:
                self.monitor_port = self.ports[self.port_box.current()]["port"]["address"]
            self.start_monitor()

    def close(self):
        if self.busy:
            messagebox.showinfo("Operação em andamento", "Aguarde a preparação ou gravação terminar antes de fechar. Você pode minimizar esta janela.")
            return
        self.stop_monitor()
        self.root.destroy()

    def diagnostic(self, path, compile_check):
        """Diagnóstico de build: nunca chama install/upload nem abre porta serial."""
        self.diagnostic_path = Path(path)
        self.status.set("Verificação automática do pacote. Nenhuma placa será gravada.")
        def check():
            report = {"version": VERSION, "platform": platform.platform(), "frozen_executable": bool(getattr(sys, "frozen", False)),
                      "gui_created": True, "physical_board_tested": False, "flashed": False}
            try:
                report["firmware_sha256"] = prepare_payload(self.device.sketch)
                # Checar presença do leitor serial no executável, sem abrir hardware.
                import serial
                report["serial_module_version"] = serial.VERSION
                if compile_check:
                    self.device.prepare()
                    self.device.compile()
                    output = (self.device.tools / "last_compile.log").read_text(encoding="utf-8")
                    report["program_storage_bytes"] = int(re.search(r"Sketch uses (\d+) bytes", output).group(1))
                    report["global_static_memory_bytes"] = int(re.search(r"Global variables use (\d+) bytes", output).group(1))
                    report["compile_succeeded"] = True
                    report["cli_version"] = backend.cli_version(self.device.cli.command[0])
                    report["core_version"] = backend.CORE_VERSION
                report["result_parser_checked"] = result_from_line("100,INCERTO,Aedes aegypti,-,0.7,0.48,0.8,1,0,20")[0] == "Espécie incerta"
                report["success"] = True
            except Exception as exc:
                report["success"], report["error"] = False, str(exc)
            self.events.put(("diagnostic_done", report))
        threading.Thread(target=check, daemon=True).start()

    def _save_diagnostic(self):
        report = self.diagnostic_result
        self.status.set("Verificação concluída. Nenhuma placa foi gravada.")
        self.root.update_idletasks()
        key_widgets = (self.scan_button, self.port_box, self.install_button,
                       self.monitor_button, self.result_title, self.result_detail, self.footer)
        report["gui_layout_fits"] = all(
            widget.winfo_ismapped() and widget.winfo_height() >= widget.winfo_reqheight()
            and widget.winfo_rooty() + widget.winfo_height() <= self.root.winfo_rooty() + self.root.winfo_height()
            for widget in key_widgets)
        report["gui_scale"] = self.scale
        report["gui_client_size"] = [self.root.winfo_width(), self.root.winfo_height()]
        if not report["gui_layout_fits"]:
            report["success"], report["error"] = False, "A janela cortou controles ou resultados."
        try:
            from PIL import ImageGrab
            if os.name == "nt":
                ancestor = ctypes.windll.user32.GetAncestor
                ancestor.argtypes, ancestor.restype = [ctypes.c_void_p, ctypes.c_uint], ctypes.c_void_p
                image = ImageGrab.grab(window=ancestor(self.root.winfo_id(), 2))
            else:
                x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
                image = ImageGrab.grab(bbox=(x, y, x + self.root.winfo_width(), y + self.root.winfo_height()),
                                       xdisplay=os.environ.get("DISPLAY", ""))
            image.save(self.diagnostic_path.with_suffix(".png"))
            report["gui_screenshot_captured"] = True
        except Exception as exc:
            report["gui_screenshot_captured"] = False
            report["screenshot_error"] = str(exc)
        self.diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        self.diagnostic_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", type=Path)
    parser.add_argument("--compile-check", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    if args.compile_check and not args.self_test:
        parser.error("--compile-check requer --self-test; nunca inicia upload")
    if os.name == "nt":
        ctypes.windll.user32.SetProcessDPIAware()
    # Carregar DLLs de screenshot/serial antes de restaurar a busca das DLLs externas.
    if args.self_test:
        from PIL import ImageGrab
    import serial
    root = tk.Tk()
    app = InstallerWindow(root, args.data_dir)
    if args.self_test:
        args.self_test.parent.mkdir(parents=True, exist_ok=True)
        root.after(300, lambda: app.diagnostic(args.self_test, args.compile_check))
    root.mainloop()
    return 0 if not args.self_test or getattr(app, "diagnostic_result", {}).get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
