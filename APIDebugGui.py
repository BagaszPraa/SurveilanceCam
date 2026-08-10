#!/usr/bin/env python3
"""
APIDebugGUI.py — GUI debug client (PySide6) untuk APIController
(drone/edge AI WebSocket server).

Fungsi:
  - Connect/disconnect ke APIController lewat GUI (host & port bisa diubah).
  - Live view pesan "result" (deteksi + crowd counting + fps) dan "config"
    (status konfigurasi AI) di panel terpisah.
  - Ubah semua parameter yang didukung config_command:
      conf, nms, class, detect, crowd, overlay, dan golongan semi-dev.
  - Lihat config_ack (applied/rejected) langsung di panel status.
  - Log mentah (raw JSON) opsional, plus statistik jumlah pesan per tipe.

Catatan format pesan broadcast (dari APIController, key "t"):
  "result" -- dikirim tiap frame:
    { "t":"result", "ts":..., "f":<frame_id>, "res":{"w":..,"h":..},
      "inf":<inference_ms>, "fps":<fps>,
      "det":[{"id":1,"cls":"person","cf":0.87,"bbox":[x,y,w,h]}, ...],
      "crwd":{"cnt":1000,"lvl":"crowded"} }

  "config" -- dikirim saat konfigurasi berubah / sinkronisasi:
    { "t":"config", "ts":..., "conf":0.5, "nms":0.45,
      "cls":["person","vehicle"], "overlay":true,
      "det_on":true, "crwd_on":true }

  Pesan command/ack (GCS<->Drone) TIDAK berubah, masih pakai key "type":
    cmd (kirim)      -> {"type":"cmd","command_id":...,"params":{...}}
    config_ack (terima) -> {"type":"config_ack","command_id":...,"status":...}

Instalasi dependency:
    pip install PySide6 websocket-client

Jalankan:
    python3 APIDebugGUI.py
"""

import sys
import json
import time
import uuid
from datetime import datetime, timezone

try:
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QFormLayout, QGridLayout, QLineEdit, QPushButton, QLabel,
        QDoubleSpinBox, QPlainTextEdit, QGroupBox, QTableWidget,
        QTableWidgetItem, QHeaderView, QSplitter, QCheckBox, QStatusBar,
        QMessageBox, QListWidget, QListWidgetItem, QFileDialog,
    )
except ImportError:
    print("Library 'PySide6' belum terinstall.")
    print("Install dulu: pip install PySide6 websocket-client")
    sys.exit(1)

try:
    import websocket
except ImportError:
    print("Library 'websocket-client' belum terinstall.")
    print("Install dulu: pip install websocket-client")
    sys.exit(1)


def now_str() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ---------------------------------------------------------------------------
# Worker thread: menjaga koneksi WebSocket tetap hidup di background,
# supaya GUI (main thread) tidak pernah blocking.
# ---------------------------------------------------------------------------
class WsClientThread(QThread):
    connected = Signal()
    disconnected = Signal(int, str)
    message = Signal(dict)
    raw_message = Signal(str)
    error = Signal(str)

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self.ws = None
        self._stop = False

    def run(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=lambda ws: self.connected.emit(),
            on_close=lambda ws, code, msg: self.disconnected.emit(code or 0, msg or ""),
            on_error=lambda ws, err: self.error.emit(str(err)),
            on_message=self._on_message,
        )
        while not self._stop:
            self.ws.run_forever(reconnect=3)
            if self._stop:
                break
            time.sleep(1)

    def _on_message(self, ws, raw: str):
        self.raw_message.emit(raw)
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self.error.emit(f"Gagal parse JSON: {raw[:200]}")
            return
        self.message.emit(msg)

    def send(self, payload: dict) -> bool:
        if self.ws is not None and self.ws.sock is not None and self.ws.sock.connected:
            self.ws.send(json.dumps(payload))
            return True
        return False

    def stop(self):
        self._stop = True
        if self.ws:
            self.ws.close()
        self.wait(2000)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("APIController Debug GUI")
        self.resize(980, 720)

        self.ws_thread: WsClientThread | None = None
        self.pending_acks: dict[str, dict] = {}
        self.msg_counts: dict[str, int] = {}
        self.last_result_ts: float | None = None

        # Nama class dari file .names (index baris = class ID), dipakai
        # untuk menampilkan nama alih-alih ID mentah di panel config &
        # status. Kosong kalau belum ada file di-load.
        self.class_names: list[str] = []
        self._syncing_class_list = False  # guard supaya event tidak saling memicu

        self._build_ui()

    # ------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        root.addWidget(self._build_connection_group())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_config_group())
        splitter.addWidget(self._build_status_group())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        root.addWidget(self._build_detections_group(), stretch=1)
        root.addWidget(self._build_log_group(), stretch=1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.stats_label = QLabel("Belum ada pesan diterima")
        self.status_bar.addPermanentWidget(self.stats_label)

    def _build_connection_group(self) -> QGroupBox:
        box = QGroupBox("Koneksi")
        layout = QHBoxLayout(box)

        layout.addWidget(QLabel("Host:"))
        self.host_edit = QLineEdit("127.0.0.1")
        self.host_edit.setFixedWidth(140)
        layout.addWidget(self.host_edit)

        layout.addWidget(QLabel("Port:"))
        self.port_edit = QLineEdit("8765")
        self.port_edit.setFixedWidth(70)
        layout.addWidget(self.port_edit)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        layout.addWidget(self.connect_btn)

        self.conn_status_label = QLabel("● Belum terhubung")
        self.conn_status_label.setStyleSheet("color: #b33; font-weight: bold;")
        layout.addWidget(self.conn_status_label)

        layout.addStretch(1)

        self.raw_checkbox = QCheckBox("Tampilkan raw JSON")
        self.raw_checkbox.stateChanged.connect(
            lambda _: self._log(f"Raw output: {'ON' if self.raw_checkbox.isChecked() else 'OFF'}")
        )
        layout.addWidget(self.raw_checkbox)

        return box

    def _build_config_group(self) -> QGroupBox:
        box = QGroupBox("Ubah konfigurasi AI (cmd)")
        outer = QVBoxLayout(box)
        form = QFormLayout()

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(0.75)
        form.addRow("conf:", self.threshold_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(0.45)
        form.addRow("nms:", self.iou_spin)

        self.classes_edit = QLineEdit("0,1,2,3")
        self.classes_edit.setPlaceholderText("contoh: 0,2,5 (kosongkan = tidak diubah)")
        self.classes_edit.textEdited.connect(self._on_classes_edit_manual)
        form.addRow("class:", self.classes_edit)

        outer.addLayout(form)

        # ---- Nama class dari file .names ----
        names_row = QHBoxLayout()
        self.load_names_btn = QPushButton("Load .names file...")
        self.load_names_btn.clicked.connect(self._on_load_names_clicked)
        names_row.addWidget(self.load_names_btn)
        self.names_path_label = QLabel("(belum ada file dimuat, isi class manual di atas)")
        self.names_path_label.setStyleSheet("color: #888;")
        self.names_path_label.setWordWrap(True)
        names_row.addWidget(self.names_path_label, stretch=1)
        outer.addLayout(names_row)

        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(140)
        self.class_list.setVisible(False)  # muncul setelah file .names di-load
        self.class_list.itemChanged.connect(self._on_class_item_changed)
        outer.addWidget(self.class_list)

        select_row = QHBoxLayout()
        self.select_all_btn = QPushButton("Pilih semua")
        self.select_all_btn.clicked.connect(lambda: self._set_all_class_checks(True))
        self.select_none_btn = QPushButton("Kosongkan")
        self.select_none_btn.clicked.connect(lambda: self._set_all_class_checks(False))
        select_row.addWidget(self.select_all_btn)
        select_row.addWidget(self.select_none_btn)
        select_row.addStretch(1)
        self.select_all_btn.setVisible(False)
        self.select_none_btn.setVisible(False)
        outer.addLayout(select_row)

        checks_row = QHBoxLayout()
        self.send_threshold_cb = QCheckBox("Kirim conf")
        self.send_threshold_cb.setChecked(True)
        self.send_iou_cb = QCheckBox("Kirim nms")
        self.send_iou_cb.setChecked(True)
        self.send_classes_cb = QCheckBox("Kirim class")
        self.send_classes_cb.setChecked(True)
        checks_row.addWidget(self.send_threshold_cb)
        checks_row.addWidget(self.send_iou_cb)
        checks_row.addWidget(self.send_classes_cb)
        checks_row.addStretch(1)
        outer.addLayout(checks_row)

        # ---- Toggle modul: detection & crowd counting & overlay ----
        modules_box = QGroupBox("Modul aktif (User)")
        modules_layout = QHBoxLayout(modules_box)

        self.detection_enabled_cb = QCheckBox("Detection")
        self.detection_enabled_cb.setChecked(True)
        self.detection_enabled_cb.setTristate(False)
        modules_layout.addWidget(self.detection_enabled_cb)

        self.crowd_enabled_cb = QCheckBox("Crowd Counting")
        self.crowd_enabled_cb.setChecked(True)
        self.crowd_enabled_cb.setTristate(False)
        modules_layout.addWidget(self.crowd_enabled_cb)

        self.overlay_enabled_cb = QCheckBox("Overlay")
        self.overlay_enabled_cb.setChecked(True)
        self.overlay_enabled_cb.setTristate(False)
        modules_layout.addWidget(self.overlay_enabled_cb)

        modules_layout.addStretch(1)
        outer.addWidget(modules_box)

        # ---- Golongan Semi-Dev: butuh restart aplikasi ----
        semidev_box = QGroupBox("Parameter Semi-Dev (butuh restart aplikasi)")
        semidev_form = QFormLayout(semidev_box)

        self.crowd_interval_cb = QCheckBox("crowd_interval:")
        self.crowd_interval_spin = QDoubleSpinBox()
        self.crowd_interval_spin.setDecimals(0)
        self.crowd_interval_spin.setRange(1, 60)
        self.crowd_interval_spin.setValue(5)
        row1 = QHBoxLayout()
        row1.addWidget(self.crowd_interval_cb)
        row1.addWidget(self.crowd_interval_spin)
        semidev_form.addRow(row1)

        self.crowd_width_cb = QCheckBox("crowd_width:")
        self.crowd_width_spin = QDoubleSpinBox()
        self.crowd_width_spin.setDecimals(0)
        self.crowd_width_spin.setRange(256, 2048)
        self.crowd_width_spin.setValue(1024)
        row2 = QHBoxLayout()
        row2.addWidget(self.crowd_width_cb)
        row2.addWidget(self.crowd_width_spin)
        semidev_form.addRow(row2)

        self.crowd_height_cb = QCheckBox("crowd_height:")
        self.crowd_height_spin = QDoubleSpinBox()
        self.crowd_height_spin.setDecimals(0)
        self.crowd_height_spin.setRange(256, 2048)
        self.crowd_height_spin.setValue(768)
        row3 = QHBoxLayout()
        row3.addWidget(self.crowd_height_cb)
        row3.addWidget(self.crowd_height_spin)
        semidev_form.addRow(row3)

        self.infer_size_cb = QCheckBox("infer_size:")
        self.infer_size_spin = QDoubleSpinBox()
        self.infer_size_spin.setDecimals(0)
        self.infer_size_spin.setRange(320, 1920)
        self.infer_size_spin.setValue(832)
        row4 = QHBoxLayout()
        row4.addWidget(self.infer_size_cb)
        row4.addWidget(self.infer_size_spin)
        semidev_form.addRow(row4)

        self.bitrate_cb = QCheckBox("bitrate_kbps:")
        self.bitrate_spin = QDoubleSpinBox()
        self.bitrate_spin.setDecimals(0)
        self.bitrate_spin.setRange(500, 20000)
        self.bitrate_spin.setValue(2000)
        row5 = QHBoxLayout()
        row5.addWidget(self.bitrate_cb)
        row5.addWidget(self.bitrate_spin)
        semidev_form.addRow(row5)

        self.reconnect_cb = QCheckBox("reconnect_ms:")
        self.reconnect_spin = QDoubleSpinBox()
        self.reconnect_spin.setDecimals(0)
        self.reconnect_spin.setRange(500, 60000)
        self.reconnect_spin.setValue(2000)
        row6 = QHBoxLayout()
        row6.addWidget(self.reconnect_cb)
        row6.addWidget(self.reconnect_spin)
        semidev_form.addRow(row6)

        outer.addWidget(semidev_box)

        self.send_btn = QPushButton("Kirim cmd")
        self.send_btn.clicked.connect(self._on_send_clicked)
        self.send_btn.setEnabled(False)
        outer.addWidget(self.send_btn)

        self.ack_label = QLabel("Belum ada command dikirim.")
        self.ack_label.setWordWrap(True)
        outer.addWidget(self.ack_label)

        outer.addStretch(1)
        return box

    def _build_status_group(self) -> QGroupBox:
        # Sumber data panel ini sekarang pesan "config" (bukan "ai_status").
        # "model" tidak lagi dikirim di protokol baru, jadi baris itu
        # dihapus. "fps" dipindah -- sekarang datang dari pesan "result"
        # (per-frame), bukan dari "config".
        box = QGroupBox("Status AI terkini (dari config)")
        form = QFormLayout(box)

        self.fps_label = QLabel("-")
        self.current_threshold_label = QLabel("-")
        self.iou_label = QLabel("-")
        self.active_classes_label = QLabel("-")
        self.active_classes_label.setWordWrap(True)
        self.detection_status_label = QLabel("-")
        self.crowd_status_label = QLabel("-")
        self.overlay_status_label = QLabel("-")

        form.addRow("FPS (dari result):", self.fps_label)
        form.addRow("conf aktif:", self.current_threshold_label)
        form.addRow("nms aktif:", self.iou_label)
        form.addRow("Class aktif:", self.active_classes_label)
        form.addRow("Detection:", self.detection_status_label)
        form.addRow("Crowd Counting:", self.crowd_status_label)
        form.addRow("Overlay:", self.overlay_status_label)

        return box

    def _build_detections_group(self) -> QGroupBox:
        box = QGroupBox("Hasil frame terakhir (result: deteksi + crowd)")
        layout = QVBoxLayout(box)

        self.frame_info_label = QLabel("Belum ada frame diterima.")
        layout.addWidget(self.frame_info_label)

        self.crowd_info_label = QLabel("Crowd counting: -")
        layout.addWidget(self.crowd_info_label)

        self.det_table = QTableWidget(0, 5)
        self.det_table.setHorizontalHeaderLabels(["ID", "Class", "Confidence", "X, Y", "W, H"])
        self.det_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.det_table.verticalHeader().setVisible(False)
        self.det_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.det_table)

        return box

    def _build_log_group(self) -> QGroupBox:
        box = QGroupBox("Log")
        layout = QVBoxLayout(box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        layout.addWidget(self.log_view)
        return box

    # ------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------
    def _on_connect_clicked(self):
        if self.ws_thread is not None:
            self._disconnect()
            return

        host = self.host_edit.text().strip()
        port = self.port_edit.text().strip()
        if not host or not port.isdigit():
            QMessageBox.warning(self, "Input tidak valid", "Host/port tidak valid.")
            return

        url = f"ws://{host}:{port}"
        self._log(f"Menghubungkan ke {url} ...")

        self.ws_thread = WsClientThread(url)
        self.ws_thread.connected.connect(self._on_ws_connected)
        self.ws_thread.disconnected.connect(self._on_ws_disconnected)
        self.ws_thread.message.connect(self._on_ws_message)
        self.ws_thread.raw_message.connect(self._on_ws_raw)
        self.ws_thread.error.connect(self._on_ws_error)
        self.ws_thread.start()

        self.connect_btn.setText("Disconnect")
        self.host_edit.setEnabled(False)
        self.port_edit.setEnabled(False)

    def _disconnect(self):
        if self.ws_thread is not None:
            self.ws_thread.stop()
            self.ws_thread = None
        self.connect_btn.setText("Connect")
        self.host_edit.setEnabled(True)
        self.port_edit.setEnabled(True)
        self.send_btn.setEnabled(False)
        self.conn_status_label.setText("● Terputus")
        self.conn_status_label.setStyleSheet("color: #b33; font-weight: bold;")

    def _on_ws_connected(self):
        self.conn_status_label.setText("● Terhubung")
        self.conn_status_label.setStyleSheet("color: #2a2; font-weight: bold;")
        self.send_btn.setEnabled(True)
        self._log("Terhubung ke server.")

    def _on_ws_disconnected(self, code: int, msg: str):
        self.conn_status_label.setText("● Terputus")
        self.conn_status_label.setStyleSheet("color: #b33; font-weight: bold;")
        self.send_btn.setEnabled(False)
        self._log(f"Koneksi ditutup (code={code}, msg={msg})")

    def _on_ws_error(self, err: str):
        self._log(f"Error: {err}")

    def _on_ws_raw(self, raw: str):
        if self.raw_checkbox.isChecked():
            self._log(f"[RAW] {raw}")

    # ------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------
    def _on_ws_message(self, msg: dict):
        # Pesan broadcast ("result"/"config") pakai key singkat "t".
        # Pesan command_ack (jalur cmd) masih pakai key "type" -- tidak
        # diubah, beda jalur dari broadcastResult/broadcastConfig.
        msg_type = msg.get("t") or msg.get("type", "unknown")
        self.msg_counts[msg_type] = self.msg_counts.get(msg_type, 0) + 1
        self._update_stats_label()

        if msg_type == "result":
            self._handle_result(msg)
        elif msg_type == "config":
            self._handle_config(msg)
        elif msg_type == "config_ack":
            self._handle_ack(msg)
        else:
            self._log(f"Tipe pesan tidak dikenal: {msg}")

    def _handle_result(self, msg: dict):
        """Tangani pesan 't':'result' -- deteksi + crowd counting + fps
        untuk satu frame. Lihat format lengkap di docstring modul."""
        now = time.time()
        interval_str = ""
        if self.last_result_ts is not None:
            dt = now - self.last_result_ts
            interval_str = f" (+{dt*1000:.0f}ms)"
        self.last_result_ts = now

        dets = msg.get("det", [])
        res = msg.get("res", {})
        infer_ms = msg.get("inf", 0)
        fps = msg.get("fps")
        crowd = msg.get("crwd", {})

        # fps sekarang datang dari "result", bukan "config"
        self.fps_label.setText(f"{fps:.1f}" if isinstance(fps, (int, float)) else "-")

        self.frame_info_label.setText(
            f"frame={msg.get('f')}  res={res.get('w')}x{res.get('h')}  "
            f"infer={infer_ms:.1f}ms  fps={fps if fps is not None else '-'}  "
            f"count={len(dets)}{interval_str}"
        )

        cnt = crowd.get("cnt")
        lvl = crowd.get("lvl")
        if cnt is not None or lvl is not None:
            self.crowd_info_label.setText(f"Crowd counting: count={cnt}  level={lvl}")
        else:
            self.crowd_info_label.setText("Crowd counting: -")

        self.det_table.setRowCount(len(dets))
        for row, d in enumerate(dets):
            bbox = d.get("bbox", [0, 0, 0, 0])  # array [x, y, w, h]
            x = bbox[0] if len(bbox) > 0 else 0
            y = bbox[1] if len(bbox) > 1 else 0
            w = bbox[2] if len(bbox) > 2 else 0
            h = bbox[3] if len(bbox) > 3 else 0
            values = [
                str(d.get("id", "")),
                str(d.get("cls", "")),
                f"{d.get('cf', 0):.2f}",
                f"{x:.3f}, {y:.3f}",
                f"{w:.3f}, {h:.3f}",
            ]
            for col, val in enumerate(values):
                self.det_table.setItem(row, col, QTableWidgetItem(val))

    def _handle_config(self, msg: dict):
        """Tangani pesan 't':'config' -- status konfigurasi AI saat ini.
        Menggantikan handler 'ai_status' lama; field 'model' sudah tidak
        ada di protokol baru, dan 'fps' sekarang berasal dari 'result'."""
        self.current_threshold_label.setText(str(msg.get("conf", "-")))
        self.iou_label.setText(str(msg.get("nms", "-")))

        classes = msg.get("cls", [])
        if classes:
            self.active_classes_label.setText(", ".join(self._class_label(c) for c in classes))
        else:
            self.active_classes_label.setText("(semua)")

        self._set_module_status_label(self.detection_status_label, msg.get("det_on"))
        self._set_module_status_label(self.crowd_status_label, msg.get("crwd_on"))
        self._set_module_status_label(self.overlay_status_label, msg.get("overlay"))

        # Sinkronkan checkbox toggle di panel config supaya merefleksikan
        # status aktual dari server, bukan cuma niat terakhir yang dikirim.
        if isinstance(msg.get("det_on"), bool):
            self.detection_enabled_cb.blockSignals(True)
            self.detection_enabled_cb.setChecked(msg["det_on"])
            self.detection_enabled_cb.blockSignals(False)
        if isinstance(msg.get("crwd_on"), bool):
            self.crowd_enabled_cb.blockSignals(True)
            self.crowd_enabled_cb.setChecked(msg["crwd_on"])
            self.crowd_enabled_cb.blockSignals(False)
        if isinstance(msg.get("overlay"), bool):
            self.overlay_enabled_cb.blockSignals(True)
            self.overlay_enabled_cb.setChecked(msg["overlay"])
            self.overlay_enabled_cb.blockSignals(False)

    def _set_module_status_label(self, label: QLabel, enabled):
        if enabled is True:
            label.setText("● Aktif")
            label.setStyleSheet("color: #2a2; font-weight: bold;")
        elif enabled is False:
            label.setText("○ Nonaktif")
            label.setStyleSheet("color: #b33; font-weight: bold;")
        else:
            label.setText("-")
            label.setStyleSheet("")

    def _handle_ack(self, msg: dict):
        # Tidak berubah -- config_ack masih pakai key "type" seperti semula.
        status = msg.get("status")
        color = "#2a2" if status == "applied" else "#b33"
        restart_note = ""
        if msg.get("requires_restart"):
            restart_note = "  ⚠ butuh restart aplikasi"
        reason = msg.get("reason")
        reason_note = f"  ({reason})" if reason else ""

        self.ack_label.setText(
            f"config_ack  command_id={msg.get('command_id')}  status={status}{restart_note}{reason_note}"
        )
        self.ack_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self._log(f"config_ack diterima: command_id={msg.get('command_id')} status={status}{restart_note}{reason_note}")

    # ------------------------------------------------------------
    # Kirim config_command
    # ------------------------------------------------------------
    def _on_send_clicked(self):
        # Tidak berubah -- format cmd/params masih sama dengan protokol asli.
        if self.ws_thread is None:
            return

        params = {}
        if self.send_threshold_cb.isChecked():
            params["conf"] = round(self.threshold_spin.value(), 2)
        if self.send_iou_cb.isChecked():
            params["nms"] = round(self.iou_spin.value(), 2)
        if self.send_classes_cb.isChecked():
            classes_text = self.classes_edit.text().strip()
            classes_list = [c.strip() for c in classes_text.split(",") if c.strip()]
            params["class"] = classes_list

        params["detect"] = self.detection_enabled_cb.isChecked()
        params["crowd"] = self.crowd_enabled_cb.isChecked()
        params["overlay"] = self.overlay_enabled_cb.isChecked()

        # ---- Golongan Semi-Dev: hanya dikirim kalau checkbox-nya dicentang ----
        if self.crowd_interval_cb.isChecked():
            params["crowd_interval"] = int(self.crowd_interval_spin.value())
        if self.crowd_width_cb.isChecked():
            params["crowd_width"] = int(self.crowd_width_spin.value())
        if self.crowd_height_cb.isChecked():
            params["crowd_height"] = int(self.crowd_height_spin.value())
        if self.infer_size_cb.isChecked():
            params["infer_size"] = int(self.infer_size_spin.value())
        if self.bitrate_cb.isChecked():
            params["bitrate_kbps"] = int(self.bitrate_spin.value())
        if self.reconnect_cb.isChecked():
            params["reconnect_ms"] = int(self.reconnect_spin.value())

        if not params:
            QMessageBox.information(self, "Tidak ada parameter",
                                    "Centang minimal satu parameter untuk dikirim.")
            return

        command_id = str(uuid.uuid4())
        payload = {
            "type": "cmd",
            "command_id": command_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": params,
        }

        ok = self.ws_thread.send(payload)
        if ok:
            self._log(f"-> Mengirim cmd: {json.dumps(params)}")
            self.ack_label.setText(f"Menunggu config_ack untuk command_id={command_id} ...")
            self.ack_label.setStyleSheet("color: #888;")
        else:
            self._log("Gagal mengirim: belum terhubung ke server.")

    # ------------------------------------------------------------
    # File .names (nama class)
    # ------------------------------------------------------------
    def _on_load_names_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih file .names", "",
            "Names files (*.names *.txt);;All files (*)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]
        except OSError as e:
            QMessageBox.warning(self, "Gagal membuka file", f"Tidak bisa membaca file:\n{e}")
            return

        if not names:
            QMessageBox.warning(self, "File kosong", "File .names tidak berisi nama class apapun.")
            return

        self.class_names = names
        self.names_path_label.setText(f"{path}  ({len(names)} class)")
        self.names_path_label.setStyleSheet("color: #2a2;")
        self._log(f"Berhasil memuat {len(names)} nama class dari: {path}")

        self._populate_class_list()

    def _populate_class_list(self):
        """Isi QListWidget dengan checkbox 'id: nama' berdasarkan class_names
        yang sudah di-load. Centang sesuai isi classes_edit saat ini kalau
        ada, kalau tidak semua tercentang secara default."""
        current_ids = self._parse_classes_text(self.classes_edit.text())

        self._syncing_class_list = True
        self.class_list.clear()
        for idx, name in enumerate(self.class_names):
            item = QListWidgetItem(f"{idx}: {name}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            checked = (not current_ids) or (str(idx) in current_ids)
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.class_list.addItem(item)
        self._syncing_class_list = False

        self.class_list.setVisible(True)
        self.select_all_btn.setVisible(True)
        self.select_none_btn.setVisible(True)
        self.classes_edit.setPlaceholderText("Diatur lewat checklist nama class di atas")
        self._sync_classes_edit_from_list()

    def _set_all_class_checks(self, checked: bool):
        self._syncing_class_list = True
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.class_list.count()):
            self.class_list.item(i).setCheckState(state)
        self._syncing_class_list = False
        self._sync_classes_edit_from_list()

    def _on_class_item_changed(self, _item: QListWidgetItem):
        if self._syncing_class_list:
            return
        self._sync_classes_edit_from_list()

    def _sync_classes_edit_from_list(self):
        """Checklist -> classes_edit (sumber kebenaran saat mengirim command)."""
        ids = [
            str(i) for i in range(self.class_list.count())
            if self.class_list.item(i).checkState() == Qt.Checked
        ]
        self._syncing_class_list = True
        self.classes_edit.setText(",".join(ids))
        self._syncing_class_list = False

    def _on_classes_edit_manual(self, _text: str):
        """Kalau user mengetik manual di classes_edit, sinkronkan balik ke
        checklist supaya keduanya tidak pernah berbeda."""
        if self._syncing_class_list or not self.class_names:
            return
        ids = self._parse_classes_text(self.classes_edit.text())
        self._syncing_class_list = True
        for i in range(self.class_list.count()):
            self.class_list.item(i).setCheckState(
                Qt.Checked if str(i) in ids else Qt.Unchecked
            )
        self._syncing_class_list = False

    @staticmethod
    def _parse_classes_text(text: str) -> set[str]:
        return {c.strip() for c in text.split(",") if c.strip()}

    def _class_label(self, class_id) -> str:
        """Format tampilan 'id' atau 'id:nama' kalau nama tersedia."""
        try:
            idx = int(class_id)
        except (TypeError, ValueError):
            return str(class_id)
        if 0 <= idx < len(self.class_names):
            return f"{idx}:{self.class_names[idx]}"
        return str(idx)

    # ------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------
    def _log(self, text: str):
        self.log_view.appendPlainText(f"[{now_str()}] {text}")

    def _update_stats_label(self):
        parts = [f"{k}: {v}" for k, v in sorted(self.msg_counts.items(), key=lambda x: -x[1])]
        self.stats_label.setText("  |  ".join(parts) if parts else "Belum ada pesan diterima")

    def closeEvent(self, event):
        if self.ws_thread is not None:
            self.ws_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()