#!/usr/bin/env python3
"""
APIDebugGUI.py — GUI debug client (PySide6) untuk APIController
(drone/edge AI WebSocket server).

Fungsi:
  - Connect/disconnect ke APIController lewat GUI (host & port bisa diubah).
  - Live view detection_result & ai_status di panel terpisah.
  - Ubah semua parameter yang didukung config_command:
      confidence_threshold, iou_threshold, classes_enabled.
  - Lihat config_ack (applied/rejected) langsung di panel status.
  - Log mentah (raw JSON) opsional, plus statistik jumlah pesan per tipe.

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
        self.last_detection_ts: float | None = None

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
        box = QGroupBox("Ubah konfigurasi AI (config_command)")
        outer = QVBoxLayout(box)
        form = QFormLayout()

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(0.75)
        form.addRow("confidence_threshold:", self.threshold_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(0.45)
        form.addRow("iou_threshold:", self.iou_spin)

        self.classes_edit = QLineEdit("0,1,2,3")
        self.classes_edit.setPlaceholderText("contoh: 0,2,5 (kosongkan = tidak diubah)")
        self.classes_edit.textEdited.connect(self._on_classes_edit_manual)
        form.addRow("classes_enabled:", self.classes_edit)

        outer.addLayout(form)

        # ---- Nama class dari file .names ----
        names_row = QHBoxLayout()
        self.load_names_btn = QPushButton("Load .names file...")
        self.load_names_btn.clicked.connect(self._on_load_names_clicked)
        names_row.addWidget(self.load_names_btn)
        self.names_path_label = QLabel("(belum ada file dimuat, isi classes_enabled manual di atas)")
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
        self.send_threshold_cb = QCheckBox("Kirim threshold")
        self.send_threshold_cb.setChecked(True)
        self.send_iou_cb = QCheckBox("Kirim iou")
        self.send_iou_cb.setChecked(True)
        self.send_classes_cb = QCheckBox("Kirim classes")
        self.send_classes_cb.setChecked(True)
        checks_row.addWidget(self.send_threshold_cb)
        checks_row.addWidget(self.send_iou_cb)
        checks_row.addWidget(self.send_classes_cb)
        checks_row.addStretch(1)
        outer.addLayout(checks_row)

        # ---- Toggle modul: detection & crowd counting ----
        modules_box = QGroupBox("Modul aktif")
        modules_layout = QHBoxLayout(modules_box)

        self.detection_enabled_cb = QCheckBox("Detection")
        self.detection_enabled_cb.setChecked(True)
        self.detection_enabled_cb.setTristate(False)
        modules_layout.addWidget(self.detection_enabled_cb)

        self.crowd_enabled_cb = QCheckBox("Crowd Counting")
        self.crowd_enabled_cb.setChecked(True)
        self.crowd_enabled_cb.setTristate(False)
        modules_layout.addWidget(self.crowd_enabled_cb)

        modules_layout.addStretch(1)
        outer.addWidget(modules_box)

        self.send_btn = QPushButton("Kirim config_command")
        self.send_btn.clicked.connect(self._on_send_clicked)
        self.send_btn.setEnabled(False)
        outer.addWidget(self.send_btn)

        self.ack_label = QLabel("Belum ada command dikirim.")
        self.ack_label.setWordWrap(True)
        outer.addWidget(self.ack_label)

        outer.addStretch(1)
        return box

    def _build_status_group(self) -> QGroupBox:
        box = QGroupBox("Status AI terkini (dari ai_status)")
        form = QFormLayout(box)

        self.model_label = QLabel("-")
        self.fps_label = QLabel("-")
        self.current_threshold_label = QLabel("-")
        self.active_classes_label = QLabel("-")
        self.active_classes_label.setWordWrap(True)
        self.detection_status_label = QLabel("-")
        self.crowd_status_label = QLabel("-")

        form.addRow("Model:", self.model_label)
        form.addRow("FPS:", self.fps_label)
        form.addRow("Threshold aktif:", self.current_threshold_label)
        form.addRow("Class aktif:", self.active_classes_label)
        form.addRow("Detection:", self.detection_status_label)
        form.addRow("Crowd Counting:", self.crowd_status_label)

        return box

    def _build_detections_group(self) -> QGroupBox:
        box = QGroupBox("Deteksi terakhir (detection_result)")
        layout = QVBoxLayout(box)

        self.frame_info_label = QLabel("Belum ada frame diterima.")
        layout.addWidget(self.frame_info_label)

        self.det_table = QTableWidget(0, 5)
        self.det_table.setHorizontalHeaderLabels(["Class", "Confidence", "X", "Y", "W/H"])
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
        msg_type = msg.get("type", "unknown")
        self.msg_counts[msg_type] = self.msg_counts.get(msg_type, 0) + 1
        self._update_stats_label()

        if msg_type == "detection_result":
            self._handle_detection(msg)
        elif msg_type == "ai_status":
            self._handle_status(msg)
        elif msg_type == "config_ack":
            self._handle_ack(msg)
        else:
            self._log(f"Tipe pesan tidak dikenal: {msg}")

    def _handle_detection(self, msg: dict):
        now = time.time()
        interval_str = ""
        if self.last_detection_ts is not None:
            dt = now - self.last_detection_ts
            interval_str = f" (+{dt*1000:.0f}ms)"
        self.last_detection_ts = now

        dets = msg.get("detections", [])
        res = msg.get("resolution", {})
        infer_ms = msg.get("inference_time_ms", 0)

        self.frame_info_label.setText(
            f"frame={msg.get('frame_id')}  res={res.get('width')}x{res.get('height')}  "
            f"infer={infer_ms:.1f}ms  count={len(dets)}{interval_str}"
        )

        self.det_table.setRowCount(len(dets))
        for row, d in enumerate(dets):
            bbox = d.get("bbox", {})
            values = [
                str(d.get("class", "")),
                f"{d.get('confidence', 0):.2f}",
                f"{bbox.get('x', 0):.3f}",
                f"{bbox.get('y', 0):.3f}",
                f"{bbox.get('w', 0):.3f} / {bbox.get('h', 0):.3f}",
            ]
            for col, val in enumerate(values):
                self.det_table.setItem(row, col, QTableWidgetItem(val))

    def _handle_status(self, msg: dict):
        self.model_label.setText(str(msg.get("model", "-")))
        fps = msg.get("fps")
        self.fps_label.setText(f"{fps:.1f}" if isinstance(fps, (int, float)) else "-")
        self.current_threshold_label.setText(str(msg.get("current_threshold", "-")))
        classes = msg.get("active_classes", [])
        if classes:
            self.active_classes_label.setText(", ".join(self._class_label(c) for c in classes))
        else:
            self.active_classes_label.setText("(semua)")

        self._set_module_status_label(self.detection_status_label, msg.get("detection_enabled"))
        self._set_module_status_label(self.crowd_status_label, msg.get("crowd_counting_enabled"))

        # Sinkronkan checkbox toggle di panel config supaya merefleksikan
        # status aktual dari server, bukan cuma niat terakhir yang dikirim.
        if isinstance(msg.get("detection_enabled"), bool):
            self.detection_enabled_cb.blockSignals(True)
            self.detection_enabled_cb.setChecked(msg["detection_enabled"])
            self.detection_enabled_cb.blockSignals(False)
        if isinstance(msg.get("crowd_counting_enabled"), bool):
            self.crowd_enabled_cb.blockSignals(True)
            self.crowd_enabled_cb.setChecked(msg["crowd_counting_enabled"])
            self.crowd_enabled_cb.blockSignals(False)

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
        status = msg.get("status")
        color = "#2a2" if status == "applied" else "#b33"
        self.ack_label.setText(
            f"config_ack  command_id={msg.get('command_id')}  status={status}"
        )
        self.ack_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self._log(f"config_ack diterima: command_id={msg.get('command_id')} status={status}")

    # ------------------------------------------------------------
    # Kirim config_command
    # ------------------------------------------------------------
    def _on_send_clicked(self):
        if self.ws_thread is None:
            return

        params = {}
        if self.send_threshold_cb.isChecked():
            params["confidence_threshold"] = round(self.threshold_spin.value(), 2)
        if self.send_iou_cb.isChecked():
            params["iou_threshold"] = round(self.iou_spin.value(), 2)
        if self.send_classes_cb.isChecked():
            classes_text = self.classes_edit.text().strip()
            classes_list = [c.strip() for c in classes_text.split(",") if c.strip()]
            params["classes_enabled"] = classes_list

        params["detection_enabled"] = self.detection_enabled_cb.isChecked()
        params["crowd_counting_enabled"] = self.crowd_enabled_cb.isChecked()

        if not params:
            QMessageBox.information(self, "Tidak ada parameter",
                                    "Centang minimal satu parameter untuk dikirim.")
            return

        command_id = str(uuid.uuid4())
        payload = {
            "type": "config_command",
            "command_id": command_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": params,
        }

        ok = self.ws_thread.send(payload)
        if ok:
            self._log(f"-> Mengirim config_command: {json.dumps(params)}")
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