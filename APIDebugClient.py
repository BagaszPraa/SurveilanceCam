#!/usr/bin/env python3
"""
api_debug_client.py — Debug client untuk APIController (drone/edge AI WebSocket server)

Fungsi:
  - Connect ke APIController dan tampilkan live stream detection_result & ai_status
    dengan format rapi di terminal
  - Kirim config_command (threshold, class filter) baik lewat argumen CLI (one-shot)
    maupun mode interaktif (REPL)
  - Hitung statistik ringan: jumlah pesan per tipe, rate pesan/detik

Instalasi dependency:
    pip install websocket-client

Contoh pemakaian:
    # Listen only (paling umum untuk debug live stream)
    python3 api_debug_client.py --host 127.0.0.1 --port 8765

    # Kirim command sekali lalu tetap listen
    python3 api_debug_client.py --threshold 0.6 --classes 0,2

    # Kirim command sekali lalu langsung keluar (cocok untuk script/CI)
    python3 api_debug_client.py --threshold 0.6 --classes 0,2 --once

    # Mode interaktif: ketik command sambil lihat stream live
    python3 api_debug_client.py --interactive
"""

import argparse
import json
import sys
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone

try:
    import websocket
except ImportError:
    print("Library 'websocket-client' belum terinstall.")
    print("Install dulu: pip install websocket-client")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Warna terminal ringan (ANSI), tanpa dependency tambahan.
# Otomatis nonaktif kalau output bukan TTY (misal di-redirect ke file).
# ---------------------------------------------------------------------------
class C:
    ENABLED = sys.stdout.isatty()
    RESET = "\033[0m" if ENABLED else ""
    DIM = "\033[2m" if ENABLED else ""
    BOLD = "\033[1m" if ENABLED else ""
    GREEN = "\033[32m" if ENABLED else ""
    YELLOW = "\033[33m" if ENABLED else ""
    RED = "\033[31m" if ENABLED else ""
    CYAN = "\033[36m" if ENABLED else ""
    MAGENTA = "\033[35m" if ENABLED else ""


def now_str():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


class ApiDebugClient:
    def __init__(self, host: str, port: int, verbose_raw: bool = False):
        self.url = f"ws://{host}:{port}"
        self.verbose_raw = verbose_raw
        self.ws = None
        self.connected = threading.Event()
        self.stop_flag = threading.Event()

        # Statistik ringan
        self.msg_counts = defaultdict(int)
        self.last_detection_ts = None
        self.pending_acks = {}  # command_id -> event, untuk mode --once

    # ---------------- Connection lifecycle ----------------
    def connect(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        thread = threading.Thread(target=self._run_forever, daemon=True)
        thread.start()
        return thread

    def _run_forever(self):
        # reconnect_delay bawaan run_forever mendukung retry otomatis
        # kalau koneksi putus (drone reboot, link radio tidak stabil, dll)
        while not self.stop_flag.is_set():
            self.connected.clear()
            self.ws.run_forever(reconnect=3)
            if self.stop_flag.is_set():
                break
            print(f"{C.YELLOW}[{now_str()}] Koneksi putus, mencoba reconnect...{C.RESET}")
            time.sleep(1)

    def _on_open(self, ws):
        self.connected.set()
        print(f"{C.GREEN}[{now_str()}] ✓ Terhubung ke {self.url}{C.RESET}")

    def _on_close(self, ws, code, msg):
        self.connected.clear()
        print(f"{C.YELLOW}[{now_str()}] Koneksi ditutup (code={code}, msg={msg}){C.RESET}")

    def _on_error(self, ws, error):
        print(f"{C.RED}[{now_str()}] Error: {error}{C.RESET}")

    # ---------------- Message dispatch ----------------
    def _on_message(self, ws, raw_message):
        if self.verbose_raw:
            print(f"{C.DIM}[RAW] {raw_message}{C.RESET}")

        try:
            msg = json.loads(raw_message)
        except json.JSONDecodeError:
            print(f"{C.RED}[{now_str()}] Gagal parse JSON: {raw_message[:200]}{C.RESET}")
            return

        msg_type = msg.get("type", "unknown")
        self.msg_counts[msg_type] += 1

        handler = {
            "detection_result": self._print_detection,
            "ai_status": self._print_status,
            "config_ack": self._print_ack,
        }.get(msg_type, self._print_unknown)

        handler(msg)

    # ---------------- Formatter tiap tipe pesan ----------------
    def _print_detection(self, msg):
        now = time.time()
        interval_str = ""
        if self.last_detection_ts is not None:
            dt = now - self.last_detection_ts
            interval_str = f" (+{dt*1000:.0f}ms)"
        self.last_detection_ts = now

        dets = msg.get("detections", [])
        res = msg.get("resolution", {})
        infer_ms = msg.get("inference_time_ms", 0)

        header = (f"{C.CYAN}[{now_str()}] detection_result{C.RESET}"
                  f" frame={msg.get('frame_id')} res={res.get('width')}x{res.get('height')}"
                  f" infer={infer_ms:.1f}ms count={len(dets)}{interval_str}")
        print(header)

        for d in dets:
            bbox = d.get("bbox", {})
            print(f"    {C.BOLD}{d.get('class'):<10}{C.RESET} "
                  f"conf={d.get('confidence'):.2f}  "
                  f"bbox=(x={bbox.get('x'):.3f}, y={bbox.get('y'):.3f}, "
                  f"w={bbox.get('w'):.3f}, h={bbox.get('h'):.3f})  "
                  f"id={d.get('id')}")

    def _print_status(self, msg):
        classes = ", ".join(msg.get("active_classes", []))
        print(f"{C.MAGENTA}[{now_str()}] ai_status{C.RESET} "
              f"model={msg.get('model')} fps={msg.get('fps'):.1f} "
              f"threshold={msg.get('current_threshold')} classes=[{classes}]")

    def _print_ack(self, msg):
        status = msg.get("status")
        color = C.GREEN if status == "applied" else C.RED
        print(f"{color}[{now_str()}] config_ack{C.RESET} "
              f"command_id={msg.get('command_id')} status={status}")

        cmd_id = msg.get("command_id")
        if cmd_id in self.pending_acks:
            self.pending_acks[cmd_id].set()

    def _print_unknown(self, msg):
        print(f"{C.YELLOW}[{now_str()}] Tipe pesan tidak dikenal: {msg}{C.RESET}")

    # ---------------- Kirim command ----------------
    def send_config_command(self, threshold=None, classes=None, iou=None, wait_ack=False, timeout=3.0):
        if not self.connected.is_set():
            print(f"{C.RED}Belum terhubung ke server, command tidak dikirim.{C.RESET}")
            return False

        params = {}
        if threshold is not None:
            params["confidence_threshold"] = threshold
        if classes is not None:
            params["classes_enabled"] = classes
        if iou is not None:
            params["iou_threshold"] = iou

        if not params:
            print(f"{C.YELLOW}Tidak ada parameter untuk dikirim.{C.RESET}")
            return False

        command_id = str(uuid.uuid4())
        payload = {
            "type": "config_command",
            "command_id": command_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": params,
        }

        event = threading.Event()
        self.pending_acks[command_id] = event

        print(f"{C.CYAN}[{now_str()}] -> Mengirim config_command: {json.dumps(params)}{C.RESET}")
        self.ws.send(json.dumps(payload))

        if wait_ack:
            if not event.wait(timeout=timeout):
                print(f"{C.RED}Timeout menunggu config_ack ({timeout}s).{C.RESET}")
                return False
        return True

    def print_stats(self):
        print(f"\n{C.BOLD}=== Statistik Pesan ==={C.RESET}")
        if not self.msg_counts:
            print("  (belum ada pesan diterima)")
        for k, v in sorted(self.msg_counts.items(), key=lambda x: -x[1]):
            print(f"  {k:<20} {v}")

    def close(self):
        self.stop_flag.set()
        if self.ws:
            self.ws.close()


def parse_classes(value: str):
    """'0,2,5' -> ['0','2','5'] (dikirim sebagai list of string, sesuai
    konvensi APIController: class ID dalam bentuk string)."""
    return [c.strip() for c in value.split(",") if c.strip()]


def interactive_loop(client: ApiDebugClient):
    print(f"\n{C.BOLD}Mode interaktif aktif.{C.RESET} Perintah yang tersedia:")
    print("  threshold <nilai>        contoh: threshold 0.6")
    print("  classes <id,id,...>      contoh: classes 0,2")
    print("  iou <nilai>              contoh: iou 0.5")
    print("  set <threshold> <kelas>  contoh: set 0.6 0,2")
    print("  stats                    tampilkan statistik pesan")
    print("  raw on|off               toggle tampilan pesan JSON mentah")
    print("  quit / exit              keluar\n")

    while True:
        try:
            line = input(f"{C.BOLD}> {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit"):
            break
        elif cmd == "threshold" and len(parts) == 2:
            client.send_config_command(threshold=float(parts[1]))
        elif cmd == "classes" and len(parts) == 2:
            client.send_config_command(classes=parse_classes(parts[1]))
        elif cmd == "iou" and len(parts) == 2:
            client.send_config_command(iou=float(parts[1]))
        elif cmd == "set" and len(parts) == 3:
            client.send_config_command(threshold=float(parts[1]), classes=parse_classes(parts[2]))
        elif cmd == "stats":
            client.print_stats()
        elif cmd == "raw" and len(parts) == 2:
            client.verbose_raw = parts[1].lower() == "on"
            print(f"Raw output: {'ON' if client.verbose_raw else 'OFF'}")
        else:
            print(f"{C.YELLOW}Perintah tidak dikenal atau argumen salah: {line}{C.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug client untuk APIController (drone AI WebSocket server)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Alamat host drone/edge (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port APIController (default: 8765)")
    parser.add_argument("--threshold", type=float, default=None, help="Set confidence_threshold saat start")
    parser.add_argument("--classes", type=str, default=None, help="Set classes_enabled saat start, contoh: 0,2,5")
    parser.add_argument("--iou", type=float, default=None, help="Set iou_threshold saat start")
    parser.add_argument("--once", action="store_true",
                         help="Kirim command (kalau ada) lalu keluar setelah dapat ack, tanpa listen terus")
    parser.add_argument("--interactive", action="store_true", help="Masuk mode REPL interaktif")
    parser.add_argument("--raw", action="store_true", help="Tampilkan juga pesan JSON mentah")
    args = parser.parse_args()

    client = ApiDebugClient(args.host, args.port, verbose_raw=args.raw)
    client.connect()

    print(f"{C.DIM}Menghubungkan ke {client.url} ...{C.RESET}")
    if not client.connected.wait(timeout=5.0):
        print(f"{C.RED}Gagal terhubung dalam 5 detik. Cek host/port dan apakah server jalan.{C.RESET}")
        sys.exit(1)

    classes_list = parse_classes(args.classes) if args.classes else None
    has_command = args.threshold is not None or classes_list is not None or args.iou is not None

    if has_command:
        client.send_config_command(
            threshold=args.threshold, classes=classes_list, iou=args.iou,
            wait_ack=True, timeout=3.0,
        )

    if args.once:
        client.close()
        return

    if args.interactive:
        interactive_loop(client)
        client.close()
        return

    # Default: mode listen-only, tampilkan stream sampai Ctrl+C
    print(f"{C.DIM}Listening... (Ctrl+C untuk berhenti){C.RESET}\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        client.print_stats()
        client.close()


if __name__ == "__main__":
    main()