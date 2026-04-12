from __future__ import annotations

import hashlib
import json
import queue
import socket
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


APP_NAME = "MTS CHAT WEB 2.5"
APP_VERSION = "1.488"
DEFAULT_TCP_PORT = 47000
DEFAULT_DISCOVERY_PORT = 47001
DISCOVERY_INTERVAL = 3.0
PEER_TTL = 12.0
MAX_MESSAGE_LEN = 2000
BUFFER_SIZE = 65535


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def format_mac(raw_mac: str) -> str:
    cleaned = "".join(ch for ch in raw_mac if ch.isalnum()).upper()
    if len(cleaned) != 12 or any(ch not in "0123456789ABCDEF" for ch in cleaned):
        raise ValueError("Неверный MAC-адрес. Используй формат AA:BB:CC:DD:EE:FF")
    return ":".join(cleaned[index : index + 2] for index in range(0, 12, 2))


def compact_mac(mac: str) -> str:
    return mac.replace(":", "")


def local_mac() -> str:
    return format_mac(f"{uuid.getnode():012X}")


def detect_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 1))
        ip = sock.getsockname()[0]
    except OSError:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            ip = "127.0.0.1"
    finally:
        sock.close()
    return ip


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def pretty_age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def short_hash(value: str, length: int = 10) -> str:
    return value[:length]


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def enable_port_reuse(sock: socket.socket) -> None:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    reuse_port = getattr(socket, "SO_REUSEPORT", None)
    if reuse_port is not None:
        try:
            sock.setsockopt(socket.SOL_SOCKET, reuse_port, 1)
        except OSError:
            pass


@dataclass(slots=True)
class Peer:
    mac: str
    name: str
    ip: str
    port: int
    session_id: str
    last_seen: float

    @property
    def alive(self) -> bool:
        return (time.time() - self.last_seen) <= PEER_TTL

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["alive"] = self.alive
        payload["age"] = pretty_age(time.time() - self.last_seen)
        return payload


class LanChatMessenger:
    def __init__(
        self,
        *,
        name: str,
        tcp_port: int = DEFAULT_TCP_PORT,
        discovery_port: int = DEFAULT_DISCOVERY_PORT,
        mac_override: str | None = None,
    ) -> None:
        self.name = name.strip() or socket.gethostname()
        self.mac = format_mac(mac_override) if mac_override else local_mac()
        self.ip = detect_local_ip()
        self.tcp_port = tcp_port
        self.discovery_port = discovery_port
        seed = f"{self.mac}|{self.name}|{self.ip}|{time.time_ns()}"
        self.session_id = sha256_text(seed)[:16]

        self.stop_event = threading.Event()
        self.peer_lock = threading.Lock()
        self.ledger_lock = threading.Lock()
        self.event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.peers: dict[str, Peer] = {}
        self.ledger: list[dict[str, Any]] = []
        self.ledger_tail = "GENESIS"
        self._threads: list[threading.Thread] = []
        self._tcp_socket: socket.socket | None = None
        self._discovery_socket: socket.socket | None = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running and not self.stop_event.is_set()

    def start(self) -> None:
        if self.running:
            return
        self.stop_event.clear()
        self._open_server_sockets()
        self._threads = []
        for target, name in (
            (self._tcp_server_loop, "lan-chat-tcp-server"),
            (self._discovery_listener_loop, "lan-chat-udp-discovery"),
            (self._discovery_broadcast_loop, "lan-chat-udp-broadcast"),
            (self._peer_prune_loop, "lan-chat-peer-prune"),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        self._running = True
        self._emit(
            "status",
            text=f"Локальный чат запущен: {self.name} {self.mac} @ {self.ip}:{self.tcp_port}",
        )
        self.broadcast_discovery()

    def stop(self) -> None:
        if not self._running and not self._tcp_socket and not self._discovery_socket:
            return
        self.stop_event.set()
        self._close_socket(self._tcp_socket)
        self._close_socket(self._discovery_socket)
        self._tcp_socket = None
        self._discovery_socket = None
        for thread in self._threads:
            thread.join(timeout=0.2)
        self._threads = []
        self._running = False
        self._emit("status", text="Локальный чат остановлен.")

    def get_self_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mac": self.mac,
            "ip": self.ip,
            "tcp_port": self.tcp_port,
            "discovery_port": self.discovery_port,
            "session_id": self.session_id,
        }

    def get_peers(self) -> list[Peer]:
        with self.peer_lock:
            peers = list(self.peers.values())
        return sorted(peers, key=lambda peer: (not peer.alive, peer.name.lower(), peer.mac))

    def get_history(self, peer_mac: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self.ledger_lock:
            ledger = list(self.ledger)
        if peer_mac:
            ledger = [item for item in ledger if item["event"].get("peer_mac") == peer_mac]
        return ledger[-max(1, limit) :]

    def drain_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while True:
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def broadcast_discovery(self) -> None:
        self._send_announce("255.255.255.255", broadcast=True)
        self._emit("status", text="Discovery-пакет отправлен в локальную сеть.")

    def manual_link(self, mac: str, ip: str, port: int, name: str = "") -> Peer:
        peer_mac = format_mac(mac)
        if peer_mac == self.mac:
            raise ValueError("Нельзя привязать самого себя.")
        peer_name = name.strip() or f"manual-{peer_mac[-5:].replace(':', '')}"
        self._upsert_peer(peer_mac, peer_name, ip.strip(), port, "manual")
        peer = self.peers[peer_mac]
        self._emit(
            "status",
            text=f"Пир сохранён: {peer.mac} -> {peer.ip}:{peer.port} ({peer.name})",
        )
        return peer

    def send_message(self, peer_mac_query: str, text: str) -> dict[str, Any]:
        peer_mac = self._resolve_mac_query(peer_mac_query)
        return self._send_to_peer(peer_mac, text)

    def broadcast_message(self, text: str) -> int:
        text = text.strip()
        if not text:
            raise ValueError("Сообщение не должно быть пустым.")
        with self.peer_lock:
            peers = [peer for peer in self.peers.values() if peer.alive and peer.mac != self.mac]
        if not peers:
            raise ValueError("Нет активных пиров для broadcast.")
        sent = 0
        for peer in peers:
            try:
                self._send_to_peer(peer.mac, text)
                sent += 1
            except Exception as exc:  # noqa: BLE001
                self._emit("error", text=f"Не удалось отправить {peer.mac}: {exc}")
        self._emit("status", text=f"Broadcast завершён. Успешно: {sent}/{len(peers)}")
        return sent

    def _emit(self, event_type: str, **payload: Any) -> None:
        self.event_queue.put({"type": event_type, **payload})

    def _open_server_sockets(self) -> None:
        discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            enable_port_reuse(discovery_socket)
            discovery_socket.bind(("", self.discovery_port))
            discovery_socket.settimeout(1.0)
        except OSError as exc:
            discovery_socket.close()
            raise RuntimeError(f"Не удалось открыть UDP discovery-порт {self.discovery_port}: {exc}") from exc

        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            tcp_socket.bind(("", self.tcp_port))
            tcp_socket.listen()
            tcp_socket.settimeout(1.0)
        except OSError as exc:
            discovery_socket.close()
            tcp_socket.close()
            raise RuntimeError(f"Не удалось открыть TCP-порт {self.tcp_port}: {exc}") from exc

        self._discovery_socket = discovery_socket
        self._tcp_socket = tcp_socket

    def _close_socket(self, sock: socket.socket | None) -> None:
        if sock is None:
            return
        try:
            sock.close()
        except OSError:
            pass

    def _append_ledger_event(self, event: dict[str, Any]) -> dict[str, Any]:
        with self.ledger_lock:
            payload = json.dumps(event, sort_keys=True, ensure_ascii=False)
            block_hash = sha256_text(f"{self.ledger_tail}|{payload}")
            record = {
                "index": len(self.ledger) + 1,
                "prev_hash": self.ledger_tail,
                "block_hash": block_hash,
                "event": event,
            }
            self.ledger.append(record)
            self.ledger_tail = block_hash
        self._emit("ledger_updated", record=record)
        return record

    def _peer_prune_loop(self) -> None:
        while not self.stop_event.wait(5.0):
            with self.peer_lock:
                stale = [mac for mac, peer in self.peers.items() if (time.time() - peer.last_seen) > (PEER_TTL * 4)]
                for mac in stale:
                    self.peers.pop(mac, None)
            if stale:
                self._emit("peers_updated")

    def _resolve_mac_query(self, query: str) -> str:
        cleaned = "".join(ch for ch in query if ch.isalnum()).upper()
        if not cleaned:
            raise ValueError("Пустой MAC-запрос.")
        if len(cleaned) == 12:
            target = format_mac(cleaned)
            if target == self.mac:
                raise ValueError("Нельзя отправить сообщение самому себе.")
            return target
        with self.peer_lock:
            matches = [mac for mac in self.peers if compact_mac(mac).startswith(cleaned)]
        if not matches:
            raise ValueError("Такой MAC не найден. Сначала дождитесь discovery или добавьте пира вручную.")
        if len(matches) > 1:
            raise ValueError(f"MAC-префикс неоднозначен: {', '.join(matches)}")
        if matches[0] == self.mac:
            raise ValueError("Нельзя отправить сообщение самому себе.")
        return matches[0]

    def _upsert_peer(self, mac: str, name: str, ip: str, port: int, session_id: str) -> bool:
        if mac == self.mac:
            return False
        with self.peer_lock:
            existing = self.peers.get(mac)
            is_new = existing is None
            self.peers[mac] = Peer(
                mac=mac,
                name=name or (existing.name if existing else "unknown"),
                ip=ip or (existing.ip if existing else "0.0.0.0"),
                port=port or (existing.port if existing else self.tcp_port),
                session_id=session_id or (existing.session_id if existing else ""),
                last_seen=time.time(),
            )
        self._emit("peers_updated", mac=mac, is_new=is_new)
        return is_new

    def _discovery_listener_loop(self) -> None:
        sock = self._discovery_socket
        if sock is None:
            return
        while not self.stop_event.is_set():
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                payload = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if payload.get("type") != "announce":
                continue
            try:
                mac = format_mac(str(payload.get("mac", "")))
            except ValueError:
                continue
            peer_ip = str(payload.get("ip") or addr[0])
            peer_port = safe_int(payload.get("port"), DEFAULT_TCP_PORT)
            peer_name = str(payload.get("name") or "unknown")
            session_id = str(payload.get("session_id") or "")
            is_new = self._upsert_peer(mac, peer_name, peer_ip, peer_port, session_id)
            if is_new:
                self._emit("status", text=f"Новый пир: {peer_name} {mac} @ {peer_ip}:{peer_port}")
                self._send_announce(addr[0], broadcast=False)

    def _discovery_broadcast_loop(self) -> None:
        while not self.stop_event.wait(DISCOVERY_INTERVAL):
            self._send_announce("255.255.255.255", broadcast=True)

    def _announce_payload(self) -> dict[str, Any]:
        return {
            "type": "announce",
            "app": APP_NAME,
            "version": APP_VERSION,
            "name": self.name,
            "mac": self.mac,
            "ip": self.ip,
            "port": self.tcp_port,
            "session_id": self.session_id,
            "timestamp": now_iso(),
        }

    def _send_announce(self, target_ip: str, *, broadcast: bool) -> None:
        payload = json.dumps(self._announce_payload(), ensure_ascii=False).encode("utf-8")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            enable_port_reuse(sock)
            if broadcast:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(payload, (target_ip, self.discovery_port))
        except OSError:
            pass
        finally:
            sock.close()

    def _tcp_server_loop(self) -> None:
        sock = self._tcp_socket
        if sock is None:
            return
        while not self.stop_event.is_set():
            try:
                conn, addr = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True)
            thread.start()

    def _handle_client(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        conn.settimeout(3.0)
        chunks: list[bytes] = []
        try:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
            if not chunks:
                return
            raw_line = b"".join(chunks).splitlines()[0].decode("utf-8")
            payload = json.loads(raw_line)
            if payload.get("type") != "chat":
                return
            peer_mac = format_mac(str(payload.get("from_mac", "")))
            if peer_mac == self.mac:
                return
            peer_name = str(payload.get("from_name") or "unknown")
            peer_port = safe_int(payload.get("reply_port"), DEFAULT_TCP_PORT)
            session_id = str(payload.get("session_id") or "")
            text = str(payload.get("text") or "").strip()
            sent_at = str(payload.get("timestamp") or now_iso())
            if not text:
                raise ValueError("Пустое сообщение")
            if len(text) > MAX_MESSAGE_LEN:
                raise ValueError("Слишком длинное сообщение")
            self._upsert_peer(peer_mac, peer_name, addr[0], peer_port, session_id)
            record = self._append_ledger_event(
                {
                    "direction": "in",
                    "peer_mac": peer_mac,
                    "peer_name": peer_name,
                    "peer_ip": addr[0],
                    "timestamp": sent_at,
                    "text": text,
                    "message_id": str(payload.get("message_id") or ""),
                }
            )
            response = {
                "status": "ok",
                "received_at": now_iso(),
                "block_hash": record["block_hash"],
            }
            conn.sendall((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
            self._emit(
                "message_received",
                peer_mac=peer_mac,
                peer_name=peer_name,
                text=text,
                timestamp=sent_at,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            try:
                conn.sendall(b'{"status":"error"}\n')
            except OSError:
                pass
        finally:
            conn.close()

    def _send_to_peer(self, peer_mac: str, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("Пустое сообщение отправлять нельзя.")
        if len(text) > MAX_MESSAGE_LEN:
            raise ValueError(f"Сообщение длиннее {MAX_MESSAGE_LEN} символов.")
        with self.peer_lock:
            peer = self.peers.get(peer_mac)
        if not peer:
            raise ValueError("Пир не найден. Сначала дождитесь discovery или добавьте пира вручную.")
        message_id = sha256_text(f"{self.mac}|{peer_mac}|{text}|{time.time_ns()}")[:20]
        payload = {
            "type": "chat",
            "app": APP_NAME,
            "version": APP_VERSION,
            "message_id": message_id,
            "from_mac": self.mac,
            "from_name": self.name,
            "reply_port": self.tcp_port,
            "session_id": self.session_id,
            "timestamp": now_iso(),
            "text": text,
        }
        ack = self._send_tcp_payload(peer.ip, peer.port, payload)
        if ack.get("status") != "ok":
            raise ValueError("Пир отклонил сообщение.")
        record = self._append_ledger_event(
            {
                "direction": "out",
                "peer_mac": peer.mac,
                "peer_name": peer.name,
                "peer_ip": peer.ip,
                "timestamp": payload["timestamp"],
                "text": text,
                "message_id": message_id,
                "remote_block_hash": ack.get("block_hash", ""),
            }
        )
        self._emit(
            "message_sent",
            peer_mac=peer.mac,
            peer_name=peer.name,
            text=text,
            timestamp=payload["timestamp"],
            block_hash=record["block_hash"],
        )
        return record

    def _send_tcp_payload(self, ip: str, port: int, payload: dict[str, Any]) -> dict[str, Any]:
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        with socket.create_connection((ip, port), timeout=5.0) as sock:
            sock.settimeout(5.0)
            sock.sendall(data)
            sock.shutdown(socket.SHUT_WR)
            chunks: list[bytes] = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
        if not chunks:
            return {}
        raw_line = b"".join(chunks).splitlines()[0].decode("utf-8")
        try:
            return json.loads(raw_line)
        except json.JSONDecodeError:
            return {}
