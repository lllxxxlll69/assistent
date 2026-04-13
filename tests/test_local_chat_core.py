from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from assistant.local_chat.core import LanChatMessenger, format_mac


class LocalChatCoreTests(unittest.TestCase):
    def test_format_mac_normalizes_input(self) -> None:
        self.assertEqual(format_mac("aabbccddeeff"), "AA:BB:CC:DD:EE:FF")

    def test_manual_link_registers_peer_and_emits_update(self) -> None:
        messenger = LanChatMessenger(
            name="Alice",
            tcp_port=47010,
            discovery_port=47001,
            mac_override="AA:BB:CC:DD:EE:01",
        )

        peer = messenger.manual_link("AA:BB:CC:DD:EE:02", "192.168.0.7", 47011, "Bob")

        self.assertEqual(peer.name, "Bob")
        peers = messenger.get_peers()
        self.assertEqual(len(peers), 1)
        self.assertEqual(peers[0].mac, "AA:BB:CC:DD:EE:02")
        events = messenger.drain_events()
        self.assertTrue(any(item["type"] == "peers_updated" for item in events))

    def test_send_message_records_ledger_with_stubbed_transport(self) -> None:
        messenger = LanChatMessenger(
            name="Alice",
            tcp_port=47010,
            discovery_port=47001,
            mac_override="AA:BB:CC:DD:EE:01",
        )
        messenger.manual_link("AA:BB:CC:DD:EE:02", "127.0.0.1", 47011, "Bob")
        messenger._send_tcp_payload = lambda *_args, **_kwargs: {"status": "ok", "block_hash": "remote"}  # type: ignore[method-assign]

        record = messenger.send_message("AA:BB:CC:DD:EE:02", "hello")

        self.assertEqual(record["event"]["direction"], "out")
        self.assertEqual(record["event"]["peer_name"], "Bob")
        history = messenger.get_history(peer_mac="AA:BB:CC:DD:EE:02")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["event"]["text"], "hello")

    def test_send_file_records_ledger_with_stubbed_transport(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            messenger = LanChatMessenger(
                name="Alice",
                tcp_port=47010,
                discovery_port=47001,
                mac_override="AA:BB:CC:DD:EE:01",
                received_files_dir=tmp_dir,
            )
            messenger.manual_link("AA:BB:CC:DD:EE:02", "127.0.0.1", 47011, "Bob")
            messenger._send_tcp_payload = lambda *_args, **_kwargs: {"status": "ok", "block_hash": "remote"}  # type: ignore[method-assign]
            source = Path(tmp_dir) / "report.bin"
            source.write_bytes(b"\x00\x01payload")

            record = messenger.send_file("AA:BB:CC:DD:EE:02", source)

        self.assertEqual(record["event"]["direction"], "out")
        self.assertEqual(record["event"]["kind"], "file")
        self.assertEqual(record["event"]["file_name"], "report.bin")
        self.assertEqual(record["event"]["file_size"], 9)

    def test_receive_file_payload_saves_to_inbox_and_emits_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            messenger = LanChatMessenger(
                name="Alice",
                tcp_port=47010,
                discovery_port=47001,
                mac_override="AA:BB:CC:DD:EE:01",
                received_files_dir=tmp_dir,
            )
            raw_bytes = b"hello file"
            payload = {
                "type": "file",
                "from_mac": "AA:BB:CC:DD:EE:02",
                "from_name": "Bob",
                "reply_port": 47011,
                "session_id": "peer-session",
                "timestamp": "2026-04-13T12:00:00+03:00",
                "message_id": "file-1",
                "file_name": "hello.txt",
                "file_size": len(raw_bytes),
                "file_sha256": __import__("hashlib").sha256(raw_bytes).hexdigest(),
                "mime_type": "text/plain",
                "data_base64": base64.b64encode(raw_bytes).decode("ascii"),
            }

            response = messenger._process_payload(payload, ("127.0.0.1", 47011))
            history = messenger.get_history(peer_mac="AA:BB:CC:DD:EE:02")
            events = messenger.drain_events()
            self.assertEqual(response["status"], "ok")
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["event"]["kind"], "file")
            saved_path = Path(history[0]["event"]["saved_path"])
            self.assertTrue(saved_path.exists())
            self.assertEqual(saved_path.read_bytes(), raw_bytes)
            self.assertTrue(any(item["type"] == "file_received" for item in events))


if __name__ == "__main__":
    unittest.main()
