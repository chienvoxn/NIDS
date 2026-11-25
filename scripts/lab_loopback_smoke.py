"""Generate only benign, bounded TCP traffic on local loopback."""
import socket
import threading
import time


def run():
    payload = b'NIDS benign lab smoke test'
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        server.listen(1)
        server.settimeout(5)
        address = server.getsockname()
        errors = []
        def serve():
            try:
                for _ in range(6):
                    with server.accept()[0] as conn:
                        conn.settimeout(3)
                        data = conn.makefile('rb').read(len(payload))
                        if data != payload:
                            raise ValueError('Unexpected payload')
                        conn.sendall(data)
            except Exception as exc:
                errors.append(exc)
        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        for _ in range(6):
            with socket.create_connection(address, timeout=3) as client:
                client.sendall(payload)
                if client.makefile('rb').read(len(payload)) != payload:
                    raise ValueError('Echo mismatch')
            time.sleep(0.5)
        worker.join(5)
        if worker.is_alive() or errors:
            raise RuntimeError(f'Loopback server incomplete: {errors}')
    return 6

if __name__ == '__main__':
    print('Benign TCP connections completed:', run())
