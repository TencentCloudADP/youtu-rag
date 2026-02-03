import socket


class PortManager:
    def __init__(self, port_range: tuple = (9000, 9999)):
        self.port_start, self.port_end = port_range
        self.used_ports: set[int] = set()
        self.reserved_ports: set[int] = set()

    def is_port_available(self, port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(("localhost", port))
                return result != 0
        except Exception:  # pylint: disable=broad-except
            return False

    def allocate_port(self) -> int | None:
        for port in range(self.port_start, self.port_end + 1):
            if port not in self.used_ports and port not in self.reserved_ports:
                if self.is_port_available(port):
                    self.used_ports.add(port)
                    return port
        return None

    def release_port(self, port: int):
        self.used_ports.discard(port)
        self.reserved_ports.discard(port)

    def reserve_port(self, port: int) -> bool:
        if port not in self.used_ports and self.is_port_available(port):
            self.reserved_ports.add(port)
            return True
        return False

    def get_host_ip(self) -> str:
        return socket.gethostbyname(socket.gethostname())
