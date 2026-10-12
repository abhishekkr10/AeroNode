"""Isolated project-local PostgreSQL for validation; existing services stay separate."""
from pathlib import Path
import os
import socket
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1",0))
        return socket_.getsockname()[1]


class LocalPostgres:
    def __init__(self):
        dependencies = ROOT/".deps"
        dependencies.mkdir(exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="telemetry-postgres-",dir=dependencies)).resolve()
        self.data=self.directory/"data"
        if not self.directory.is_relative_to(dependencies.resolve()):
            raise RuntimeError("PostgreSQL validation directory must be project-local")
        self.bin = Path(os.environ.get("AERONODE_PG_BIN",r"C:\Program Files\PostgreSQL\18\bin"))
        self.port = free_port()
        self.database_url = f"postgresql://aeronode@127.0.0.1:{self.port}/postgres"
        self.started = False

    def command(self, name, *arguments):
        executable = self.bin/(name+".exe" if os.name=="nt" else name)
        if not executable.is_file(): raise FileNotFoundError(executable)
        flags = subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
        # Windows PostgreSQL descendants can inherit pipe handles. A real log
        # file avoids waiting indefinitely for EOF after pg_ctl has exited.
        logfile=self.directory/f"{name}-command.log"
        with logfile.open("w",encoding="utf-8") as log:
            result = subprocess.run([str(executable),*map(str,arguments)],stdout=log,stderr=log,
                                    text=True,creationflags=flags,timeout=60)
        if result.returncode:
            raise RuntimeError(f"{name} failed: {logfile.read_text(encoding='utf-8',errors='replace')}")
        return result

    def __enter__(self):
        self.command("initdb","-D",self.data,"-U","aeronode","-A","trust","--encoding=UTF8","--locale=C")
        try:
            self.command("pg_ctl","-D",self.data,"-l",self.directory/"server.log",
                         "-o",f"-h 127.0.0.1 -p {self.port}","-w","start")
        except Exception:
            if (self.data/"postmaster.pid").is_file():
                self.command("pg_ctl","-D",self.data,"-m","fast","-w","stop")
            raise
        self.started = True
        return self

    def __exit__(self,*args):
        if self.started:
            self.command("pg_ctl","-D",self.data,"-m","fast","-w","stop")
            self.started = False
        # Retain ignored database/log evidence; never delete existing user data.
