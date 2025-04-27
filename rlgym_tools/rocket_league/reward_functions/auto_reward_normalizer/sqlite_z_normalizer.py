import sqlite3
from typing import Sequence

from rlgym_tools.rocket_league.reward_functions.auto_reward_normalizer.simple_z_normalizer import SimpleZNormalizer


class SQLiteZNormalizer(SimpleZNormalizer):
    # Shares stats between processes via SQLite.
    def __init__(self, db_path: str, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_path = db_path
        self.name = name
        self.conn = sqlite3.connect(db_path, isolation_level=None)
        self.cursor = self.conn.cursor()
        with self.conn:
            self.cursor.execute(f"CREATE TABLE IF NOT EXISTS stats ("
                                f"name TEXT PRIMARY KEY, "
                                f"count REAL, "
                                f"sum REAL, "
                                f"sq_sum REAL"
                                f")")
            self.cursor.execute(f"INSERT OR IGNORE INTO stats (name, count, sum, sq_sum) VALUES (?, ?, ?, ?)",
                                (self.name, 0, 0, 0))

            self.cursor.execute(f"SELECT count, sum, sq_sum FROM stats WHERE name=?", (self.name,))
            row = self.cursor.fetchone()
            self._count, self._sum, self._sq_sum = row

    def __del__(self):
        self.conn.close()

    def update(self, values: Sequence):
        # Get latest stats from the database in case they were updated by another process
        with self.conn:
            row = self.conn.execute(f"SELECT count, sum, sq_sum FROM stats WHERE name=?", (self.name,)).fetchone()

            self._count, self._sum, self._sq_sum = row

            super().update(values)

            self.conn.execute("""
                UPDATE stats SET count = ?, sum = ?, sq_sum = ? WHERE name = ?
            """, (self._count, self._sum, self._sq_sum, self.name))

    # normalize() is inherited from SimpleZNormalizer
