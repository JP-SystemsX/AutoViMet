import sqlite3
from pathlib import Path
from tqdm.auto import tqdm
import re

cache_folder = "../cache/trials"
master_file = Path("../results.db")

def safe_alias(path: Path) -> str:
    alias = re.sub(r"\W+", "_", path.stem)
    if alias[0].isdigit():
        alias = "a_" + alias
    return alias

def str_exists(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    value: str
) -> bool:
    """
    Return True if any row in `table.column` equals `value` exactly.
    """
    query = f"""
        SELECT EXISTS(
            SELECT 1
            FROM {table}
            WHERE {column} = ?
        )
    """

    
    (exists,) = conn.execute(query, (value,)).fetchone()
    return bool(exists)

def merge_sqlite_dbs(master_path: str, others: list[str]):
    master = sqlite3.connect(master_path, uri=True)
    cur = master.cursor()

    for db in tqdm(others):
        # TODO Check if search id already finished else skip for now

        p = Path(db).resolve()
        if str_exists(
            master,
            table="results",
            column="search_id",
            value=p.name,
        ): # Only add finished runs
            alias = safe_alias(p)

            # Proper URI: absolute path + uri=True on main connection
            uri = f"file:{p}?mode=ro"

            cur.execute(f'ATTACH DATABASE ? AS "{alias}"', (uri,))

            cur.execute("BEGIN")

            tables = cur.execute(
                f'SELECT name FROM "{alias}".sqlite_master WHERE type="table"'
            ).fetchall()

            for (table,) in tables:
                cur.execute(
                    f'INSERT OR IGNORE INTO "{table}" '
                    f'SELECT * FROM "{alias}"."{table}"'
                )

            cur.execute("COMMIT")
            cur.execute(f'DETACH DATABASE "{alias}"')

            # Commit + Delete Cache File
            master.commit()
            p.unlink()

    master.commit()
    master.close()

