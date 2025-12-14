import sqlite3
from pathlib import Path
from tqdm.auto import tqdm
import ast
import pandas as pd
import yaml
import re
from ConfigSpace import Configuration
import os

cache_folder = "./cache/trials"
master_file = Path("./results.db")

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

def table_exists(cur, name: str) -> bool:
    return cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone() is not None

def clone_table(cur, alias: str, table: str):
    sql = cur.execute(
        f'SELECT sql FROM "{alias}".sqlite_master '
        f'WHERE type="table" AND name=?',
        (table,),
    ).fetchone()[0]

    cur.execute(sql)


def merge_sqlite_dbs(master_path: str, others: list[str]):
    master = sqlite3.connect(master_path, uri=True)
    cur = master.cursor()

    for db in tqdm(others):
        # TODO Check if search id already finished else skip for now

        p = Path(db).resolve()
        print(p.stem)
        # #! Problem: Works only with newest version
        if str_exists( 
            master,
            table="results",
            column="experiment_id",
            value=p.stem,
        ): # Only add finished runs
            alias = safe_alias(p)

            # Proper URI: absolute path + uri=True on main connection
            uri = f"file:{p}?mode=ro"

            cur.execute(f'ATTACH DATABASE "file:{p}?mode=rw" AS "{alias}"')

            cur.execute("BEGIN")

            tables = cur.execute(
                f'SELECT name FROM "{alias}".sqlite_master WHERE type="table"'
            ).fetchall()

            for (table,) in tables:
                if table.startswith("sqlite_"):
                    continue

                if not table_exists(cur, table):
                    clone_table(cur, alias, table)
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

if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    cache_files = list(Path(cache_folder).glob("*.db"))
    print(cache_files)
    merge_sqlite_dbs(str(master_file.resolve()), others=cache_files)