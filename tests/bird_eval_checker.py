import json
import sqlite3
import os

EVAL_PATH = '../evals/eval_set.jsonl'  # adjust if needed
DB_ROOT = 'databases'        # base directory containing BIRD .db files


def load_eval(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def find_db_path(db_root, db_id):
    # common patterns: db_id.db or db_id/database.sqlite
    candidates = [
        os.path.join(db_root, f"{db_id}.db"),
        os.path.join(db_root, db_id, f"{db_id}.db"),
        os.path.join(db_root, db_id, 'database.sqlite'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def run_eval(eval_items, db_root):
    results = []
    for idx, item in enumerate(eval_items):
        q = item.get('question')
        db_id = item.get('db_id')
        sql = item.get('gold_sql')
        rec = {
            'index': idx,
            'question': q,
            'db_id': db_id,
            'sql': sql,
            'status': 'not_run',
            'error': None,
            'rowcount': None,
            'sample_row': None,
        }
        db_path = find_db_path(db_root, db_id)
        if not db_path:
            rec['status'] = 'db_not_found'
            results.append(rec)
            continue
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            rec['status'] = 'ok'
            rec['rowcount'] = len(rows)
            rec['sample_row'] = rows[0] if rows else None
        except Exception as e:
            rec['status'] = 'error'
            rec['error'] = str(e)
        finally:
            try:
                conn.close()
            except Exception:
                pass
        results.append(rec)
    return results


def main():
    eval_items = load_eval(EVAL_PATH)
    results = run_eval(eval_items, DB_ROOT)
    os.makedirs('output', exist_ok=True)
    out_path = os.path.join('output', 'eval_check_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results to {out_path}")


if __name__ == '__main__':
    main()
