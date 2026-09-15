import { existsSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const DB = fileURLToPath(new URL("./data/mm.sqlite", import.meta.url));

/**
 * Read-only JSON over the walk store:
 *   /api/runs?dimension=d   the latest run of every walk in that tab, in walk order
 *   /api/run/:id            one run with its nodes and edges
 */
function mmApi(): Plugin {
  return {
    name: "mm-api",
    configureServer(server) {
      server.middlewares.use("/api", (req, res) => {
        const send = (body: unknown, status = 200) => {
          res.statusCode = status;
          res.setHeader("content-type", "application/json");
          res.end(JSON.stringify(body));
        };
        if (!existsSync(DB)) return send([]);
        let db: DatabaseSync | null = null;
        try {
          db = new DatabaseSync(DB);
          db.exec("PRAGMA busy_timeout = 5000"); // a walk may be writing: wait for it rather than fail
          const url = new URL(req.url ?? "/", "http://local");
          if (url.pathname === "/runs") {
            const dim = Number(url.searchParams.get("dimension") ?? "2");
            return send(
              db
                .prepare("SELECT id, walk, dimension, container, rail, seal, created FROM runs WHERE id IN (SELECT MAX(id) FROM runs WHERE dimension = ? GROUP BY walk) ORDER BY id")
                .all(dim),
            );
          }
          if (url.pathname === "/ledger") return send(db.prepare("SELECT * FROM ledger ORDER BY id").all());
          const one = url.pathname.match(/^\/run\/(\d+)$/);
          if (one) {
            const run = db.prepare("SELECT * FROM runs WHERE id = ?").get(Number(one[1])) as { id: number } | undefined;
            if (!run) return send({ error: "no such run" }, 404);
            const nodes = db.prepare("SELECT * FROM nodes WHERE run_id = ? ORDER BY id").all(run.id);
            const edges = db.prepare("SELECT * FROM edges WHERE run_id = ? ORDER BY id").all(run.id);
            return send({ run, nodes, edges });
          }
          send({ error: "not found" }, 404);
        } catch (e) {
          // the store is being rebuilt by a walk: answer, do not crash the page
          send({ error: String(e) }, 503);
        } finally {
          db?.close();
        }
      });
    },
  };
}

export default defineConfig({ root: "web", plugins: [react(), mmApi()] });
