/**
 * WHICH gitnexus this repo runs, resolved at RUNTIME.
 *
 * The generated npm scripts get the answer baked in at install time, but the shipped helpers here
 * spawn gitnexus themselves and used to hardcode `npx -y gitnexus@latest`. That is a different
 * program from the one everything else uses: npx never consults PATH, it downloads and caches its
 * own copy of the published package. So on a machine running a locally linked build,
 * `bearing:agent-status` reported the version of the STOCK npm build while every real operation
 * used the linked one — both printing the same version string, so the health check stayed green
 * even if the two had diverged completely. The doctor was examining a different patient.
 *
 * Order: the recorded choice (what the operator actually configured) → whatever is installed →
 * npx as the last resort so a machine with no global install still works.
 */
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

/** Manifest locations, newest first — order matters, the first readable one wins. */
const MANIFESTS = ['.bearing/manifest.json', '.gitnexus/agent-kit-manifest.json'];

/** @param {string} root @returns {string|null} the recorded command, if any */
function recordedCmd(root) {
  for (const rel of MANIFESTS) {
    try {
      const m = JSON.parse(fs.readFileSync(path.join(root, rel), 'utf8'));
      if (typeof m.gitnexusCmd === 'string' && m.gitnexusCmd.trim()) return m.gitnexusCmd.trim();
    } catch {
      /* missing or malformed → try the next, then fall through to detection */
    }
  }
  return null;
}

let _resolved;
/** @returns {string} `gitnexus` when it is installed, else `npx -y gitnexus@latest` */
function detectCmd() {
  if (_resolved) return _resolved;
  const probe = process.platform === 'win32' ? 'where' : 'which';
  const r = spawnSync(probe, ['gitnexus'], { encoding: 'utf8' });
  const hit = (r.stdout || '').trim().split(/\r?\n/)[0];
  _resolved = r.status === 0 && hit ? 'gitnexus' : 'npx -y gitnexus@latest';
  return _resolved;
}

/**
 * The full command string, e.g. `gitnexus` or `npx -y gitnexus@latest`.
 * @param {string} [root] repo root (defaults to cwd)
 */
export function gitnexusCmd(root = process.cwd()) {
  return recordedCmd(root) ?? detectCmd();
}

/**
 * The linked-worktree name git assigned to `root`, or null for a main checkout. A linked worktree's
 * `.git` is a file pointing at `<common>/.git/worktrees/<name>`, and git keeps `<name>` unique.
 * @param {string} root
 */
function linkedWorktreeName(root) {
  try {
    const m = fs.readFileSync(path.join(root, '.git'), 'utf8').match(/^gitdir:\s*(.+?)\s*$/m);
    return m?.[1].replace(/\\/g, '/').match(/\/worktrees\/([^/]+)\/?$/)?.[1] ?? null;
  } catch {
    return null; // `.git` is a directory (main checkout) or absent
  }
}

const _repoNames = new Map();
/**
 * The repository alias used by both GitNexus and Bearing. `GITNEXUS_REPO` wins, then `.gitnexusrc`
 * `name`, then the directory name. A linked worktree gets `-<worktree>` appended: `.gitnexusrc` is
 * committed, so without the suffix every worktree would claim the main checkout's registry alias.
 * @param {string} [root]
 */
export function gitnexusRepoName(root = process.cwd()) {
  if (process.env.GITNEXUS_REPO) return process.env.GITNEXUS_REPO;
  if (_repoNames.has(root)) return _repoNames.get(root);
  let name = path.basename(root);
  try {
    const config = JSON.parse(fs.readFileSync(path.join(root, '.gitnexusrc'), 'utf8'));
    const configured = config.analyze?.name ?? config.name;
    if (typeof configured === 'string' && configured.trim()) name = configured.trim();
  } catch {
    // No project-local alias; GitNexus defaults to the directory name.
  }
  const worktree = linkedWorktreeName(root);
  if (worktree) name = `${name}-${worktree}`;
  _repoNames.set(root, name);
  return name;
}

/**
 * Split into the shape spawnSync wants, with any extra args appended. `analyze` always gets an
 * explicit `--name` so the alias GitNexus registers is the one Bearing queries — `.gitnexusrc`
 * alone knows nothing about GITNEXUS_REPO or linked worktrees.
 * @param {string[]} args e.g. ['--version']
 * @param {string} [root]
 * @returns {{ command: string, args: string[] }}
 */
export function gitnexusSpawn(args = [], root = process.cwd()) {
  const parts = gitnexusCmd(root).split(/\s+/).filter(Boolean);
  const extra = args[0] === 'analyze' && !args.includes('--name') ? ['--name', gitnexusRepoName(root)] : [];
  return { command: parts[0], args: [...parts.slice(1), ...args, ...extra] };
}

/**
 * The MCP entry for this repo, matching what lib/mcp-config.mjs writes at install time.
 *
 * Shell callers need this: bearing-setup.sh writes .cursor/mcp.json AFTER the installer has
 * already written it, so hardcoding an entry there silently reverted both the transport and the
 * binary choice on every install.
 * @param {string} [root]
 */
export function mcpEntryFor(root = process.cwd()) {
  let transport = null;
  for (const rel of MANIFESTS) {
    try {
      transport = JSON.parse(fs.readFileSync(path.join(root, rel), 'utf8')).mcpTransport;
      if (transport) break;
    } catch {
      /* try the next */
    }
  }
  if (transport?.mode === 'http' && transport.url) {
    return { type: 'http', url: transport.url };
  }
  const parts = gitnexusCmd(root).split(/\s+/).filter(Boolean);
  return { command: parts[0], args: [...parts.slice(1), 'mcp'] };
}

// `node .bearing/lib/gitnexus-cmd.mjs [--mcp-entry | --repo-name]` — so shell scripts can ask the same question
// the JS callers do instead of hardcoding an answer that goes stale.
// `node .bearing/lib/gitnexus-cmd.mjs --exec <gitnexus args…>` runs gitnexus exactly as
// gitnexusSpawn would and exits with its status, so shell wrappers need no word-splitting.
if (process.argv[1] && fs.realpathSync(process.argv[1]) === fs.realpathSync(fileURLToPath(import.meta.url))) {
  const root = process.cwd();
  if (process.argv[2] === '--exec') {
    const { command, args } = gitnexusSpawn(process.argv.slice(3), root);
    const r = spawnSync(command, args, { stdio: 'inherit', shell: process.platform === 'win32' });
    if (r.error) {
      console.error(`gitnexus: could not run \`${command}\`: ${r.error.message}`);
      process.exit(127);
    }
    process.exit(r.status ?? 1);
  }
  process.stdout.write(
    (process.argv.includes('--mcp-entry')
      ? JSON.stringify(mcpEntryFor(root))
      : process.argv.includes('--repo-name')
        ? gitnexusRepoName(root)
        : gitnexusCmd(root)) + '\n',
  );
}
