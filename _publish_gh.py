"""用 GitHub token 建公开仓库 + 推送 + 校验。"""
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

REPO = "zhixun-harness"
DESC = "智训引擎：工业模型自进化 Harness（漂移监测-归因-优化-守门-回滚）+ 三包模型优化与实验日志"
UA = "dsh-agent"
PROXY = os.environ.get("GIT_PROXY", "")


def api(method, path, token, payload=None):
    url = f"https://api.github.com{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/vnd.github+json",
               "User-Agent": UA,
               "Content-Type": "application/json"}
    last = None
    for attempt in range(4):
        for proxy in ([None, PROXY] if PROXY else [None]):
            handlers = ([urllib.request.ProxyHandler({"https": proxy, "http": proxy})]
                        if proxy else [urllib.request.ProxyHandler({})])
            opener = urllib.request.build_opener(*handlers)
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with opener.open(req, timeout=30) as resp:
                    return resp.status, json.loads(resp.read() or b"{}")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                try:
                    return exc.code, json.loads(body)
                except json.JSONDecodeError:
                    return exc.code, {"raw": body[:300]}
            except Exception as exc:                          # noqa: BLE001
                last = {"error": f"{type(exc).__name__}: {exc}"}
        if attempt < 3:
            print(f"  ...第 {attempt + 1} 次失败，重试")
            time.sleep(2)
    return None, last or {"error": "unknown"}


def run(*args):
    r = subprocess.run(args, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def main():
    token = os.environ.get("GH_TOKEN", "").strip()
    if not token:
        print("需要环境变量 GH_TOKEN")
        return 1
    print(f"token: {token[:4]}... (len={len(token)})")

    code, me = api("GET", "/user", token)
    if code != 200:
        print(f"token 校验失败: {code} {me}")
        return 1
    login = me["login"]
    print(f"身份: {login} ({me.get('name')})")

    code, repo = api("POST", "/user/repos", token,
                     {"name": REPO, "description": DESC, "private": False,
                      "has_issues": True, "has_wiki": False, "auto_init": False})
    if code == 201:
        print(f"已创建公开仓库: {repo.get('html_url')}")
    elif code == 422:
        code, repo = api("GET", f"/repos/{login}/{REPO}", token)
        print(f"仓库已存在, 复用: https://github.com/{login}/{REPO} (private={repo.get('private')})")
    else:
        print(f"创建失败: {code} {repo}")
        return 1

    url = f"https://github.com/{login}/{REPO}.git"
    code, _, err = run("git", "remote", "get-url", "origin")
    if code == 0:
        run("git", "remote", "set-url", "origin", url)
    else:
        code, _, err = run("git", "remote", "add", "origin", url)
        if code != 0:
            print(f"配置 remote 失败: {err}")
            return 1
    print(f"remote origin = {url}")

    tracked = run("git", "ls-files")[1].splitlines()
    if any(".env.local" in p for p in tracked):
        print("危险: .env.local 在跟踪列表里, 已中止")
        return 1
    print(f"确认 .env.local 未跟踪; 待推 {len(tracked)} 个文件")

    print("\n=== 推送 main ===")
    code, out, err = run("git", "push", "-u", "origin", "main")
    print(((out + "\n" + err).strip())[-1200:])
    if code != 0:
        print("推送失败")
        return 1

    code, repo2 = api("GET", f"/repos/{login}/{REPO}", token)
    print(f"\n仓库: private={repo2.get('private')} size={repo2.get('size')}KB "
          f"branch={repo2.get('default_branch')}")

    code, tree = api("GET",
                     f"/repos/{login}/{REPO}/git/trees/{repo2.get('default_branch')}?recursive=1",
                     token)
    paths = [t["path"] for t in (tree.get("tree") or []) if t.get("type") == "blob"]
    print(f"远端文件数: {len(paths)}")
    bad = [p for p in paths
           if ".env.local" in p or p.startswith((".tools/", "dist/", "runs/"))]
    print(f"敏感/大文件: {'发现 ' + str(bad) if bad else '无 .env.local / .tools / dist / runs'}")
    for key in ("README.md", "harness/runner.py", "harness/policy.py", "Dockerfile",
                "实验日志.md", "models/B/checkpoint_rank.npz"):
        print(f"   {'OK' if key in paths else 'MISSING'}  {key}")
    print(f"\n>>> 表单里的链接: https://github.com/{login}/{REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
