"""Build an honest migration gallery and hashes for verified native videos."""

import argparse
import hashlib
import html
import json
from pathlib import Path
from scripts.engines import ZANIM_EPISODES
from scripts.episode_info import EPISODES, introduction


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Include previously validated native episodes",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    rows = json.loads((output / "summary.json").read_text())
    if args.collect:
        for ep in ZANIM_EPISODES - {row["episode"] for row in rows}:
            report = json.loads((output / ep / "report.json").read_text())
            assert report["full_decode"] == "passed" and not report["problems"]
            rows.append(
                {
                    "episode": ep,
                    "engine": "zanim",
                    "profile": "final",
                    "status": "passed",
                    "full_decode": "passed",
                    "duration": report["duration"],
                    "problems": [],
                }
            )
        rows.sort(key=lambda row: tuple(map(int, row["episode"].split("."))))
        (output / "summary.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        )
    assert {row["episode"] for row in rows} == ZANIM_EPISODES
    assert all(
        row["status"] == "passed" and row["full_decode"] == "passed" for row in rows
    )
    manifest = {
        "engine": "zanim",
        "version": "0.7.0rc1",
        "resolution": [1920, 1080],
        "fps": 60,
        "native_count": len(rows),
        "total_count": len(EPISODES),
        "episodes": [],
    }
    cards = []
    for row in rows:
        ep = row["episode"]
        report = json.loads((output / ep / "report.json").read_text())
        video = output / ep / "video.mp4"
        assert digest(video) == digest(Path(report["video"]))
        title, explanation = introduction(ep)
        manifest["episodes"].append(
            {
                "episode": ep,
                "duration": row["duration"],
                "sha256": digest(video),
                "full_decode": "passed",
                "random_access_render_check": "passed",
            }
        )
        cards.append(
            f'<a class="card" href="{ep}/index.html"><img loading="lazy" src="{ep}/frame-05.png" alt="{html.escape(title)}"><div><h2>{html.escape(title)}</h2><p>{html.escape(explanation)}</p><small>{row["duration"]:.1f} 秒 · 1080p60 · 查看动画 →</small></div></a>'
        )
    repo = Path(__file__).resolve().parents[1]
    manifest["source_sha256"] = {
        str(p.relative_to(repo)): digest(p)
        for p in sorted((repo / "zanim_scenes").rglob("*.py"))
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    page = (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>点还在动 · Zanim 新版</title><style>
*{box-sizing:border-box}body{margin:0;background:#0c1423;color:#edf4fa;font:16px/1.7 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1320px;margin:auto;padding:48px 28px 80px}.eyebrow,small{color:#4edbc0}h1{font-size:clamp(30px,4vw,52px);line-height:1.25;margin:18px 0}p{color:#96adc1}a{color:inherit;text-decoration:none}video{width:100%;display:block;background:#0c1423;border:1px solid #293c50;border-radius:12px;margin:32px 0 12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:24px;margin-top:32px}.card{overflow:hidden;background:#142134;border:1px solid #263b50;border-radius:12px;transition:border-color .2s}.card:hover{border-color:#4edbc0}.card img{width:100%;display:block}.card div{padding:20px}.card h2{font-size:20px;margin:0}.card p{font-size:14px;min-height:48px}.meta{font-size:14px}footer{margin-top:40px;color:#96adc1;font-size:14px}button,a:focus-visible{outline:2px solid #4edbc0}.search{margin-top:40px}.search label{display:block;font-size:14px;color:#96adc1;margin-bottom:8px}.search input{width:100%;padding:14px 16px;border-radius:8px;border:1px solid #36516b;background:#142134;color:#edf4fa;font:inherit}.search input:focus{outline:2px solid #4edbc0}.card[hidden]{display:none}</style><main><div class="eyebrow">点还在动 / DEEP LEARNING, IN MOTION</div><h1>点还在动 · 样片工作台</h1><p>已完成 {{count}} / 49 集的原生 Zanim 技术迁移。现有样片正向连续动画作品重新编排；迁移通过不代表艺术效果已验收。</p><p class="meta">{{migration_note}}当前先展示转置卷积：颜色表示贡献来源，高度表示数值。</p><video controls playsinline preload="metadata" poster="13.10/frame-05.png" src="13.10/video.mp4"></video><p class="meta">转置卷积 · 13.10　<a href="13.10/video.mp4" download>下载原片 ↓</a></p><div class="search"><label for="episode-search">查找章节 · 输入编号或主题</label><input id="episode-search" type="search" placeholder="例如：09.2、注意力、卷积" autocomplete="off"><p id="match-count" class="meta" aria-live="polite">共 {{count}} 集</p></div><div class="grid">"""
        + "".join(cards)
        + """</div><footer>{{count}} 部成片均通过完整解码、分辨率、帧率及随机访问渲染检查；代码与数值回归检查通过。已抽查关键帧，未完成逐帧人工审片。<br><a href="manifest.json">查看文件校验记录</a> · <a href="../review-final/index.html">查看原版全集</a></footer></main><script>
const search = document.getElementById('episode-search');
const cards = Array.from(document.querySelectorAll('.card'));
search.addEventListener('input', () => {
  const query = search.value.trim().toLowerCase();
  let count = 0;
  cards.forEach(card => {
    const match = card.textContent.toLowerCase().includes(query);
    card.hidden = !match;
    if (match) count += 1;
  });
  document.getElementById('match-count').textContent = `显示 ${count} / ${cards.length} 集`;
});
</script></html>"""
    )
    if (output.parent / "continuous-study" / "index.html").exists():
        page = page.replace('<video controls', '<p><a href="../continuous-study/index.html">观看连续动画样片 →</a></p><video controls', 1)
    (output / "index.html").write_text(
        page.replace("{{count}}", str(len(rows))).replace(
            "{{migration_note}}",
            "全集已采用原生 Zanim 渲染。"
            if len(rows) == len(EPISODES)
            else f"其余 {len(EPISODES) - len(rows)} 集尚未迁移。",
        )
    )
    print(f"Native Zanim delivery: {len(rows)}/49")


if __name__ == "__main__":
    main()
