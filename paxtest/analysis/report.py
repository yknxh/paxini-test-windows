"""세션 리포트 HTML (브라우저로 바로 열 수 있는 단일 파일 + plots/ 이미지)."""
from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd

CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;
--good:#006300;--good-bg:#e6f4e6;--bad:#b42626;--bad-bg:#fbe9e9;--na-bg:#f0efec}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--line:#2c2c2a;--good:#0ca30c;--good-bg:#132613;--bad:#e66767;--bad-bg:#2e1616;--na-bg:#262624}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 10px}
.sub{color:var(--ink2);margin:0 0 16px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--ink2);font-weight:600;font-size:12.5px}td.num{text-align:right}
.badge{display:inline-block;padding:2px 10px;border-radius:99px;font-weight:700;font-size:13px}
.pass{color:var(--good);background:var(--good-bg)}.fail{color:var(--bad);background:var(--bad-bg)}
.na{color:var(--ink2);background:var(--na-bg)}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:4px 24px}
.kv div{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding:4px 0}
.kv span:first-child{color:var(--ink2)}
figure{margin:12px 0;background:#fcfcfb;border:1px solid var(--line);border-radius:10px;padding:8px}
figure img{width:100%;height:auto;display:block}figcaption{color:#52514e;font-size:12.5px;padding:4px 6px 0}
details summary{cursor:pointer;color:var(--ink2)}ul.notes{margin:6px 0;padding-left:20px}
.scroll{overflow-x:auto}a{color:#2a78d6}
"""


def fmt(v, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "–" if not (isinstance(v, float) and math.isinf(v)) else "∞"
    if isinstance(v, bool):
        return "예" if v else "아니오"
    if isinstance(v, (int,)) or (isinstance(v, float) and v.is_integer() and abs(v) >= 100):
        return f"{v:,.0f}"
    if isinstance(v, float):
        a = abs(v)
        return f"{v:.{digits}f}" if a < 1 else f"{v:.3f}" if a < 100 else f"{v:.1f}"
    return html.escape(str(v))


def badge(passed) -> str:
    if passed is True:
        return '<span class="badge pass">✓ PASS</span>'
    if passed is False:
        return '<span class="badge fail">✕ FAIL</span>'
    return '<span class="badge na">– N/A</span>'


def df_table(df: pd.DataFrame, index: bool = False) -> str:
    cols = ([df.index.name or ""] if index else []) + list(map(str, df.columns))
    out = ["<div class='scroll'><table><thead><tr>", *[f"<th>{html.escape(c)}</th>" for c in cols], "</tr></thead><tbody>"]
    for idx, r in df.iterrows():
        out.append("<tr>")
        if index:
            out.append(f"<td>{html.escape(str(idx))}</td>")
        for v in r.tolist():
            num = isinstance(v, (int, float))
            out.append(f"<td class='{'num' if num else ''}'>{fmt(v) if num else html.escape(str(v))}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def write_report(path: Path, meta: Dict, res, overall, sync: Dict, step_stats: pd.DataFrame,
                 files: List[str]) -> None:
    o = {"PASS": True, "FAIL": False}.get(overall)
    sensors = ", ".join(f"{s['id']} ({s['label']}, {s['rated_N']:g} N, CH{s['channel']})" for s in meta.get("sensors", []))
    opts = meta.get("options", {})
    kv = [("세션", meta.get("session_id")), ("시작", meta.get("start_local")), ("종료", meta.get("end_local", "–")),
          ("소요", f"{meta.get('duration_s', 0) / 60:.1f} 분"), ("상태", meta.get("status")), ("모드", meta.get("mode")),
          ("작업자", opts.get("operator") or "–"), ("시간 배율", opts.get("quick", 1)),
          ("PXSR 파일", len(meta.get("pxsr_files", []))), ("영상 프레임", meta.get("video_frames", 0)),
          ("비교 기준", "합력 |F|" if str(meta.get("test_code", "")).startswith("R") else "Fz")]
    sync_txt = ("미적용 (데이터 부족 또는 상관 낮음)" if not sync.get("applied") else
                f"{sync['offset_s'] * 1000:+.0f} ms 보정 · 상관 {sync.get('corr')} · 방법 {sync.get('method')}")
    parts = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{html.escape(meta.get('test_code', ''))} {html.escape(meta.get('test_name', ''))} 리포트</title>"
             f"<style>{CSS}</style></head><body><main>",
             f"<h1>{html.escape(meta.get('test_code', ''))} · {html.escape(meta.get('test_name', ''))} {badge(o)}</h1>",
             f"<p class='sub'>{html.escape(sensors)}</p>",
             "<div class='card kv'>" + "".join(f"<div><span>{k}</span><span>{fmt(v) if not isinstance(v, str) else html.escape(v)}</span></div>" for k, v in kv)
             + f"<div style='grid-column:1/-1'><span>시간 동기</span><span>{html.escape(sync_txt)}</span></div></div>"]
    if opts.get("notes"):
        parts.append(f"<div class='card'><b>메모</b><br>{html.escape(opts['notes'])}</div>")

    parts.append("<h2>판정</h2>")
    if res.checks:
        rows = "".join(
            f"<tr><td>{html.escape(c['sensor'])}</td><td>{html.escape(c['label'])}</td>"
            f"<td class='num'>{fmt(c['value'])} {html.escape(c['unit'])}</td>"
            f"<td class='num'>{'≤ ' if c['mode'] == 'le' else '≥ ' if c['mode'] == 'ge' else ''}"
            f"{fmt(c['limit']) if c['mode'] != 'bool' else '일치'} {html.escape(c['unit']) if c['mode'] != 'bool' else ''}</td>"
            f"<td>{badge(c['passed'])}</td></tr>" for c in res.checks)
        parts.append(f"<div class='card scroll'><table><thead><tr><th>센서</th><th>항목</th><th>값</th><th>기준</th>"
                     f"<th>판정</th></tr></thead><tbody>{rows}</tbody></table></div>")
    else:
        parts.append("<div class='card'>이 테스트에는 판정 기준이 없습니다 (측정값만 기록).</div>")
    if res.notes:
        parts.append("<div class='card'><b>참고</b><ul class='notes'>" +
                     "".join(f"<li>{html.escape(n)}</li>" for n in res.notes) + "</ul></div>")

    parts.append("<h2>지표</h2>")
    if res.metrics:
        m = pd.DataFrame(res.metrics)
        rows = "".join(f"<tr><td>{html.escape(r.sensor)}</td><td>{html.escape(r.label)}</td>"
                       f"<td class='num'>{fmt(r.value)}</td><td>{html.escape(r.unit)}</td></tr>" for r in m.itertuples())
        parts.append(f"<div class='card scroll'><table><thead><tr><th>센서</th><th>지표</th><th>값</th><th>단위</th>"
                     f"</tr></thead><tbody>{rows}</tbody></table></div>")
    for name, t in res.tables.items():
        parts.append(f"<h2>{html.escape(name)}</h2><div class='card'>{df_table(t, index=not isinstance(t.index, pd.RangeIndex))}</div>")

    parts.append("<h2>그래프</h2>")
    for p in res.plots:
        parts.append(f"<figure><img src='{html.escape(p['file'])}' alt='{html.escape(p['caption'])}'>"
                     f"<figcaption>{html.escape(p['caption'])}</figcaption></figure>")

    if not step_stats.empty:
        cols = [c for c in ["step_idx", "sensor", "title", "ref_N", "Fmag_mean", "Fmag_std", "Fz_mean",
                            "Fz_std", "error_N", "gauge_std", "n"] if c in step_stats]
        parts.append("<h2>단계별 통계</h2><details><summary>펼치기 (step_stats.csv 와 동일)</summary>"
                     f"<div class='card'>{df_table(step_stats[cols])}</div></details>")
    parts.append("<h2>파일</h2><div class='card'><ul class='notes'>" +
                 "".join(f"<li><a href='{html.escape(f)}'>{html.escape(f)}</a></li>" for f in files) + "</ul></div>")
    parts.append("</main></body></html>")
    path.write_text("".join(parts), encoding="utf-8")
