import pandas as pd
import json
from itertools import combinations

# ─── 데이터 로드 ──────────────────────────────────────────────
df = pd.read_csv("songs_tagged_v2.csv", encoding="utf-8-sig")
df = df.fillna("")

# ─── 아티스트별 색상 ──────────────────────────────────────────
ARTIST_COLORS = {
    "NewJeans":                  "#FF6B9D",
    "투모로우바이투게더":          "#6B9DFF",
    "연준":                      "#A8D0FF",
    "KATSEYE":                   "#FFD700",
    "IVE (아이브)":               "#FF9966",
    "범규":                      "#3560C0",
    "Hearts2Hearts (하츠투하츠)": "#48C9B0",
    "KiiKii (키키)":             "#F8C471",
    "NMIXX":                     "#DA70D6",
    "BLACKPINK":                 "#FF1493",
    "aespa":                     "#00E5FF",
    "LE SSERAFIM (르세라핌)":    "#FF8C42",
    "NCT DREAM":                 "#4FC3F7",
    "TWS (투어스)":              "#7ED4AD",
    "아일릿(ILLIT)":             "#FF85A2",
    "BABYMONSTER":               "#E74C3C",
    "RIIZE":                     "#5B9BD5",
}
DEFAULT_COLOR = "#AAAAAA"

# ─── 장르 패밀리 (같은 패밀리면 부분 유사도 부여) ────────────────
GENRE_FAMILY = {
    # Electronic Dance (하우스 계열)
    "dance pop":          "electronic_dance",
    "house":              "electronic_dance",
    "uk garage":          "electronic_dance",
    "jersey club":        "electronic_dance",
    "drum & bass":        "electronic_dance",
    "breakbeat":          "electronic_dance",
    "edm":                "electronic_dance",
    "rave":               "electronic_dance",
    "hardstyle":          "electronic_dance",
    "electronic dance":   "electronic_dance",
    "deep house":         "electronic_dance",
    # Synth / Electropop
    "synthpop":           "synth_pop",
    "new wave":           "synth_pop",
    "hyperpop":           "synth_pop",
    "city pop":           "synth_pop",
    "dream pop":          "synth_pop",
    # Disco / Funk
    "disco":              "disco_funk",
    "funk pop":           "disco_funk",
    "new jack swing":     "disco_funk",
    "miami bass":         "disco_funk",
    # Global Dance (라틴/아프로 계열)
    "dancehall":          "global_dance",
    "moombahton":         "global_dance",
    "reggaeton":          "global_dance",
    "brazilian funk":     "global_dance",
    "balie funk":         "global_dance",
    "afrobeats":          "global_dance",
    "latin pop":          "global_dance",
    # Hip Hop / R&B
    "hip hop":            "hip_hop_rb",
    "trap":               "hip_hop_rb",
    "old-school hip hop": "hip_hop_rb",
    "r&b":                "hip_hop_rb",
    "alternative r&b":    "hip_hop_rb",
    "soul":               "hip_hop_rb",
    # Rock
    "pop rock":           "rock",
    "hard rock":          "rock",
    "indie rock":         "rock",
    "alternative rock":   "rock",
    "emo rock":           "rock",
    "britpop":            "rock",
    "pop punk":           "rock",
    "punk rock":          "rock",
    "stadium rock":       "rock",
    "reggae rock":        "rock",
    # Soft Pop / Ballad
    "ballad":             "soft_pop",
    "indie pop":          "soft_pop",
    "alternative pop":    "soft_pop",
    "acoustic pop":       "soft_pop",
    "pop":                "soft_pop",
    # Global Dance (라틴/아프로 계열) — 추가
    "reggae pop":         "global_dance",
    # Hip Hop / R&B — 추가
    "boom bap":           "hip_hop_rb",
    # Electronic Dance — 추가
    "french house":       "electronic_dance",
    # Synth / Electropop — 추가
    "lo-fi":              "synth_pop",
    # Rock — 추가
    "punk pop":           "rock",
    # Hip Hop / R&B — 추가
    "rage":               "hip_hop_rb",
    "rock and roll":      "rock",
    # 패밀리 없음: jazz, country, orchestral, bossa nova
}
FAMILY_WEIGHT = 1.5  # main(2)과 sub(1) 사이 — 같은 패밀리면 이 가중치로 교집합 기여

# ─── 노드 생성 ────────────────────────────────────────────────
nodes = []
for _, row in df.iterrows():
    genres = [g for g in [row["main_genre"], row["sub_genre_1"], row["sub_genre_2"]] if g]
    moods  = [m for m in [row["mood_1"], row["mood_2"]] if m]
    energy     = int(row["energy"])     if str(row["energy"])     != "" else 3
    brightness = int(row["brightness"]) if str(row["brightness"]) != "" else 3
    album = str(row["album"]) if "album" in row and str(row["album"]) not in ("", "nan") else ""
    year  = str(int(float(row["year"]))) if "year" in row and str(row["year"]) not in ("", "nan") else ""
    album_line = ""
    if album and year:
        album_line = f"💿 {album} ({year})<br>"
    elif album:
        album_line = f"💿 {album}<br>"
    elif year:
        album_line = f"📅 {year}<br>"
    nodes.append({
        "id":         int(row["song_id"]),
        "label":      row["title"],
        "artist":     row["artist"],
        "genres":     genres,
        "moods":      moods,
        "energy":     energy,
        "brightness": brightness,
        "color":      ARTIST_COLORS.get(row["artist"], DEFAULT_COLOR),
        "size":       9,
        "tooltip": (
            f"<b>{row['title']}</b><br>"
            f"🎤 {row['artist']}<br>"
            f"{album_line}"
            f"🎸 {' / '.join(genres) if genres else '-'}"
        ),
    })

# ─── 유사도 계산 ──────────────────────────────────────────────
# 가중치: 장르 40%, energy 20%, brightness 20%, 무드 20%
W = {"genre": 0.40, "energy": 0.20, "brightness": 0.20, "mood": 0.20}
# 최소 threshold (HTML에서 슬라이더로 추가 필터링)
MIN_THRESHOLD = 0.55

def genre_sim(a, b):
    """family-augmented weighted Jaccard
    main_genre=2, sub_genre=1, 같은 패밀리 태그=FAMILY_WEIGHT(1.5)
    같은 패밀리 장르끼리는 패밀리 태그가 intersection에 기여해 부분 점수 발생
    """
    if not a or not b:
        return 0.0
    def build_weights(genres):
        w = {}
        for i, g in enumerate(genres):
            w[g] = 2 if i == 0 else 1
            fam = GENRE_FAMILY.get(g)
            if fam:
                w[fam] = max(w.get(fam, 0), FAMILY_WEIGHT)
        return w
    wa, wb = build_weights(a), build_weights(b)
    all_g = set(wa) | set(wb)
    inter = sum(min(wa.get(g, 0), wb.get(g, 0)) for g in all_g)
    union = sum(max(wa.get(g, 0), wb.get(g, 0)) for g in all_g)
    return inter / union if union else 0.0

def num_sim(a, b, scale=4):
    return 1.0 - abs(a - b) / scale

edges = []
for na, nb in combinations(nodes, 2):
    g_sim  = genre_sim(na["genres"], nb["genres"])
    e_sim  = num_sim(na["energy"],     nb["energy"])
    b_sim  = num_sim(na["brightness"], nb["brightness"])
    both_mood_empty = not na["moods"] and not nb["moods"]
    if both_mood_empty:
        total = (W["genre"]*g_sim + W["energy"]*e_sim + W["brightness"]*b_sim) / (1 - W["mood"])
    else:
        sa, sb = set(na["moods"]), set(nb["moods"])
        m_sim = len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0
        total = W["genre"]*g_sim + W["energy"]*e_sim + W["brightness"]*b_sim + W["mood"]*m_sim

    if total >= MIN_THRESHOLD:
        edges.append({
            "source":         na["id"],
            "target":         nb["id"],
            "similarity":     round(total, 3),
            "genre_sim":      round(g_sim,  3),
            "energy_sim":     round(e_sim,  3),
            "brightness_sim": round(b_sim,  3),
            "mood_sim":       round(m_sim,  3),
        })

edges.sort(key=lambda e: e["similarity"], reverse=True)
MAX_EDGES = 5000  # 노드 수 늘어나도 HTML이 너무 무거워지지 않도록
if len(edges) > MAX_EDGES:
    edges = edges[:MAX_EDGES]
print(f"nodes={len(nodes)}, edges={len(edges)} (threshold>={MIN_THRESHOLD}, cap={MAX_EDGES})")

# ─── 아티스트 체크박스 & 범례 HTML 사전 생성 ─────────────────────
active_artists = sorted([a for a in ARTIST_COLORS if a in df["artist"].values], key=lambda x: (x[0].isascii(), x))
artist_checkboxes = "\n".join(
    f'    <label class="check-item">'
    f'<input type="checkbox" class="artist-cb" value="{a}">'
    f'<span class="check-dot" style="background:{ARTIST_COLORS[a]}"></span>'
    f'{a}</label>'
    for a in active_artists
)
legend_items = "\n".join(
    f'  <div class="legend-item"><div class="legend-dot" style="background:{ARTIST_COLORS[a]}"></div>{a}</div>'
    for a in active_artists
)

# ─── HTML 생성 ────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>K-pop Song Network v2 — Similarity Graph</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0d0d1a; color:#eee; font-family:'Segoe UI',sans-serif; overflow:hidden; }}
  #canvas {{ width:100vw; height:100vh; }}

  .panel {{
    position:fixed; background:rgba(18,18,36,0.93); border:1px solid #2a2a4a;
    border-radius:14px; padding:14px 18px; font-size:13px; z-index:10;
    backdrop-filter:blur(6px);
  }}
  #legend {{ top:16px; left:16px; }}
  #legend h3 {{ margin-bottom:10px; font-size:14px; color:#bbb; }}
  .legend-item {{ display:flex; align-items:center; gap:8px; margin:5px 0; }}
  .legend-dot {{ width:11px; height:11px; border-radius:50%; flex-shrink:0; }}
  .legend-note {{ margin-top:10px; color:#666; font-size:11px; line-height:1.7; }}

  #filters {{
    position:fixed; width:260px; z-index:10;
    transition: top 0.4s ease, left 0.4s ease, right 0.4s ease, transform 0.4s ease, width 0.4s ease;
    /* 기본: 우상단 */
    top:16px; right:16px; left:auto; transform:none;
  }}
  body.explore-empty #filters {{
    top:50%; left:50%; right:auto;
    transform:translate(-50%, -50%);
    width:280px;
  }}
  #search-wrap {{ position:relative; display:flex; align-items:center; }}
  #search-input {{
    flex:1; background:#131326; color:#eee;
    border:1px solid #3a3a5a; border-radius:6px; padding:4px 8px;
    font-size:12px; outline:none;
  }}
  #search-input:focus {{ border-color:#6B9DFF; }}
  #search-input::placeholder {{ color:#555; }}
  #search-clear {{
    display:none; position:absolute; right:7px;
    background:none; border:none; color:#555; cursor:pointer;
    font-size:14px; line-height:1; padding:0;
  }}
  #search-clear:hover {{ color:#ccc; }}
  #search-suggestions {{
    display:none; list-style:none;
    background:#131326; border:1px solid #3a3a5a; border-radius:6px;
    margin-top:3px; max-height:160px; overflow-y:auto; font-size:12px;
  }}
  #search-suggestions li {{
    padding:5px 8px; cursor:pointer; color:#ccc;
    border-bottom:1px solid #1e1e36;
  }}
  #search-suggestions li:last-child {{ border-bottom:none; }}
  #search-suggestions li:hover {{ background:#1e1e42; color:#fff; }}
  #search-suggestions li span.artist-tag {{
    font-size:10px; color:#666; margin-left:5px;
  }}
  #search-suggestions li.sg-artist {{
    display:flex; align-items:center; gap:6px;
  }}
  #search-suggestions li.sg-artist input {{
    accent-color:#6B9DFF; cursor:pointer; flex-shrink:0; pointer-events:none;
  }}
  #search-suggestions li.sg-artist .check-dot {{
    width:8px; height:8px; border-radius:50%; flex-shrink:0;
  }}
  #search-suggestions .sg-divider {{
    padding:3px 8px; color:#555; font-size:10px; border-bottom:1px solid #1e1e36;
    pointer-events:none; user-select:none;
  }}
  #filters h3 {{ margin-bottom:12px; font-size:14px; color:#bbb; }}
  .fg {{ margin-bottom:13px; }}
  .fg label {{ display:block; color:#999; margin-bottom:3px; font-size:12px; }}
  .fg input[type=range] {{ width:100%; accent-color:#6B9DFF; cursor:pointer; }}
  .fg select {{
    width:100%; background:#131326; color:#eee;
    border:1px solid #3a3a5a; border-radius:6px; padding:4px 6px; font-size:12px;
  }}
  .vd {{ color:#6B9DFF; font-size:11px; }}

  #tooltip {{
    position:fixed; pointer-events:none; display:none;
    background:rgba(12,12,28,0.97); border:1px solid #3a3a6a;
    border-radius:11px; padding:11px 15px; font-size:12.5px; line-height:1.7;
    max-width:280px; z-index:20; box-shadow:0 6px 24px rgba(0,0,0,0.7);
  }}
  #ranking {{
    position:fixed; bottom:16px; right:16px; width:280px;
    background:rgba(18,18,36,0.93); border:1px solid #2a2a4a;
    border-radius:14px; padding:14px 18px; font-size:12px; z-index:10;
    backdrop-filter:blur(6px); display:none;
  }}
  #ranking h3 {{ margin-bottom:10px; font-size:13px; color:#bbb; }}
  #ranking .rank-item {{
    display:flex; align-items:center; gap:8px; padding:4px 0;
    border-bottom:1px solid #1e1e36; cursor:pointer;
  }}
  #ranking .rank-item:last-child {{ border-bottom:none; }}
  #ranking .rank-item:hover {{ color:#fff; }}
  #ranking .rank-num {{ color:#555; width:18px; text-align:right; flex-shrink:0; }}
  #ranking .rank-title {{ flex:1; color:#ccc; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  #ranking .rank-artist {{ color:#666; font-size:11px; flex-shrink:0; }}
  #ranking .rank-sim {{ color:#aaddff; font-size:11px; width:32px; text-align:right; flex-shrink:0; }}

  #stats {{
    position:fixed; bottom:16px; left:16px; display:flex; align-items:center; gap:10px;
    background:rgba(18,18,36,0.85); border:1px solid #2a2a4a;
    border-radius:10px; padding:8px 14px; font-size:12px; color:#777; z-index:10;
  }}
  #fit-btn, #reset-btn {{
    background:none; border:1px solid #3a3a5a; color:#777;
    border-radius:5px; padding:2px 8px; cursor:pointer; font-size:11px;
  }}
  #fit-btn:hover, #reset-btn:hover {{ color:#ccc; border-color:#6B9DFF; }}
  #reset-btn {{ display:none; border-color:#FF6B9D; color:#FF6B9D; }}
  #reset-btn:hover {{ border-color:#FF6B9D; color:#fff; background:rgba(255,107,157,0.15); }}

  #artist-checks {{ max-height:148px; overflow-y:auto; margin-top:4px; }}
  .check-item {{ display:flex; align-items:center; gap:6px; padding:2px 0; font-size:12px; color:#ccc; cursor:pointer; }}
  .check-item input {{ accent-color:#6B9DFF; cursor:pointer; flex-shrink:0; }}
  .check-dot {{ width:8px; height:8px; border-radius:50%; flex-shrink:0; }}
  .check-all {{ border-bottom:1px solid #2a2a4a; padding-bottom:5px; margin-bottom:2px; color:#999; }}

  .node circle {{ cursor:pointer; stroke:rgba(255,255,255,0.1); stroke-width:1px; transition:r 0.2s; }}
  .node circle:hover {{ stroke:white; stroke-width:2px; }}
  .node text {{ pointer-events:none; fill:#ccc; font-size:9px; }}

  /* 탐색 모드: 불필요한 패널 숨김 */
  body.explore-empty #legend,
  body.explore-empty #stats,
  body.explore-empty #fg-sim {{ display:none !important; }}
  body.explore-empty #filters h3 {{ display:none; }}
  body.explore-empty #filters {{ border-color:#3a3a5a; }}

  /* 탐색 모드 안내: 필터 패널 내부 */
  #explore-guide {{
    text-align:center; padding-bottom:10px; margin-bottom:10px;
    border-bottom:1px solid #2a2a4a;
  }}
  #explore-guide .guide-text {{
    color:#888; font-size:15px; line-height:1.8;
  }}
  #explore-guide .guide-sub {{
    color:#555; font-size:12px; margin-top:4px;
  }}

  /* 탐색 모드 노드 수 경고 */
  #explore-warn {{
    position:fixed; top:50%; left:50%; transform:translate(-50%, -50%);
    background:rgba(255,107,157,0.15); border:1px solid #FF6B9D;
    border-radius:10px; padding:10px 18px; font-size:12px; color:#FF6B9D;
    z-index:15; display:none; pointer-events:none;
  }}
</style>
</head>
<body>
<svg id="canvas"></svg>

<div id="explore-warn">⚠ 노드가 많습니다. 초기화 후 다시 탐색해보세요</div>

<!-- 범례 -->
<div class="panel" id="legend">
  <h3>아티스트</h3>
  <div id="legend-list"></div>
  <div class="legend-note">
    엣지 밝기 = 종합 유사도
  </div>
</div>

<!-- 필터 -->
<div class="panel" id="filters">
  <div id="explore-guide">
    <div class="guide-text">🔍 곡을 검색하거나<br>아티스트를 5개 이하로 선택하세요</div>
    <div class="guide-sub">곡을 선택하면 유사곡 네트워크가 펼쳐집니다</div>
  </div>
  <h3>필터</h3>
  <div class="fg">
    <label>노래 검색</label>
    <div id="search-wrap">
      <input type="text" id="search-input" placeholder="노래 또는 아티스트 검색..." autocomplete="off">
      <button id="search-clear">✕</button>
    </div>
    <ul id="search-suggestions"></ul>
  </div>
  <div class="fg" id="fg-sim">
    <label>최소 유사도: <span class="vd" id="sim-val">0.70</span></label>
    <input type="range" id="sim-filter" min="0.55" max="0.85" value="0.70" step="0.05">
  </div>
  <div class="fg" id="fg-artist">
    <label>아티스트</label>
    <label class="check-item check-all"><input type="checkbox" id="artist-all"> 전체 선택/해제</label>
    <div id="artist-checks">
{artist_checkboxes}
    </div>
  </div>
</div>


<div id="tooltip"></div>
<div id="ranking"><h3 id="ranking-title"></h3><div id="ranking-list"></div></div>
<div id="stats">노드: <b id="s-nodes">{len(nodes)}</b>  엣지: <b id="s-edges">{len(edges)}</b><button id="fit-btn">⊙ 화면 맞추기</button><button id="reset-btn">↺ 탐색 초기화</button></div>

<script>
console.log("=== V2 SCRIPT START ===");
const RAW_NODES = {json.dumps(nodes, ensure_ascii=False)};
const RAW_EDGES_BASE = {json.dumps(edges, ensure_ascii=False)};
const ARTIST_COLORS = {json.dumps(ARTIST_COLORS, ensure_ascii=False)};

// ── 색상 스케일: 유사도 0.35 ~ 1.0 → dim grey → bright gold ──
const simColor = d3.scaleSequential()
  .domain([0.35, 0.9])
  .interpolator(d3.interpolate("#2a2a4a", "#aaddff"));

const width  = window.innerWidth;
const height = window.innerHeight;

const svg = d3.select("#canvas").attr("width", width).attr("height", height);
const g   = svg.append("g");
const zoom = d3.zoom().scaleExtent([0.15, 6]).on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);

const linkLayer = g.append("g").attr("class", "links");
const nodeLayer = g.append("g").attr("class", "nodes");
const tooltip   = document.getElementById("tooltip");

let simulation = d3.forceSimulation()
  .force("link",      d3.forceLink().id(d => d.id).distance(d => 120 - d.similarity * 70))
  .force("charge",    d3.forceManyBody().strength(-70))
  .force("center",    d3.forceCenter(width/2, height/2))
  .force("collision", d3.forceCollide().radius(d => d.size + 5));

let linkSel, nodeSel;
let displayedNodes = [], displayedEdges = [];

// ── 모드 관리 ─────────────────────────────────────────────────
let currentMode = "explore";  // "full" | "explore"
let exploreNodes = new Set();  // 탐색 모드에서 표시 중인 노드 ID들
let exploreSeed = null;        // 탐색 시작 곡 ID

const exploreGuide = document.getElementById("explore-guide");
const exploreWarn  = document.getElementById("explore-warn");
const resetBtn     = document.getElementById("reset-btn");

function getMode() {{
  const checked = document.querySelectorAll(".artist-cb:checked").length;
  const total   = document.querySelectorAll(".artist-cb").length;
  // 전체 선택 → 전체 뷰 (전 아티스트 표시)
  if (checked === total) return "full";
  // 1~5개 선택 → 전체 뷰 (선택된 아티스트만)
  if (checked >= 1 && checked <= MAX_ARTIST_CHECK) return "full";
  // 0개 → 탐색 모드
  return "explore";
}}

function showExploreGuide(show) {{
  exploreGuide.style.display = show ? "block" : "none";
  document.body.classList.toggle("explore-empty", show);
}}

// ── 메인 렌더 ─────────────────────────────────────────────────
function update(nodes, edges) {{
  displayedNodes = nodes;
  const nodeMap    = new Map(nodes.map(n => [n.id, n]));
  const validEdges = edges
    .filter(e => nodeMap.has(e.source?.id ?? e.source) && nodeMap.has(e.target?.id ?? e.target))
    .map(e => ({{ ...e, source: e.source?.id ?? e.source, target: e.target?.id ?? e.target }}));
  displayedEdges = validEdges;

  // 링크
  linkSel = linkLayer.selectAll("line")
    .data(validEdges, d => d.source + "-" + d.target)
    .join("line")
    .attr("stroke",         d => simColor(d.similarity))
    .attr("stroke-width",   d => d.similarity * 2.5)
    .attr("stroke-opacity", d => 0.2 + d.similarity * 0.5)
    .on("mouseover", (event, d) => {{
      const na = nodeMap.get(d.source?.id ?? d.source);
      const nb = nodeMap.get(d.target?.id ?? d.target);
      if (!na || !nb) return;
      tooltip.style.display = "block";
      tooltip.innerHTML = `
        <b>${{na.label}}</b> ↔ <b>${{nb.label}}</b><br>
        <span style="color:#aaddff;font-size:13px">종합 유사도: ${{Math.round(d.similarity*100)}}%</span>`;
    }})
    .on("mousemove", ev => {{
      tooltip.style.left = (ev.clientX + 14) + "px";
      tooltip.style.top  = (ev.clientY - 10) + "px";
    }})
    .on("mouseout", () => {{ tooltip.style.display = "none"; }});

  // 노드
  nodeSel = nodeLayer.selectAll("g.node")
    .data(nodes, d => d.id)
    .join(
      enter => {{
        const ng = enter.append("g").attr("class", "node").call(drag(simulation));
        ng.append("circle");
        ng.append("text").attr("text-anchor", "middle");
        return ng;
      }}
    );

  nodeSel.select("circle")
    .attr("r",    d => d.size)
    .attr("fill", d => d.color)
    .on("mouseover", (event, d) => {{
      tooltip.style.display = "block";
      tooltip.innerHTML = d.tooltip;
    }})
    .on("mousemove", ev => {{
      tooltip.style.left = (ev.clientX + 14) + "px";
      tooltip.style.top  = (ev.clientY - 10) + "px";
    }})
    .on("mouseout", () => {{ tooltip.style.display = "none"; }})
    .on("click", (ev, d) => {{
      if (currentMode === "explore") {{
        expandExplore(d.id);
      }} else {{
        highlightNode(d, nodes, validEdges);
      }}
    }});

  nodeSel.select("text")
    .attr("dy", d => d.size + 12)
    .text(d => d.label.length > 14 ? d.label.slice(0,13) + "…" : d.label);

  // 기존 위치 유지, 시뮬레이션 부드럽게 재시작
  nodes.forEach(n => {{ n.fx = null; n.fy = null; }});
  simulation.nodes(nodes).on("tick", () => {{
    linkSel
      .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    nodeSel.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
  }});
  simulation.force("link").links(validEdges);
  simulation.alpha(currentMode === "explore" ? 0.08 : 0.15).restart();

  document.getElementById("s-nodes").textContent = nodes.length;
  document.getElementById("s-edges").textContent = validEdges.length;
}}

// ── 하이라이트 ───────────────────────────────────────────────
const rankingPanel = document.getElementById("ranking");
const rankingTitle = document.getElementById("ranking-title");
const rankingList  = document.getElementById("ranking-list");

let highlighted = null;
function highlightNode(d, nodes, edges) {{
  if (highlighted === d.id) {{
    highlighted = null;
    nodeSel.select("circle").attr("opacity", 1).style("stroke", null).style("stroke-width", null);
    nodeSel.select("text").style("opacity", null);
    linkSel.attr("stroke", e => simColor(e.similarity))
           .attr("stroke-opacity", e => 0.2 + e.similarity * 0.5);
    rankingPanel.style.display = "none";
    return;
  }}
  highlighted = d.id;
  const connectedIds = new Set([d.id]);
  const connectedEdges = [];
  edges.forEach(e => {{
    const s = e.source?.id ?? e.source, t = e.target?.id ?? e.target;
    if (s === d.id) {{ connectedIds.add(t); connectedEdges.push({{ id: t, sim: e.similarity }}); }}
    if (t === d.id) {{ connectedIds.add(s); connectedEdges.push({{ id: s, sim: e.similarity }}); }}
  }});

  // Top 10 유사곡 패널
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const top10 = connectedEdges.sort((a, b) => b.sim - a.sim).slice(0, 10);
  rankingTitle.textContent = `${{d.label}} — 유사곡 Top ${{top10.length}}`;
  rankingList.innerHTML = top10.map((item, i) => {{
    const n = nodeMap.get(item.id);
    if (!n) return "";
    return `<div class="rank-item" data-id="${{item.id}}">
      <span class="rank-num">${{i + 1}}</span>
      <span class="rank-title">${{n.label}}</span>
      <span class="rank-artist">${{n.artist}}</span>
      <span class="rank-sim">${{Math.round(item.sim * 100)}}%</span>
    </div>`;
  }}).join("");
  rankingPanel.style.display = "block";

  // 랭킹 항목 클릭 시 해당 노드 하이라이트
  rankingList.querySelectorAll(".rank-item").forEach(el => {{
    el.addEventListener("click", () => {{
      const targetId = +el.dataset.id;
      const target = displayedNodes.find(n => n.id === targetId);
      if (target) {{
        if (currentMode === "explore") {{
          expandExplore(targetId);
        }} else {{
          highlightNode(target, displayedNodes, displayedEdges);
        }}
      }}
    }});
  }});
  nodeSel.select("circle")
    .attr("opacity", n => connectedIds.has(n.id) ? 1 : 0.12)
    .style("stroke", n => n.id === d.id ? "white" : null)
    .style("stroke-width", n => n.id === d.id ? "2.5px" : null);
  nodeSel.select("text")
    .style("opacity", n => connectedIds.has(n.id) ? 1 : 0);
  nodeSel.filter(n => connectedIds.has(n.id)).raise();
  linkSel
    .attr("stroke", e => {{
      const s = e.source?.id ?? e.source, t = e.target?.id ?? e.target;
      return (s===d.id || t===d.id) ? "#FFD700" : "#2a2a4a";
    }})
    .attr("stroke-opacity", e => {{
      const s = e.source?.id ?? e.source, t = e.target?.id ?? e.target;
      return (s===d.id || t===d.id) ? 0.95 : 0;
    }});
  linkSel.filter(e => {{
    const s = e.source?.id ?? e.source, t = e.target?.id ?? e.target;
    return s===d.id || t===d.id;
  }}).raise();

  // 연결된 노드가 화면 밖일 수 있으므로 자동 화면 맞추기
  fitView();
}}

// ── 탐색 모드 ─────────────────────────────────────────────────
function getSimilarNodeIds(songId, limit) {{
  const minSim = +document.getElementById("sim-filter").value;
  return RAW_EDGES_BASE
    .filter(e => (e.source === songId || e.target === songId) && e.similarity >= minSim)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, limit || 10)
    .map(e => e.source === songId ? e.target : e.source);
}}

function startExplore(songId) {{
  // 아티스트 체크 전부 해제 → 전체 아티스트 대상 탐색
  document.querySelectorAll(".artist-cb").forEach(cb => {{ cb.checked = false; }});
  artistAllCb.checked = false;
  artistAllCb.indeterminate = false;
  currentMode = "explore";
  highlighted = null;
  rankingPanel.style.display = "none";
  exploreNodes.clear();
  exploreSeed = songId;
  exploreNodes.add(songId);
  getSimilarNodeIds(songId, 10).forEach(id => exploreNodes.add(id));
  renderExplore();
}}

function expandExplore(songId) {{
  if (!exploreNodes.has(songId)) return;
  // 클릭한 노드의 현재 위치 저장 (새 노드 초기 위치용)
  const anchor = displayedNodes.find(n => n.id === songId);
  const anchorX = anchor ? anchor.x : width / 2;
  const anchorY = anchor ? anchor.y : height / 2;

  const prevIds = new Set(exploreNodes);
  const newIds = getSimilarNodeIds(songId, 10);
  newIds.forEach(id => exploreNodes.add(id));

  if (exploreNodes.size > 100) {{
    exploreWarn.style.display = "block";
    setTimeout(() => {{ exploreWarn.style.display = "none"; }}, 2000);
  }}

  // 새 노드를 앵커 근처에 배치
  RAW_NODES.forEach(n => {{
    if (!prevIds.has(n.id) && exploreNodes.has(n.id)) {{
      n.x = anchorX + (Math.random() - 0.5) * 40;
      n.y = anchorY + (Math.random() - 0.5) * 40;
    }}
  }});

  renderExplore(songId);
}}

function renderExplore(highlightId) {{
  // 유사도 슬라이더 변경 시 seed 기준으로 새 노드 추가 (기존 노드는 유지)
  if (exploreSeed != null) {{
    const freshIds = getSimilarNodeIds(exploreSeed, 10);
    freshIds.forEach(id => exploreNodes.add(id));
  }}

  if (exploreNodes.size === 0) {{
    showExploreGuide(true);
    update([], []);
    resetBtn.style.display = "none";
    return;
  }}
  showExploreGuide(false);
  resetBtn.style.display = "inline-block";

  const minSim = +document.getElementById("sim-filter").value;

  const nodes = RAW_NODES.filter(n => exploreNodes.has(n.id));
  const visibleIds = new Set(nodes.map(n => n.id));
  const edges = RAW_EDGES_BASE.filter(e =>
    e.similarity >= minSim &&
    visibleIds.has(e.source) && visibleIds.has(e.target)
  );

  // 범례: 탐색 중인 노드의 아티스트만
  const visibleArtists = [...new Set(nodes.map(n => n.artist))];
  document.getElementById("legend-list").innerHTML = visibleArtists
    .map(a => `<div class="legend-item"><div class="legend-dot" style="background:${{ARTIST_COLORS[a] || '#AAAAAA'}}"></div>${{a}}</div>`)
    .join("");

  update(nodes, edges);

  // 지정된 노드 하이라이트
  const targetId = highlightId != null ? highlightId : exploreSeed;
  if (targetId != null) {{
    const target = displayedNodes.find(n => n.id === targetId);
    if (target) {{
      setTimeout(() => highlightNode(target, displayedNodes, displayedEdges), 100);
    }}
  }}
}}

function resetExplore() {{
  exploreNodes.clear();
  exploreSeed = null;
  highlighted = null;
  rankingPanel.style.display = "none";
  searchInput.value = "";
  searchClear.style.display = "none";
  renderExplore();
}}

resetBtn.addEventListener("click", resetExplore);

// ── 드래그 ───────────────────────────────────────────────────
function drag(sim) {{
  let dragging = false;
  return d3.drag()
    .on("start", (e,d) => {{ dragging = false; }})
    .on("drag",  (e,d) => {{
      if (!dragging) {{ dragging = true; if(!e.active) sim.alphaTarget(0.3).restart(); }}
      d.fx=e.x; d.fy=e.y;
    }})
    .on("end",   (e,d) => {{
      if(!e.active) sim.alphaTarget(0);
      d.fx=null; d.fy=null;
    }});
}}

// ── 필터 ─────────────────────────────────────────────────────
function applyFilters() {{
  const minSim = +document.getElementById("sim-filter").value;
  document.getElementById("sim-val").textContent = minSim.toFixed(2);

  const newMode = getMode();
  const modeChanged = newMode !== currentMode;
  currentMode = newMode;

  console.log("applyFilters: mode=", currentMode, "modeChanged=", modeChanged);
  if (currentMode === "explore") {{
    // 탐색 모드
    if (modeChanged) {{
      // 전체뷰에서 탐색 모드로 전환 시 리셋
      exploreNodes.clear();
      exploreSeed = null;
      highlighted = null;
      rankingPanel.style.display = "none";
    }}
    renderExplore();
    return;
  }}

  // ── 전체 뷰 모드 ──
  showExploreGuide(false);
  resetBtn.style.display = "none";

  const checkedArtists = new Set([...document.querySelectorAll(".artist-cb:checked")].map(cb => cb.value));
  const filteredNodes = RAW_NODES.filter(n =>
    (checkedArtists.size === 0 || checkedArtists.has(n.artist))
  );

  // 범례 업데이트: 현재 필터에 선택된 아티스트만 표시
  const visibleArtists = [...new Set(filteredNodes.map(n => n.artist))];
  document.getElementById("legend-list").innerHTML = visibleArtists
    .map(a => `<div class="legend-item"><div class="legend-dot" style="background:${{ARTIST_COLORS[a] || '#AAAAAA'}}"></div>${{a}}</div>`)
    .join("");
  const filteredIds   = new Set(filteredNodes.map(n => n.id));
  const filteredEdges = RAW_EDGES_BASE.filter(e =>
    e.similarity >= minSim &&
    filteredIds.has(e.source) && filteredIds.has(e.target)
  );
  const prevHighlighted = highlighted;
  highlighted = null;
  update(filteredNodes, filteredEdges);
  if (prevHighlighted != null) {{
    const target = displayedNodes.find(n => n.id === prevHighlighted);
    if (target) {{
      highlightNode(target, displayedNodes, displayedEdges);
    }} else {{
      // 하이라이트된 노드가 필터에서 빠졌으면 전체뷰로 리셋
      nodeSel.select("circle").attr("opacity", 1).style("stroke", null).style("stroke-width", null);
      nodeSel.select("text").style("opacity", null);
      linkSel.attr("stroke", e => simColor(e.similarity))
             .attr("stroke-opacity", e => 0.2 + e.similarity * 0.5);
      rankingPanel.style.display = "none";
    }}
  }}
}}

document.getElementById("sim-filter").addEventListener("input", applyFilters);

// ── 아티스트 체크박스 ─────────────────────────────────────────
const MAX_ARTIST_CHECK = 5;
const artistAllCb = document.getElementById("artist-all");

function showArtistWarn() {{
  exploreWarn.textContent = `⚠ 아티스트는 최대 ${{MAX_ARTIST_CHECK}}개까지 선택할 수 있습니다`;
  exploreWarn.style.display = "block";
  setTimeout(() => {{ exploreWarn.style.display = "none"; }}, 2000);
}}

function syncArtistAll() {{
  const all = document.querySelectorAll(".artist-cb");
  const checked = document.querySelectorAll(".artist-cb:checked");
  artistAllCb.indeterminate = checked.length > 0 && checked.length < all.length;
  artistAllCb.checked = checked.length === all.length;
}}

artistAllCb.addEventListener("change", () => {{
  if (artistAllCb.checked) {{
    // 전체 선택 차단
    artistAllCb.checked = false;
    showArtistWarn();
    return;
  }}
  // 전체 해제는 허용
  document.querySelectorAll(".artist-cb").forEach(cb => {{ cb.checked = false; }});
  applyFilters();
}});
document.querySelectorAll(".artist-cb").forEach(cb => {{
  cb.addEventListener("change", () => {{
    const checkedCount = document.querySelectorAll(".artist-cb:checked").length;
    if (checkedCount > MAX_ARTIST_CHECK) {{
      cb.checked = false;
      showArtistWarn();
      return;
    }}
    syncArtistAll();
    applyFilters();
  }});
}});

// ── 검색 ─────────────────────────────────────────────────────
const searchInput = document.getElementById("search-input");
const searchClear = document.getElementById("search-clear");
const suggestions = document.getElementById("search-suggestions");

function clearHighlight() {{
  highlighted = null;
  if (nodeSel) {{
    nodeSel.select("circle").attr("opacity", 1).style("stroke", null).style("stroke-width", null);
    nodeSel.select("text").style("opacity", null);
  }}
  if (linkSel) linkSel.attr("stroke-opacity", e => 0.2 + e.similarity * 0.5)
                      .attr("stroke", e => simColor(e.similarity));
}}

// 아티스트 목록 (체크박스 value에서 추출)
const ALL_ARTISTS = [...document.querySelectorAll(".artist-cb")].map(cb => ({{
  name: cb.value, cb,
  color: ARTIST_COLORS[cb.value] || '#AAAAAA'
}}));

searchInput.addEventListener("input", () => {{
  const q = searchInput.value.toLowerCase().trim();
  suggestions.innerHTML = "";
  searchClear.style.display = q ? "block" : "none";
  if (!q) {{
    suggestions.style.display = "none";
    if (currentMode === "full") clearHighlight();
    return;
  }}

  // 아티스트 매칭
  const artistMatches = ALL_ARTISTS.filter(a => a.name.toLowerCase().includes(q)).slice(0, 5);
  // 곡 매칭
  const songMatches = RAW_NODES.filter(n => n.label.toLowerCase().includes(q)).slice(0, 10);

  if (artistMatches.length === 0 && songMatches.length === 0) {{
    suggestions.style.display = "none";
    return;
  }}

  // 아티스트 결과
  if (artistMatches.length > 0) {{
    const divider = document.createElement("li");
    divider.className = "sg-divider";
    divider.textContent = "아티스트";
    suggestions.appendChild(divider);
    artistMatches.forEach(a => {{
      const li = document.createElement("li");
      li.className = "sg-artist";
      const checked = a.cb.checked ? "checked" : "";
      li.innerHTML = `<input type="checkbox" ${{checked}}><span class="check-dot" style="background:${{a.color}}"></span>${{a.name}}`;
      li.addEventListener("mousedown", (e) => {{
        e.preventDefault();
        const willCheck = !a.cb.checked;
        if (willCheck) {{
          const checkedCount = document.querySelectorAll(".artist-cb:checked").length;
          if (checkedCount >= MAX_ARTIST_CHECK) {{
            showArtistWarn();
            return;
          }}
        }}
        a.cb.checked = willCheck;
        li.querySelector("input").checked = a.cb.checked;
        syncArtistAll();
        applyFilters();
      }});
      suggestions.appendChild(li);
    }});
  }}

  // 곡 결과
  if (songMatches.length > 0) {{
    const divider = document.createElement("li");
    divider.className = "sg-divider";
    divider.textContent = "노래";
    suggestions.appendChild(divider);
    songMatches.forEach(n => {{
      const li = document.createElement("li");
      li.innerHTML = n.label + `<span class="artist-tag">${{n.artist}}</span>`;
      li.addEventListener("mousedown", () => {{
        searchInput.value = n.label;
        suggestions.style.display = "none";

        startExplore(n.id);
      }});
      suggestions.appendChild(li);
    }});
  }}
  suggestions.style.display = "block";
}});

searchInput.addEventListener("blur", () => {{
  setTimeout(() => {{ suggestions.style.display = "none"; }}, 150);
}});

searchClear.addEventListener("click", () => {{
  searchInput.value = "";
  searchClear.style.display = "none";
  suggestions.style.display = "none";
  if (currentMode === "explore") {{
    resetExplore();
  }} else {{
    clearHighlight();
  }}
  searchInput.focus();
}});

// ── 화면 맞추기 ───────────────────────────────────────────────
function fitView() {{
  let targetNodes = displayedNodes;
  if (highlighted != null) {{
    const connectedIds = new Set([highlighted]);
    displayedEdges.forEach(e => {{
      const s = e.source?.id ?? e.source, t = e.target?.id ?? e.target;
      if (s === highlighted) connectedIds.add(t);
      if (t === highlighted) connectedIds.add(s);
    }});
    targetNodes = displayedNodes.filter(n => connectedIds.has(n.id));
  }}
  const xs = targetNodes.map(n => n.x).filter(v => v != null);
  const ys = targetNodes.map(n => n.y).filter(v => v != null);
  if (!xs.length) return;
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = 60;
  const scale = Math.min((width - pad*2) / (x1 - x0 || 1), (height - pad*2) / (y1 - y0 || 1), 3);
  const tx = width/2  - scale * (x0 + x1) / 2;
  const ty = height/2 - scale * (y0 + y1) / 2;
  svg.transition().duration(600)
    .call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}}
document.getElementById("fit-btn").addEventListener("click", fitView);

// 초기 렌더
console.log("=== ABOUT TO CALL applyFilters ===");
applyFilters();
console.log("=== INIT DONE ===");
</script>
<script data-goatcounter="https://kpop-network.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>
</body>
</html>
"""

out_path = "index.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print("saved: " + out_path)
