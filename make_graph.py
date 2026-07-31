import pandas as pd
import json

# ─── 데이터 로드 ──────────────────────────────────────────────

df = pd.read_csv("songs_tagged.csv", encoding="utf-8-sig")
df = df.fillna("")

# ─── 아티스트별 색상 ──────────────────────────────────────────

ARTIST_COLORS = {
    "NewJeans": "#FF6B9D",
    "투모로우바이투게더": "#6B9DFF",
    "연준": "#A8D0FF",
    "KATSEYE": "#FFD700",
    "IVE (아이브)": "#FF9966",
    "범규": "#3560C0",
    "Hearts2Hearts (하츠투하츠)": "#48C9B0",
    "KiiKii (키키)": "#F8C471",
    "NMIXX": "#DA70D6",
    "aespa": "#00E5FF",
    "LE SSERAFIM (르세라핌)": "#FF8C42",
    "NCT WISH": "#7CFC00",
    "TWS (투어스)": "#7ED4AD",
    "아일릿(ILLIT)": "#FF85A2",
    "BABYMONSTER": "#E74C3C",
    "RIIZE": "#5B9BD5",
    "CORTIS (코르티스)": "#B39DDB",   # 새로 추가
    "Billlie (빌리)": "#F06292",      # 새로 추가
    "SHINee (샤이니)": "#3EE6C4",      # 새로 추가 (민트)
    "RESCENE (리센느)": "#7C4DFF",     # 새로 추가 (인디고 바이올렛)
    "Red Velvet (레드벨벳)": "#B71C4A", # 새로 추가 (딥 크림슨)
}

DEFAULT_COLOR = "#AAAAAA"

# ─── 장르 패밀리 (같은 패밀리면 부분 유사도 부여) ────────────────

GENRE_FAMILY = {
    # Electronic Dance (하우스 계열)
    "dance pop": "electronic_dance",
    "house": "electronic_dance",
    "uk garage": "electronic_dance",
    "jersey club": "electronic_dance",
    "drum & bass": "electronic_dance",
    "breakbeat": "electronic_dance",
    "edm": "electronic_dance",
    "rave": "electronic_dance",
    "hardstyle": "electronic_dance",
    "electronic dance": "electronic_dance",
    "deep house": "electronic_dance",

    # Synth / Electropop
    "synthpop": "synth_pop",
    "new wave": "synth_pop",
    "hyperpop": "synth_pop",
    "city pop": "synth_pop",
    "dream pop": "synth_pop",

    # Disco / Funk
    "disco": "disco_funk",
    "funk pop": "disco_funk",
    "new jack swing": "disco_funk",
    "miami bass": "disco_funk",

    # Global Dance (라틴/아프로 계열)
    "dancehall": "global_dance",
    "moombahton": "global_dance",
    "reggaeton": "global_dance",
    "brazilian funk": "global_dance",
    "balie funk": "global_dance",
    "afrobeats": "global_dance",
    "latin pop": "global_dance",

    # Hip Hop / R&B
    "hip hop": "hip_hop_rb",
    "trap": "hip_hop_rb",
    "old-school hip hop": "hip_hop_rb",
    "r&b": "hip_hop_rb",
    "alternative r&b": "hip_hop_rb",
    "soul": "hip_hop_rb",

    # Rock
    "pop rock": "rock",
    "hard rock": "rock",
    "indie rock": "rock",
    "alternative rock": "rock",
    "emo rock": "rock",
    "britpop": "rock",
    "pop punk": "rock",
    "punk rock": "rock",
    "stadium rock": "rock",
    "reggae rock": "rock",

    # Soft Pop / Ballad
    "ballad": "soft_pop",
    "indie pop": "soft_pop",
    "alternative pop": "soft_pop",
    "acoustic pop": "soft_pop",
    "pop": "soft_pop",

    # Global Dance (라틴/아프로 계열) — 추가
    "reggae pop": "global_dance",

    # Hip Hop / R&B — 추가
    "boom bap": "hip_hop_rb",

    # Electronic Dance — 추가
    "french house": "electronic_dance",

    # Synth / Electropop — 추가
    "lo-fi": "synth_pop",

    # Rock — 추가
    "punk pop": "rock",

    # Hip Hop / R&B — 추가
    "rage": "hip_hop_rb",
    "rock and roll": "rock",

    # ── 신규 곡 반영 추가분 ──
    # Electronic Dance
    "dubstep": "electronic_dance",
    "techno": "electronic_dance",
    "melodic techno": "electronic_dance",
    "trance": "electronic_dance",
    "latin house": "electronic_dance",

    # Synth / Electropop
    "electropop": "synth_pop",
    "future bass": "synth_pop",

    # Global Dance (baile funk = balie funk 정상 철자)
    "baile funk": "global_dance",

    # Rock
    "psychedelic rock": "rock",

    # Jazz / Soul (재즈·보사노바·두왑 — 어쿠스틱 화성 + 스윙 계열)
    "jazz": "jazz_soul",
    "bossa nova": "jazz_soul",
    "doo-wap": "jazz_soul",

    # Acoustic Roots (컨트리 등 루츠 계열)
    "country": "acoustic_roots",

    # Orchestral (단독 패밀리)
    "orchestral": "orchestral",

    # ── 신규 곡 반영 추가분 (2차) ──
    "alternative hip hop": "hip_hop_rb",
    "funk rock": "rock",
    "rap rock": "rock",

    # ── 신규 곡 반영 추가분 (3차) ──
    "contemporary r&b": "hip_hop_rb",   # R&B 계열
    "doo-wop": "jazz_soul",             # 기존 doo-wap과 동일 계열 (정식 철자)
    "electro funk": "synth_pop",        # 신스 기반 (Knock On Wood, sub도 synthpop)
    "ragga": "global_dance",            # 레게/댄스홀 계열
    "waltz": "soft_pop",                # 팝 발라드 리듬 요소 (Remember Forever)

    # ── 신규 곡 반영 추가분 (4차) ──
    "samba": "global_dance",            # 브라질 계열 (brazilian funk 등과 동일)
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
        "id":     int(row["song_id"]),
        "label":  row["title"],
        "artist": row["artist"],
        "genres": genres,
        "moods":  moods,
        "energy": energy,
        "brightness": brightness,
        "color":  ARTIST_COLORS.get(row["artist"], DEFAULT_COLOR),
        "size":   9,
        "tooltip": (
            f"<b>{row['title']}</b><br>"
            f"🎤 {row['artist']}<br>"
            f"{album_line}"
            f"🎸 {' / '.join(genres) if genres else '-'}"
        ),
    })

# ─── 유사도 계산 ──────────────────────────────────────────────

W = {"genre": 0.40, "energy": 0.20, "brightness": 0.20, "mood": 0.20}
MIN_THRESHOLD = 0.55

def genre_sim(a, b):
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
    all_g  = set(wa) | set(wb)
    inter  = sum(min(wa.get(g, 0), wb.get(g, 0)) for g in all_g)
    union  = sum(max(wa.get(g, 0), wb.get(g, 0)) for g in all_g)
    return inter / union if union else 0.0

def num_sim(a, b, scale=4):
    return 1.0 - abs(a - b) / scale

# 엣지는 더 이상 빌드 타임에 계산하지 않는다.
# 브라우저에서 탐색 시점에 실시간으로 계산한다 (아래 JS의 computeSimilarity 참고).
# 파이썬 함수는 JS 포팅의 기준(reference)으로만 남겨둔다.
print(f"nodes={len(nodes)} (엣지는 브라우저에서 실시간 계산, threshold>={MIN_THRESHOLD})")

# ─── 아티스트 체크박스 & 범례 HTML 사전 생성 ─────────────────────

active_artists = sorted(
    [a for a in ARTIST_COLORS if a in df["artist"].values],
    key=lambda x: (x[0].isascii(), x)
)

artist_checkboxes = "\n".join(
    f'  <label class="check-item">'
    f'<input type="checkbox" class="artist-cb" value="{a}">'
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
  position:fixed; width:280px; z-index:10;
  transition: top 0.4s ease, left 0.4s ease, right 0.4s ease, transform 0.4s ease, width 0.4s ease;
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
#search-suggestions:not(.kb-nav) li:hover, #search-suggestions li.sg-active {{ background:#1e1e42; color:#fff; }}
#search-suggestions li span.artist-tag {{
  font-size:10px; color:#666; margin-left:5px;
}}
#search-suggestions li.sg-artist {{
  display:flex; align-items:center; gap:6px;
}}
#search-suggestions li.sg-artist input {{
  accent-color:#6B9DFF; cursor:pointer; flex-shrink:0; pointer-events:none;
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
.fg .check-item {{ display:flex; align-items:center; gap:6px; padding:5px 8px; font-size:12px; color:#ccc; cursor:pointer; }}
.check-item input {{ accent-color:#6B9DFF; cursor:pointer; flex-shrink:0; }}
.node circle {{ cursor:pointer; stroke:rgba(255,255,255,0.1); stroke-width:1px; transition:r 0.2s; }}
.node circle:hover {{ stroke:white; stroke-width:2px; }}
.node text {{ pointer-events:none; fill:#ccc; font-size:9px; }}
body.explore-empty #legend,
body.explore-empty #stats,
body.explore-empty #fg-sim {{ display:none !important; }}
body.explore-empty #filters h3 {{ display:none; }}
body.explore-empty #filters {{ border-color:#3a3a5a; }}
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
    <div class="guide-text">🔍 곡을 검색하거나<br>아티스트를 선택하세요</div>
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
    <div id="artist-checks">
{artist_checkboxes}
    </div>
  </div>
</div>

<div id="tooltip"></div>
<div id="ranking"><h3 id="ranking-title"></h3><div id="ranking-list"></div></div>
<div id="stats">노드: <b id="s-nodes">{len(nodes)}</b> 엣지: <b id="s-edges">0</b><button id="fit-btn">⊙ 화면 맞추기</button><button id="reset-btn">↺ 탐색 초기화</button></div>

<script>
const RAW_NODES = {json.dumps(nodes, ensure_ascii=False)};
const ARTIST_COLORS = {json.dumps(ARTIST_COLORS, ensure_ascii=False)};

// ─── 실시간 유사도 계산 (파이썬 로직을 JS로 포팅) ───────────────
const GENRE_FAMILY = {json.dumps(GENRE_FAMILY, ensure_ascii=False)};
const FAMILY_WEIGHT = {FAMILY_WEIGHT};
const W = {json.dumps(W, ensure_ascii=False)};
const MIN_THRESHOLD = {MIN_THRESHOLD};

function buildGenreWeights(genres) {{
  const w = {{}};
  genres.forEach((gname, i) => {{
    w[gname] = (i === 0) ? 2 : 1;
    const fam = GENRE_FAMILY[gname];
    if (fam) w[fam] = Math.max(w[fam] || 0, FAMILY_WEIGHT);
  }});
  return w;
}}

function genreSim(a, b) {{
  if (!a.length || !b.length) return 0.0;
  const wa = buildGenreWeights(a), wb = buildGenreWeights(b);
  const allG = new Set([...Object.keys(wa), ...Object.keys(wb)]);
  let inter = 0, union = 0;
  allG.forEach(gname => {{
    const va = wa[gname] || 0, vb = wb[gname] || 0;
    inter += Math.min(va, vb);
    union += Math.max(va, vb);
  }});
  return union ? inter / union : 0.0;
}}

function numSim(a, b, scale=4) {{ return 1.0 - Math.abs(a - b) / scale; }}

// 두 노드 간 종합 유사도 (파이썬과 동일 로직). 임계값 미만이면 null.
function pairSimilarity(na, nb) {{
  const gSim = genreSim(na.genres, nb.genres);
  const eSim = numSim(na.energy, nb.energy);
  const bSim = numSim(na.brightness, nb.brightness);
  const bothMoodEmpty = na.moods.length === 0 && nb.moods.length === 0;
  let total, mSim;
  if (bothMoodEmpty) {{
    total = (W.genre*gSim + W.energy*eSim + W.brightness*bSim) / (1 - W.mood);
    mSim = 0.0;
  }} else {{
    const sa = new Set(na.moods), sb = new Set(nb.moods);
    const interM = [...sa].filter(m => sb.has(m)).length;
    const unionM = new Set([...sa, ...sb]).size;
    mSim = unionM ? interM / unionM : 0.0;
    total = W.genre*gSim + W.energy*eSim + W.brightness*bSim + W.mood*mSim;
  }}
  return total;
}}

const NODE_BY_ID = new Map(RAW_NODES.map(n => [n.id, n]));

// 한 곡과 다른 모든 곡의 유사도를 실시간 계산해, 임계값 이상을 내림차순 반환
function similaritiesFrom(songId) {{
  const src = NODE_BY_ID.get(songId);
  if (!src) return [];
  const out = [];
  for (const other of RAW_NODES) {{
    if (other.id === songId) continue;
    const sim = pairSimilarity(src, other);
    if (sim >= MIN_THRESHOLD) out.push({{ id: other.id, similarity: sim }});
  }}
  out.sort((a, b) => b.similarity - a.similarity);
  return out;
}}

// 주어진 노드 집합 내부의 모든 엣지를 실시간 계산 (보이는 노드끼리만)
function edgesAmong(nodeList, minSim) {{
  const edges = [];
  for (let i = 0; i < nodeList.length; i++) {{
    for (let j = i + 1; j < nodeList.length; j++) {{
      const sim = pairSimilarity(nodeList[i], nodeList[j]);
      if (sim >= minSim) {{
        edges.push({{ source: nodeList[i].id, target: nodeList[j].id, similarity: sim }});
      }}
    }}
  }}
  return edges;
}}

const simColor = d3.scaleSequential()
  .domain([0.35, 0.9])
  .interpolator(d3.interpolate("#2a2a4a", "#aaddff"));

const width = window.innerWidth;
const height = window.innerHeight;
const svg = d3.select("#canvas").attr("width", width).attr("height", height);
const g = svg.append("g");
const zoom = d3.zoom().scaleExtent([0.15, 6]).on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);

const linkLayer = g.append("g").attr("class", "links");
const nodeLayer = g.append("g").attr("class", "nodes");
const tooltip = document.getElementById("tooltip");

let simulation = d3.forceSimulation()
  .force("link", d3.forceLink().id(d => d.id).distance(d => 120 - d.similarity * 70))
  .force("charge", d3.forceManyBody().strength(-70))
  .force("center", d3.forceCenter(width/2, height/2))
  .force("collision", d3.forceCollide().radius(d => d.size + 5));

let linkSel, nodeSel;
let displayedNodes = [], displayedEdges = [];

let currentMode = "explore";
let exploreNodes = new Set();
let exploreSeed = null;
const exploreGuide = document.getElementById("explore-guide");
const exploreWarn  = document.getElementById("explore-warn");
const resetBtn     = document.getElementById("reset-btn");

function getMode() {{
  const checked = document.querySelectorAll(".artist-cb:checked").length;
  // 아티스트가 1개 선택되면 해당 아티스트 보기(full), 아니면 탐색(explore)
  return checked >= 1 ? "full" : "explore";
}}

function showExploreGuide(show) {{
  exploreGuide.style.display = show ? "block" : "none";
  document.body.classList.toggle("explore-empty", show);
}}

function update(nodes, edges) {{
  displayedNodes = nodes;
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const validEdges = edges
    .filter(e => nodeMap.has(e.source?.id ?? e.source) && nodeMap.has(e.target?.id ?? e.target))
    .map(e => ({{ ...e, source: e.source?.id ?? e.source, target: e.target?.id ?? e.target }}));
  displayedEdges = validEdges;

  linkSel = linkLayer.selectAll("line")
    .data(validEdges, d => d.source + "-" + d.target)
    .join("line")
    .attr("stroke", d => simColor(d.similarity))
    .attr("stroke-width", d => d.similarity * 2.5)
    .attr("stroke-opacity", d => 0.2 + d.similarity * 0.5)
    .on("mouseover", (event, d) => {{
      const na = nodeMap.get(d.source?.id ?? d.source);
      const nb = nodeMap.get(d.target?.id ?? d.target);
      if (!na || !nb) return;
      tooltip.style.display = "block";
      tooltip.innerHTML = `<b>${{na.label}}</b> ↔ <b>${{nb.label}}</b><br><span style="color:#aaddff;font-size:13px">종합 유사도: ${{Math.round(d.similarity*100)}}%</span>`;
    }})
    .on("mousemove", ev => {{
      tooltip.style.left = (ev.clientX + 14) + "px";
      tooltip.style.top  = (ev.clientY - 10) + "px";
    }})
    .on("mouseout", () => {{ tooltip.style.display = "none"; }});

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
    .attr("r", d => d.size)
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
  nodeSel.select("text").style("opacity", n => connectedIds.has(n.id) ? 1 : 0);
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

  fitView();
}}

function getSimilarNodeIds(songId, limit) {{
  const minSim = +document.getElementById("sim-filter").value;
  return similaritiesFrom(songId)
    .filter(e => e.similarity >= minSim)
    .slice(0, limit || 10)
    .map(e => e.id);
}}

function startExplore(songId) {{
  document.querySelectorAll(".artist-cb").forEach(cb => {{ cb.checked = false; }});
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
  const anchor = displayedNodes.find(n => n.id === songId);
  const anchorX = anchor ? anchor.x : width / 2;
  const anchorY = anchor ? anchor.y : height / 2;
  const prevIds = new Set(exploreNodes);
  const newIds  = getSimilarNodeIds(songId, 10);
  newIds.forEach(id => exploreNodes.add(id));
  if (exploreNodes.size > 100) {{
    exploreWarn.style.display = "block";
    setTimeout(() => {{ exploreWarn.style.display = "none"; }}, 2000);
  }}
  RAW_NODES.forEach(n => {{
    if (!prevIds.has(n.id) && exploreNodes.has(n.id)) {{
      n.x = anchorX + (Math.random() - 0.5) * 40;
      n.y = anchorY + (Math.random() - 0.5) * 40;
    }}
  }});
  renderExplore(songId);
}}

function renderExplore(highlightId) {{
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
  const edges = edgesAmong(nodes, minSim);
  const visibleArtists = [...new Set(nodes.map(n => n.artist))];
  document.getElementById("legend-list").innerHTML = visibleArtists
    .map(a => `<div class="legend-item"><div class="legend-dot" style="background:${{ARTIST_COLORS[a] || '#AAAAAA'}}"></div>${{a}}</div>`)
    .join("");
  update(nodes, edges);
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
  // 아티스트 선택도 모두 해제하고 탐색 초기 상태로
  document.querySelectorAll(".artist-cb").forEach(cb => {{ cb.checked = false; }});
  currentMode = "explore";
  renderExplore();
}}
resetBtn.addEventListener("click", resetExplore);

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

function fitView() {{
  if (!displayedNodes.length) return;
  const xs = displayedNodes.map(n => n.x).filter(Boolean);
  const ys = displayedNodes.map(n => n.y).filter(Boolean);
  if (!xs.length) return;
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = 60;
  const scale = Math.min(0.9, Math.min(
    (width  - pad*2) / (x1 - x0 || 1),
    (height - pad*2) / (y1 - y0 || 1)
  ));
  const tx = width/2  - scale*(x0+x1)/2;
  const ty = height/2 - scale*(y0+y1)/2;
  svg.transition().duration(600)
    .call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}}
document.getElementById("fit-btn").addEventListener("click", fitView);

function applyFilters() {{
  const minSim = +document.getElementById("sim-filter").value;
  document.getElementById("sim-val").textContent = minSim.toFixed(2);
  const newMode = getMode();
  const modeChanged = newMode !== currentMode;
  currentMode = newMode;

  if (currentMode === "explore") {{
    if (modeChanged) {{
      exploreNodes.clear();
      exploreSeed = null;
      highlighted = null;
      rankingPanel.style.display = "none";
    }}
    renderExplore();
    return;
  }}

  showExploreGuide(false);
  resetBtn.style.display = "inline-block";
  const checkedArtists = new Set([...document.querySelectorAll(".artist-cb:checked")].map(cb => cb.value));
  const filteredNodes = RAW_NODES.filter(n =>
    (checkedArtists.size === 0 || checkedArtists.has(n.artist))
  );
  const visibleArtists = [...new Set(filteredNodes.map(n => n.artist))];
  document.getElementById("legend-list").innerHTML = visibleArtists
    .map(a => `<div class="legend-item"><div class="legend-dot" style="background:${{ARTIST_COLORS[a] || '#AAAAAA'}}"></div>${{a}}</div>`)
    .join("");
  const filteredIds = new Set(filteredNodes.map(n => n.id));
  const filteredEdges = edgesAmong(filteredNodes, minSim);
  const prevHighlighted = highlighted;
  highlighted = null;
  update(filteredNodes, filteredEdges);
  if (prevHighlighted != null) {{
    const target = displayedNodes.find(n => n.id === prevHighlighted);
    if (target) {{
      highlightNode(target, displayedNodes, displayedEdges);
    }} else {{
      nodeSel.select("circle").attr("opacity", 1).style("stroke", null).style("stroke-width", null);
      nodeSel.select("text").style("opacity", null);
      linkSel.attr("stroke", e => simColor(e.similarity))
             .attr("stroke-opacity", e => 0.2 + e.similarity * 0.5);
      rankingPanel.style.display = "none";
    }}
  }}
}}

document.getElementById("sim-filter").addEventListener("input", applyFilters);

document.querySelectorAll(".artist-cb").forEach(cb => {{
  cb.addEventListener("change", () => {{
    if (cb.checked) {{
      // 라디오처럼 동작: 다른 아티스트 선택은 모두 해제
      document.querySelectorAll(".artist-cb").forEach(other => {{
        if (other !== cb) other.checked = false;
      }});
    }}
    applyFilters();
  }});
}});

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

const ALL_ARTISTS = [...document.querySelectorAll(".artist-cb")].map(cb => ({{
  name: cb.value, cb
}}));

let sgIndex = -1;

function getSelectableItems() {{
  return [...suggestions.querySelectorAll("li:not(.sg-divider)")];
}}

function updateSgActive() {{
  const items = getSelectableItems();
  items.forEach((li, i) => li.classList.toggle("sg-active", i === sgIndex));
  if (sgIndex >= 0 && items[sgIndex]) items[sgIndex].scrollIntoView({{ block: "nearest" }});
}}

searchInput.addEventListener("keydown", (e) => {{
  if (suggestions.style.display === "none") return;
  const items = getSelectableItems();
  if (!items.length) return;
  if (e.key === "ArrowDown") {{
    e.preventDefault();
    suggestions.classList.add("kb-nav");
    sgIndex = (sgIndex + 1) % items.length;
    updateSgActive();
  }} else if (e.key === "ArrowUp") {{
    e.preventDefault();
    suggestions.classList.add("kb-nav");
    sgIndex = (sgIndex - 1 + items.length) % items.length;
    updateSgActive();
  }} else if (e.key === "Enter") {{
    e.preventDefault();
    if (sgIndex >= 0 && items[sgIndex]) {{
      items[sgIndex].dispatchEvent(new Event("mousedown"));
    }}
  }}
}});

searchInput.addEventListener("input", () => {{
  sgIndex = -1;
  // 곡선 따옴표(’ ‘ ` ´)와 직선 따옴표(')를 같은 것으로 취급해 검색 누락 방지
  const normalize = s => s.toLowerCase().replace(/[\u2018\u2019\u0060\u00b4]/g, "'");
  const q = normalize(searchInput.value).trim();
  suggestions.innerHTML = "";
  searchClear.style.display = q ? "block" : "none";
  if (!q) {{
    suggestions.style.display = "none";
    if (currentMode === "full") clearHighlight();
    return;
  }}
  const artistMatches = ALL_ARTISTS.filter(a => normalize(a.name).includes(q)).slice(0, 5);
  const songMatches   = RAW_NODES.filter(n => normalize(n.label).includes(q)).slice(0, 10);
  if (artistMatches.length === 0 && songMatches.length === 0) {{
    suggestions.style.display = "none";
    return;
  }}
  if (artistMatches.length > 0) {{
    const divider = document.createElement("li");
    divider.className = "sg-divider";
    divider.textContent = "아티스트";
    suggestions.appendChild(divider);
    artistMatches.forEach(a => {{
      const li = document.createElement("li");
      li.className = "sg-artist";
      const checked = a.cb.checked ? "checked" : "";
      li.innerHTML = `<input type="checkbox" ${{checked}}>${{a.name}}`;
      li.addEventListener("mousedown", (e) => {{
        e.preventDefault();
        const willCheck = !a.cb.checked;
        if (willCheck) {{
          // 라디오처럼 동작: 다른 아티스트 선택은 모두 해제
          document.querySelectorAll(".artist-cb").forEach(other => {{
            if (other !== a.cb) other.checked = false;
          }});
        }}
        a.cb.checked = willCheck;
        li.querySelector("input").checked = a.cb.checked;
        applyFilters();
      }});
      suggestions.appendChild(li);
    }});
  }}
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
  suggestions.addEventListener("mousemove", () => {{
    suggestions.classList.remove("kb-nav");
    sgIndex = -1;
    getSelectableItems().forEach(li => li.classList.remove("sg-active"));
  }}, {{ once: true }});
}});

searchInput.addEventListener("blur", () => {{
  setTimeout(() => {{ suggestions.style.display = "none"; }}, 150);
}});

searchClear.addEventListener("click", () => {{
  searchInput.value = "";
  searchClear.style.display = "none";
  suggestions.style.display = "none";
  if (currentMode === "full") clearHighlight();
}});

// 초기 상태: 탐색 모드 (아무것도 선택 안 됨)
showExploreGuide(true);
update([], []);

</script>
</body>
</html>"""

with open("kpop_network.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ kpop_network.html 생성 완료!")
