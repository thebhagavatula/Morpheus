"""
Matrix Theme — CSS & Plotly template for the dashboard

Provides the digital rain animation (pure CSS), custom Streamlit styling,
and a reusable Plotly chart template with the Matrix colour scheme.
"""

# ── Colour Palette ────────────────────────────────────────────────────
MATRIX_BLACK = "#0a0a0a"
MATRIX_BG    = "#0d0d0d"
MATRIX_CARD  = "#111111"
MATRIX_GREEN = "#00ff41"
MATRIX_DIM   = "#00cc33"
MATRIX_DARK  = "#003b00"
MATRIX_AMBER = "#ffb000"
MATRIX_RED   = "#ff2040"
MATRIX_GREY  = "#1a1a1a"
MATRIX_TEXT  = "#b0ffb0"
MATRIX_MUTED = "#4fa84f"   # >=7:1 on MATRIX_BLACK (was #2a6e2a, 3.2:1)

# Colours for chart series
CHART_COLORS = [
    MATRIX_GREEN, MATRIX_AMBER, "#00e5ff", "#ff6ec7",
    "#7fff00", "#ffd700", "#00fa9a", "#ff4500",
]


# ── Digital Rain (CSS-only, no JS) ────────────────────────────────────
# Uses CSS columns + keyframe animations to create the iconic falling
# green characters.  Runs as a fixed background overlay in the app.

_RAIN_CHARS = "01ﾊﾐﾋｰｳｼﾅﾓﾆｻﾜﾂｵﾘｱﾎﾃﾏｹﾒｴｶｷﾑﾕﾗｾﾈｽﾀﾇ"

def _rain_columns_css(n_cols: int = 30) -> str:
    """Generate CSS for N falling character columns."""
    import random
    cols = []
    for i in range(n_cols):
        left = (i / n_cols) * 100
        delay = random.uniform(0, 8)
        duration = random.uniform(6, 14)
        char = random.choice(_RAIN_CHARS)
        opacity = random.uniform(0.06, 0.16)
        cols.append(
            f'.rain-col-{i} {{'
            f'  left: {left:.1f}%;'
            f'  animation-delay: {delay:.1f}s;'
            f'  animation-duration: {duration:.1f}s;'
            f'  opacity: {opacity:.2f};'
            f'}}'
            f'.rain-col-{i}::after {{ content: "{char}"; }}'
        )
    return "\n".join(cols)


def _rain_html(n_cols: int = 30) -> str:
    """Returns HTML div elements for the rain columns."""
    divs = "".join(f'<div class="rain-col rain-col-{i}"></div>' for i in range(n_cols))
    return f'<div class="matrix-rain">{divs}</div>'


MATRIX_RAIN_CSS = f"""
/* ── Matrix Digital Rain ────────────────────────────────────────── */
.matrix-rain {{
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
    user-select: none;
}}

.rain-col {{
    position: absolute;
    top: -20px;
    font-family: 'Courier New', 'MS Gothic', monospace;
    font-size: 14px;
    color: {MATRIX_GREEN};
    text-shadow: 0 0 8px {MATRIX_GREEN}, 0 0 20px {MATRIX_DARK};
    writing-mode: vertical-rl;
    animation: fall linear infinite;
}}

.rain-col::after {{
    content: "0";
}}

@keyframes fall {{
    0%   {{ transform: translateY(-100vh); opacity: 1; }}
    75%  {{ opacity: 0.5; }}
    100% {{ transform: translateY(110vh); opacity: 0; }}
}}

{_rain_columns_css(30)}
"""


# ── Accessibility (WCAG 2.2 AA) ───────────────────────────────────────
A11Y_CSS = f"""
/* Keep all page content above the decorative rain overlay */
.block-container, section[data-testid="stSidebar"] > div {{
    position: relative;
    z-index: 1;
}}

/* Visible keyboard focus (WCAG 2.4.7) */
:focus-visible,
.stButton > button:focus-visible,
.stTabs [data-baseweb="tab"]:focus-visible,
div[role="radiogroup"] label:focus-within,
div[data-baseweb="select"]:focus-within {{
    outline: 2px solid {MATRIX_AMBER} !important;
    outline-offset: 2px;
}}

/* Body copy: comfortable size and line height */
.stApp p, .stApp li, .stMarkdown {{
    font-size: 1rem;
    line-height: 1.6;
}}
.stTabs [data-baseweb="tab"] {{ font-size: 0.95rem !important; }}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    text-shadow: 0 0 6px rgba(0, 255, 65, 0.4) !important;
}}

/* Respect reduced-motion preference (WCAG 2.3.3) */
@media (prefers-reduced-motion: reduce) {{
    .matrix-rain {{ display: none !important; }}
    .glow {{ animation: none !important; }}
    * {{ transition: none !important; animation: none !important; }}
}}
"""


# ── Streamlit Page CSS ────────────────────────────────────────────────
STREAMLIT_CSS = f"""
<style>
/* ── Global ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

.stApp {{
    background-color: {MATRIX_BLACK};
    color: {MATRIX_TEXT};
    font-family: 'Share Tech Mono', 'Courier New', monospace;
}}

/* Hide Streamlit chrome — keep header visible so sidebar toggle works */
#MainMenu, footer {{ visibility: hidden; }}
header {{ background: transparent !important; }}
.stDeployButton {{ display: none; }}

/* Keep sidebar collapse/expand button always accessible */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="collapsedControl"],
[data-testid="collapsedControl"] {{
    visibility: visible !important;
    z-index: 999 !important;
    pointer-events: auto !important;
    color: {MATRIX_GREEN} !important;
    background: {MATRIX_CARD} !important;
    border: 1px solid {MATRIX_DARK} !important;
    border-radius: 4px;
}}
button[data-testid="stSidebarCollapseButton"]:hover,
button[data-testid="collapsedControl"]:hover,
[data-testid="collapsedControl"]:hover {{
    background: {MATRIX_DARK} !important;
    box-shadow: 0 0 10px {MATRIX_GREEN};
}}

/* ── Sidebar ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {MATRIX_BG} 0%, {MATRIX_BLACK} 100%);
    border-right: 1px solid {MATRIX_DARK};
}}
section[data-testid="stSidebar"] * {{
    color: {MATRIX_TEXT} !important;
}}

/* ── Cards ──────────────────────────────────────────────────────── */
div[data-testid="stMetric"] {{
    background: {MATRIX_CARD};
    border: 1px solid {MATRIX_DARK};
    border-radius: 4px;
    padding: 16px;
    box-shadow: 0 0 15px rgba(0, 255, 65, 0.08);
}}
div[data-testid="stMetric"] label {{
    color: {MATRIX_DIM} !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 13px !important;
}}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    color: {MATRIX_GREEN} !important;
    font-family: 'Orbitron', monospace !important;
    text-shadow: 0 0 10px {MATRIX_GREEN};
    font-size: 28px !important;
}}
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] > div {{
    font-family: 'Share Tech Mono', monospace !important;
}}

/* ── Headers ────────────────────────────────────────────────────── */
h1, h2, h3 {{
    font-family: 'Orbitron', sans-serif !important;
    color: {MATRIX_GREEN} !important;
    text-shadow: 0 0 8px rgba(0, 255, 65, 0.45);
    letter-spacing: 2px;
    line-height: 1.3;
}}
h1 {{
    font-size: 2.2rem !important;
    border-bottom: 1px solid {MATRIX_DARK};
    padding-bottom: 10px;
}}

/* ── Tabs ───────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: {MATRIX_CARD};
    border-radius: 4px;
    gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    color: {MATRIX_DIM} !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 2px solid transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {MATRIX_GREEN} !important;
    border-bottom: 2px solid {MATRIX_GREEN} !important;
    text-shadow: 0 0 8px {MATRIX_GREEN};
}}

/* ── Tables ─────────────────────────────────────────────────────── */
.stDataFrame, .stTable {{
    border: 1px solid {MATRIX_DARK} !important;
}}
.stDataFrame td, .stDataFrame th {{
    color: {MATRIX_TEXT} !important;
    background: {MATRIX_CARD} !important;
    border-color: {MATRIX_DARK} !important;
    font-family: 'Share Tech Mono', monospace !important;
}}
.stDataFrame th {{
    color: {MATRIX_GREEN} !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* ── Buttons & Inputs ───────────────────────────────────────────── */
.stButton > button {{
    background: {MATRIX_CARD} !important;
    color: {MATRIX_GREEN} !important;
    border: 1px solid {MATRIX_GREEN} !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    transition: all 0.3s ease;
}}
.stButton > button:hover {{
    background: {MATRIX_DARK} !important;
    box-shadow: 0 0 20px {MATRIX_GREEN}, inset 0 0 20px rgba(0,255,65,0.1);
}}

.stSelectbox > div > div {{
    background: {MATRIX_CARD} !important;
    color: {MATRIX_GREEN} !important;
    border-color: {MATRIX_DARK} !important;
}}

/* ── Plotly dark overrides ──────────────────────────────────────── */
.js-plotly-plot .plotly .modebar {{
    background: transparent !important;
}}
.js-plotly-plot .plotly .modebar .modebar-btn {{
    color: {MATRIX_DIM} !important;
}}

/* ── Scrollbar ──────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {MATRIX_BLACK}; }}
::-webkit-scrollbar-thumb {{
    background: {MATRIX_DARK};
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{ background: {MATRIX_DIM}; }}

/* ── Glow animation for key elements ───────────────────────────── */
@keyframes glow-pulse {{
    0%, 100% {{ text-shadow: 0 0 10px {MATRIX_GREEN}, 0 0 20px {MATRIX_DARK}; }}
    50%      {{ text-shadow: 0 0 20px {MATRIX_GREEN}, 0 0 40px {MATRIX_GREEN}, 0 0 60px {MATRIX_DARK}; }}
}}
.glow {{ animation: glow-pulse 2s ease-in-out infinite; }}

/* ── Scanline overlay ──────────────────────────────────────────── */
.scanlines {{
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: -1;
    background: repeating-linear-gradient(
        0deg,
        rgba(0, 0, 0, 0.08) 0px,
        rgba(0, 0, 0, 0.08) 1px,
        transparent 1px,
        transparent 3px
    );
}}

/* ── Smooth transitions for all interactive elements ───────────── */
*, *::before, *::after {{
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}
.stButton > button, div[data-testid="stMetric"],
.stSelectbox, section[data-testid="stSidebar"],
.stTabs [data-baseweb="tab"] {{
    transition: all 0.25s ease-in-out !important;
}}

/* ── Live clock styling ────────────────────────────────────────── */
#matrix-clock {{
    font-family: 'Orbitron', monospace;
    color: {MATRIX_GREEN};
    font-size: 1.1rem;
    letter-spacing: 2px;
    text-shadow: 0 0 10px {MATRIX_GREEN}, 0 0 20px {MATRIX_DARK};
    text-align: center;
    padding: 8px 0;
}}
#matrix-clock .clock-label {{
    font-family: 'Share Tech Mono', monospace;
    color: {MATRIX_DIM};
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    display: block;
    margin-bottom: 4px;
    text-shadow: none;
}}

/* ── Index selector styling ────────────────────────────────────── */
.stSelectbox label {{
    color: {MATRIX_DIM} !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-size: 0.85rem !important;
}}
.stRadio label {{
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 0.5px !important;
    font-size: 1rem !important;
}}

{A11Y_CSS}

{MATRIX_RAIN_CSS}
</style>
"""


# ── Plotly Template ───────────────────────────────────────────────────
def get_plotly_template():
    """Returns a Plotly layout template matching the Matrix theme."""
    import plotly.graph_objects as go

    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=MATRIX_BLACK,
            plot_bgcolor=MATRIX_BG,
            font=dict(family="Share Tech Mono, Courier New, monospace", color=MATRIX_TEXT, size=13),
            title=dict(font=dict(family="Orbitron, sans-serif", color=MATRIX_GREEN, size=18)),
            xaxis=dict(
                gridcolor=MATRIX_GREY, zerolinecolor=MATRIX_DARK,
                tickfont=dict(color=MATRIX_DIM),
            ),
            yaxis=dict(
                gridcolor=MATRIX_GREY, zerolinecolor=MATRIX_DARK,
                tickfont=dict(color=MATRIX_DIM),
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0.6)", bordercolor=MATRIX_DARK,
                font=dict(color=MATRIX_TEXT),
            ),
            colorway=CHART_COLORS,
        )
    )


def matrix_line_chart(fig):
    """Apply Matrix glow effect to line traces."""
    for trace in fig.data:
        if hasattr(trace, 'line'):
            color = trace.line.color or MATRIX_GREEN
            trace.update(line=dict(width=2))
    return fig


def live_clock_html() -> str:
    """
    Returns an HTML/JS snippet that renders a live ticking clock.
    Uses setInterval to update every second — no Streamlit rerun needed.
    Inject via st.components.v1.html().
    """
    return f"""
    <div id="matrix-clock"
         style="font-family: 'Orbitron', monospace;
                color: {MATRIX_GREEN};
                font-size: 1.1rem;
                letter-spacing: 2px;
                text-shadow: 0 0 10px {MATRIX_GREEN}, 0 0 20px {MATRIX_DARK};
                text-align: center;
                padding: 4px 0;
                background: transparent;">
        <span class="clock-label"
              style="font-family: 'Share Tech Mono', monospace;
                     color: {MATRIX_DIM};
                     font-size: 0.75rem;
                     text-transform: uppercase;
                     letter-spacing: 3px;
                     display: block;
                     margin-bottom: 2px;
                     text-shadow: none;">System Time</span>
        <span id="clock-digits">--:--:--</span>
    </div>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <script>
    function updateClock() {{
        const now = new Date();
        const h = String(now.getHours()).padStart(2, '0');
        const m = String(now.getMinutes()).padStart(2, '0');
        const s = String(now.getSeconds()).padStart(2, '0');
        document.getElementById('clock-digits').textContent = h + ':' + m + ':' + s;
    }}
    updateClock();
    setInterval(updateClock, 1000);
    </script>
    """

