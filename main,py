import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ritonavir – Molecular Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #0a0e1a;
    color: #e8f4f8;
}
.stApp { background: #0a0e1a; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1525;
    border-right: 1px solid #1e3a5f;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a2a4a 40%, #0e1a35 100%);
    border: 1px solid #1e4d7b;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,200,255,0.08) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    color: #00d4ff;
    letter-spacing: -1px;
    margin: 0;
    text-shadow: 0 0 40px rgba(0,212,255,0.3);
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    color: #7fbcd2;
    font-size: 0.85rem;
    margin-top: 6px;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.credit-badge {
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 8px;
    padding: 8px 16px;
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #7fbcd2;
    margin-top: 20px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #0f1e38, #0a1628);
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: #00d4ff; }
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    color: #00d4ff;
    font-weight: 700;
}
.metric-label {
    font-size: 0.78rem;
    color: #7fbcd2;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.35rem;
    color: #00d4ff;
    border-left: 4px solid #00d4ff;
    padding-left: 14px;
    margin: 28px 0 16px 0;
    letter-spacing: -0.3px;
}

/* SMILES box */
.smiles-box {
    background: #050d1a;
    border: 1px solid #1a3a5c;
    border-radius: 10px;
    padding: 16px 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #4dffc3;
    word-break: break-all;
    line-height: 1.7;
}

/* Chiral table */
.chiral-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
}
.chiral-table th {
    color: #7fbcd2;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 0.7rem;
    padding: 0 16px 8px 16px;
    border-bottom: 1px solid #1a3a5c;
}
.chiral-table td {
    padding: 10px 16px;
    background: #0f1e38;
    color: #e8f4f8;
}
.chiral-table tr td:first-child { border-radius: 8px 0 0 8px; }
.chiral-table tr td:last-child  { border-radius: 0 8px 8px 0; }
.badge-S {
    background: rgba(0,212,255,0.15);
    border: 1px solid #00d4ff;
    border-radius: 6px;
    padding: 3px 10px;
    color: #00d4ff;
}
.badge-R {
    background: rgba(255,100,100,0.12);
    border: 1px solid #ff6464;
    border-radius: 6px;
    padding: 3px 10px;
    color: #ff8080;
}

/* Info pills */
.info-pill {
    display: inline-block;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    margin: 4px 4px;
    font-size: 0.8rem;
    color: #a8ddf0;
}

/* Viewer container */
.viewer-container {
    background: #050d1a;
    border: 1px solid #1a3a5c;
    border-radius: 14px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ─── Data ────────────────────────────────────────────────────────────────────
SMILES = (
    "CC(C)c1nc(N2CC(O)C(Cc3ccccc3)NC(=O)c3nc(C(C)C)c(-c4ccccn4)s3)CS1(=O)(=O)"
    "... "  # placeholder – correct SMILES below
)
SMILES_CORRECT = (
    "CC(C)[C@@H]1Nc2nc(C(C)C)c(-c3ccccn3)s2C(=O)N[C@H](Cc2ccccc2)[C@@H](O)C[C@@H]2CC(=O)N(C[C@@H]12)C(=O)NC(C)(C)C"
)
# The canonical isomeric SMILES from PubChem CID 392622
SMILES_FULL = (
    "CC(C)[C@@H]1Nc2nc(C(C)C)c(-c3ccccn3)s2C(=O)N[C@H](Cc2ccccc2)"
    "[C@@H](O)C[C@@H]2CC(=O)N(C[C@@H]12)C(=O)NC(C)(C)C"
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-family:Syne;font-weight:800;font-size:1.2rem;color:#00d4ff;margin-bottom:4px'>🧬 Ritonavir</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem;color:#7fbcd2;margin-bottom:20px;font-family:Space Mono'>Molecular Explorer v1.0</div>", unsafe_allow_html=True)
    st.markdown("---")
    nav = st.radio(
        "Navigate",
        ["Overview", "3D Structure", "2D Structure", "Chirality & Stereocenters", "Pharmacology", "Graphs & Charts"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem;color:#4a6b88;font-family:Space Mono;line-height:1.8'>
    <b style='color:#7fbcd2'>Drug class</b><br>HIV Protease Inhibitor<br><br>
    <b style='color:#7fbcd2'>CAS No.</b><br>155213-67-5<br><br>
    <b style='color:#7fbcd2'>PubChem CID</b><br>392622<br><br>
    <b style='color:#7fbcd2'>ChEMBL ID</b><br>CHEMBL163<br><br>
    <b style='color:#7fbcd2'>DrugBank</b><br>DB00503
    </div>
    """, unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">Ritonavir</div>
  <div class="hero-sub">HIV-1 Protease Inhibitor · Antiretroviral Agent · CYP3A4 Inhibitor</div>
  <div class="credit-badge">
    👤 Prepared by <b>Saksham Malviya</b> &nbsp;|&nbsp; Reg No: <b>RA2511026050017</b>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if nav == "Overview":
    # Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        ("720.94 g/mol", "Molecular Weight"),
        ("C₃₇H₄₈N₆O₅S₂", "Molecular Formula"),
        ("5", "Chiral Centers"),
        ("−0.27", "logP (XLogP3)"),
        ("147.7 Å²", "TPSA"),
    ]
    for col, (val, label) in zip([c1,c2,c3,c4,c5], metrics):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">{val}</div>
          <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">SMILES Notation</div>', unsafe_allow_html=True)
    smiles_display = "CC(C)[C@@H]1Nc2nc(C(C)C)c(-c3ccccn3)s2C(=O)N[C@H](Cc2ccccc2)[C@@H](O)C[C@@H]2CC(=O)N(C[C@@H]12)C(=O)NC(C)(C)C"
    st.markdown(f'<div class="smiles-box">{smiles_display}</div>', unsafe_allow_html=True)
    st.caption("Isomeric SMILES (PubChem CID 392622) — @@ / @ denote stereocenters")

    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        st.markdown('<div class="section-header">General Information</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style='line-height:1.85;color:#c0daea;font-size:0.9rem'>
        <b style='color:#00d4ff'>Ritonavir</b> (brand name <i>Norvir</i>) is an antiretroviral drug used to treat HIV/AIDS.
        It was developed by Abbott Laboratories and FDA-approved in <b>1996</b>, making it one of the first protease inhibitors
        on the market. Ritonavir works by inhibiting the HIV-1 <b>protease enzyme</b>, preventing the cleavage of
        polyprotein precursors and thus blocking the maturation of infectious virions.
        </p>
        <p style='line-height:1.85;color:#c0daea;font-size:0.9rem'>
        A landmark application is its role as a <b>pharmacokinetic booster</b>: at sub-therapeutic doses it potently inhibits
        CYP3A4, dramatically increasing plasma concentrations of co-administered protease inhibitors. This approach
        is central to regimens like lopinavir/ritonavir (Kaletra).
        </p>
        <p style='line-height:1.85;color:#c0daea;font-size:0.9rem'>
        Ritonavir famously exhibited <b>polymorphism</b> — a new crystalline Form II emerged in 1998 causing global 
        drug shortages, and became a textbook case in pharmaceutical solid-state chemistry.
        </p>
        """, unsafe_allow_html=True)

        tags = ["HIV Protease Inhibitor", "CYP3A4 Inhibitor", "Antiretroviral", "Peptidomimetic",
                "Norvir®", "FDA 1996", "Polymorphic", "BCS Class II"]
        pills = "".join(f'<span class="info-pill">{t}</span>' for t in tags)
        st.markdown(pills, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header">Physicochemical Properties</div>', unsafe_allow_html=True)
        props = {
            "Property": ["Molecular Weight", "Formula", "H-Bond Donors", "H-Bond Acceptors",
                          "Rotatable Bonds", "XLogP3", "TPSA", "Heavy Atom Count",
                          "Stereocenters", "Ring Count", "Complexity"],
            "Value": ["720.94 g/mol", "C₃₇H₄₈N₆O₅S₂", "4", "11",
                      "14", "−0.27", "147.7 Å²", "50",
                      "5 (all defined)", "5", "1380"]
        }
        df_props = pd.DataFrame(props)
        st.dataframe(
            df_props,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('<div class="section-header">Lipinski\'s Rule of Five Analysis</div>', unsafe_allow_html=True)
    ro5_col1, ro5_col2 = st.columns(2)
    with ro5_col1:
        rules = ["MW ≤ 500 Da", "HBD ≤ 5", "HBA ≤ 10", "LogP ≤ 5"]
        vals  = [720.94, 4, 11, -0.27]
        limits= [500, 5, 10, 5]
        colors= ["#ff6464" if v > l else "#4dffc3" for v, l in zip(vals, limits)]
        fig_ro5 = go.Figure()
        for r, v, l, c in zip(rules, vals, limits, colors):
            fig_ro5.add_trace(go.Bar(
                x=[r], y=[v],
                marker_color=c, name=r, showlegend=False,
                text=[f"{v}"], textposition="outside",
            ))
            fig_ro5.add_shape(type="line", x0=-0.4+rules.index(r), x1=0.4+rules.index(r),
                              y0=l, y1=l, line=dict(color="white", dash="dot", width=1.5))
        fig_ro5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c0daea", family="Space Mono"),
            yaxis=dict(gridcolor="#1a3a5c", showgrid=True),
            xaxis=dict(showgrid=False),
            margin=dict(t=20, b=10, l=0, r=0), height=260,
        )
        st.plotly_chart(fig_ro5, use_container_width=True)
        st.caption("⚠️ Ritonavir violates MW and HBA rules — it's a 'beyond Rule of Five' compound.")

elif nav == "3D Structure":
    st.markdown('<div class="section-header">Interactive 3D Molecular Viewer</div>', unsafe_allow_html=True)

    viewer_code = """
<!DOCTYPE html>
<html>
<head>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  body { margin: 0; background: #050d1a; }
  #mol_container { width: 100%; height: 490px; position: relative; }
  .btn-box {
    position: absolute; bottom: 16px; left: 50%;
    transform: translateX(-50%);
    display: flex; gap: 10px; z-index: 100;
  }
  .m-btn {
    background: #0f1e38; border: 1px solid #00d4ff; color: #00d4ff;
    padding: 8px 18px; border-radius: 5px; cursor: pointer;
    font-family: monospace; font-size: 11px; text-transform: uppercase;
  }
  .m-btn:hover, .m-btn.active { background: rgba(0,212,255,0.22); }
  .tag {
    position: absolute; top: 12px; left: 16px; z-index: 10;
    background: rgba(5,13,26,0.85); border: 1px solid #00d4ff;
    color: #00d4ff; font-size: 10px; font-family: monospace;
    letter-spacing: 2px; padding: 4px 10px; border-radius: 4px;
  }
  .hint {
    position: absolute; top: 12px; right: 16px; z-index: 10;
    background: rgba(5,13,26,0.75); color: #4a7a99;
    font-size: 10px; font-family: monospace; padding: 4px 10px;
    border-radius: 4px;
  }
</style>
</head>
<body>
<div id="mol_container">
  <div class="tag">RITONAVIR — 3D INTERACTIVE</div>
  <div class="hint">Drag · Scroll · Click atoms</div>
</div>
<div class="btn-box">
  <button class="m-btn active" id="btn-stick"  onclick="setV('stick')">Stick</button>
  <button class="m-btn"        id="btn-sphere" onclick="setV('sphere')">Sphere</button>
  <button class="m-btn"        id="btn-line"   onclick="setV('line')">Line</button>
  <button class="m-btn"        id="btn-surf"   onclick="toggleSurf()">Surface</button>
</div>

<script>
var glviewer = null;
var surfOn   = false;
var curStyle = 'stick';

// Ritonavir SMILES — 3Dmol will parse and generate 3D coords via its
// internal coordinate generator (same as friend's PubChem fetch fallback)
var smiles = "CC(C)[C@@H]1Nc2nc(C(C)C)c(-c3ccccn3)s2C(=O)N[C@H](Cc2ccccc2)[C@@H](O)C[C@@H]2CC(=O)N(C[C@@H]12)C(=O)NC(C)(C)C";

function applyStyle(style) {
  glviewer.setStyle({}, {});
  if (style === 'stick')
    glviewer.setStyle({}, {
      stick:  {colorscheme: 'Jmol', radius: 0.15},
      sphere: {colorscheme: 'Jmol', radius: 0.32}
    });
  else if (style === 'sphere')
    glviewer.setStyle({}, {sphere: {colorscheme: 'Jmol'}});
  else if (style === 'line')
    glviewer.setStyle({}, {line: {colorscheme: 'Jmol', linewidth: 3}});
  glviewer.render();
}

function setV(style) {
  curStyle = style;
  document.querySelectorAll('.m-btn').forEach(function(b) { b.classList.remove('active'); });
  document.getElementById('btn-' + style).classList.add('active');
  applyStyle(style);
}

function toggleSurf() {
  surfOn = !surfOn;
  document.getElementById('btn-surf').classList.toggle('active', surfOn);
  glviewer.removeAllSurfaces();
  if (surfOn) {
    glviewer.addSurface($3Dmol.SurfaceType.VDW, {
      opacity: 0.28,
      colorscheme: {gradient: 'rwb'}
    });
  }
  glviewer.render();
}

window.onload = function() {
  setTimeout(function() {
    glviewer = $3Dmol.createViewer(
      document.getElementById('mol_container'),
      {backgroundColor: '#050d1a'}
    );

    // Primary: fetch real 3D SDF from PubChem (works when internet available)
    fetch('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/' +
          encodeURIComponent(smiles) + '/SDF?record_type=3d')
      .then(function(r) { return r.text(); })
      .then(function(sdf) {
        glviewer.addModel(sdf, 'sdf');
        applyStyle('stick');
        glviewer.zoomTo();
        glviewer.render();
      })
      .catch(function() {
        // Fallback: add via SMILES — 3Dmol generates coords internally
        glviewer.addModel(smiles, 'smi');
        applyStyle('stick');
        glviewer.zoomTo();
        glviewer.render();
      });

  }, 250);
};
</script>
</body>
</html>
"""
    components.html(viewer_code, height=520)

    st.markdown('<div class="section-header">Atom Legend</div>', unsafe_allow_html=True)
    st.write("⚫ Carbon | 🔵 Nitrogen | 🔴 Oxygen | 🟡 Sulfur")
# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 2D STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "2D Structure":
    st.markdown('<div class="section-header">2D Structure Viewer</div>', unsafe_allow_html=True)

    viewer_2d = """
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
      <script src="https://unpkg.com/kekule/dist/kekule.min.js?modules=chemWidget,io"></script>
      <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { background:#050d1a; display:flex; flex-direction:column; align-items:center; padding:20px; }

        .panel-title {
          font-family: monospace;
          color: #00d4ff;
          font-size: 1rem;
          margin-bottom: 16px;
          letter-spacing: 2px;
        }

        /* We'll render using 3Dmol 2D  */
        #mol2d { width: 680px; height: 460px; border: 1px solid #1a3a5c; border-radius: 12px; background:#0a1628; }

        .smiles-area {
          margin-top: 16px;
          background: #050d1a;
          border: 1px solid #1a3a5c;
          border-radius: 10px;
          padding: 14px 20px;
          font-family: monospace;
          font-size: 0.7rem;
          color: #4dffc3;
          word-break: break-all;
          width: 680px;
          line-height: 1.8;
        }
        .legend {
          margin-top: 14px;
          font-family: monospace;
          font-size: 0.7rem;
          color: #7fbcd2;
          text-align: center;
        }
      </style>
    </head>
    <body>
      <div class="panel-title">RITONAVIR — 2D DEPICTION</div>

      <!-- Use SDF image from PubChem as fallback-safe approach -->
      <div id="mol2d" style="display:flex;align-items:center;justify-content:center;overflow:hidden;">
        <img
          src="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/392622/PNG?record_type=2d&image_size=large"
          alt="Ritonavir 2D Structure"
          style="max-width:100%;max-height:100%;object-fit:contain;border-radius:10px;filter:invert(1) hue-rotate(180deg) brightness(0.85) contrast(1.2);"
          onerror="this.onerror=null;this.src='https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/ritonavir/PNG';"
        />
      </div>

      <div class="smiles-area">
        <b>Isomeric SMILES:</b><br>
        CC(C)[C@@H]1Nc2nc(C(C)C)c(-c3ccccn3)s2C(=O)N[C@H](Cc2ccccc2)[C@@H](O)C[C@@H]2CC(=O)N(C[C@@H]12)C(=O)NC(C)(C)C
      </div>

      <div class="legend">
        ★ Chiral centres marked with @@ (S) and @ (R) in SMILES notation<br>
        Image source: PubChem CID 392622 — colour inverted for dark theme
      </div>
    </body>
    </html>
    """
    components.html(viewer_2d, height=620, scrolling=False)

    st.markdown('<div class="section-header">SMILES Breakdown</div>', unsafe_allow_html=True)
    st.markdown("""
    <table style='width:100%;font-family:Space Mono;font-size:0.78rem;border-collapse:separate;border-spacing:0 5px;color:#c0daea'>
    <tr style='color:#7fbcd2;font-size:0.65rem;text-transform:uppercase;letter-spacing:1px'>
      <th style='padding:6px 14px;border-bottom:1px solid #1a3a5c'>Fragment</th>
      <th style='padding:6px 14px;border-bottom:1px solid #1a3a5c'>Description</th>
      <th style='padding:6px 14px;border-bottom:1px solid #1a3a5c'>Atoms</th>
    </tr>
    <tr style='background:#0f1e38'><td style='padding:9px 14px;border-radius:8px 0 0 8px;color:#4dffc3'>CC(C)</td><td style='padding:9px 14px'>Isopropyl group</td><td style='padding:9px 14px;border-radius:0 8px 8px 0'>C₃</td></tr>
    <tr style='background:#0f1e38'><td style='padding:9px 14px;border-radius:8px 0 0 8px;color:#4dffc3'>-c3ccccn3-</td><td style='padding:9px 14px'>Pyridine ring</td><td style='padding:9px 14px;border-radius:0 8px 8px 0'>C₅N</td></tr>
    <tr style='background:#0f1e38'><td style='padding:9px 14px;border-radius:8px 0 0 8px;color:#4dffc3'>s2C(=O)N</td><td style='padding:9px 14px'>Thiazole carbonyl + amide</td><td style='padding:9px 14px;border-radius:0 8px 8px 0'>CSO·N</td></tr>
    <tr style='background:#0f1e38'><td style='padding:9px 14px;border-radius:8px 0 0 8px;color:#4dffc3'>Cc2ccccc2</td><td style='padding:9px 14px'>Benzyl group (phenyl)</td><td style='padding:9px 14px;border-radius:0 8px 8px 0'>C₇</td></tr>
    <tr style='background:#0f1e38'><td style='padding:9px 14px;border-radius:8px 0 0 8px;color:#4dffc3'>[C@@H](O)</td><td style='padding:9px 14px'>Hydroxyl-bearing chiral carbon (S)</td><td style='padding:9px 14px;border-radius:0 8px 8px 0'>C·OH</td></tr>
    <tr style='background:#0f1e38'><td style='padding:9px 14px;border-radius:8px 0 0 8px;color:#4dffc3'>NC(C)(C)C</td><td style='padding:9px 14px'>tert-Butyl carbamate</td><td style='padding:9px 14px;border-radius:0 8px 8px 0'>C₄N</td></tr>
    </table>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CHIRALITY
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "Chirality & Stereocenters":
    st.markdown('<div class="section-header">Chiral Centers — Complete Map</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#a0c8e0;font-size:0.88rem;line-height:1.8;margin-bottom:20px'>
    Ritonavir has <b style='color:#00d4ff'>5 defined stereocenters</b>, all carbon atoms. They are encoded in the
    SMILES via the <code style='color:#4dffc3'>@@</code> (S configuration) and <code style='color:#ff8080'>@</code> (R configuration)
    notations. These stereocenters are critical for the molecule's binding affinity to HIV-1 protease.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <table class="chiral-table">
      <thead>
        <tr>
          <th>Center #</th>
          <th>SMILES Token</th>
          <th>Configuration</th>
          <th>Environment</th>
          <th>Importance</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>C-1</td>
          <td style='color:#4dffc3'>[C@@H]</td>
          <td><span class="badge-S">S</span></td>
          <td>Adjacent to isopropyl & thiazole N</td>
          <td>Thiazole ring junction — key scaffold geometry</td>
        </tr>
        <tr>
          <td>C-2</td>
          <td style='color:#4dffc3'>[C@H]</td>
          <td><span class="badge-R">R</span></td>
          <td>Benzyl side-chain bearing carbon</td>
          <td>Fits P1 pocket of HIV protease</td>
        </tr>
        <tr>
          <td>C-3</td>
          <td style='color:#4dffc3'>[C@@H]</td>
          <td><span class="badge-S">S</span></td>
          <td>Hydroxyl group — transition state mimic</td>
          <td>Mimics tetrahedral intermediate of protease cleavage</td>
        </tr>
        <tr>
          <td>C-4</td>
          <td style='color:#4dffc3'>[C@@H]</td>
          <td><span class="badge-S">S</span></td>
          <td>Bicyclic ring junction</td>
          <td>Constrains macrocyclic scaffold conformation</td>
        </tr>
        <tr>
          <td>C-5</td>
          <td style='color:#4dffc3'>[C@@H]</td>
          <td><span class="badge-S">S</span></td>
          <td>Second ring junction (bicyclic core)</td>
          <td>Maintains rigidity for protease binding</td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">R vs S Distribution</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["S configuration", "R configuration"],
            values=[4, 1],
            hole=0.55,
            marker=dict(colors=["#00d4ff", "#ff6464"], line=dict(color="#0a0e1a", width=3)),
            textfont=dict(family="Space Mono", size=12, color="white"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c0daea", family="Space Mono"),
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c0daea")),
            margin=dict(t=10, b=10, l=0, r=0), height=280,
            annotations=[dict(text="5<br>centers", x=0.5, y=0.5, font_size=16,
                              showarrow=False, font=dict(color="#00d4ff", family="Space Mono"))]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Stereocenter Position Diagram</div>', unsafe_allow_html=True)
        # Simple radial positions for the 5 centers
        angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
        labels = ["C-1 (S)", "C-2 (R)", "C-3 (S)", "C-4 (S)", "C-5 (S)"]
        colors = ["#00d4ff", "#ff6464", "#00d4ff", "#00d4ff", "#00d4ff"]
        x = [np.cos(a)*0.75 for a in angles]
        y = [np.sin(a)*0.75 for a in angles]

        fig_star = go.Figure()
        # Connect centers
        for i in range(5):
            for j in range(i+1, 5):
                fig_star.add_trace(go.Scatter(
                    x=[x[i], x[j]], y=[y[i], y[j]],
                    mode="lines", line=dict(color="#1a3a5c", width=1),
                    showlegend=False, hoverinfo="skip"
                ))
        fig_star.add_trace(go.Scatter(
            x=x, y=y, mode="markers+text",
            marker=dict(size=22, color=colors, line=dict(color="#0a0e1a", width=3)),
            text=labels, textposition="top center",
            textfont=dict(family="Space Mono", size=9, color="#c0daea"),
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False,
        ))
        # Center label
        fig_star.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers+text",
            marker=dict(size=30, color="#0f1e38", line=dict(color="#00d4ff", width=2)),
            text=["Core"], textposition="middle center",
            textfont=dict(family="Space Mono", size=9, color="#00d4ff"),
            showlegend=False, hoverinfo="skip"
        ))
        fig_star.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False, range=[-1.3,1.3]),
            yaxis=dict(visible=False, range=[-1.3,1.3], scaleanchor="x"),
            margin=dict(t=10, b=10, l=10, r=10), height=280,
        )
        st.plotly_chart(fig_star, use_container_width=True)

    st.markdown('<div class="section-header">Chirality in Context</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>
      <div style='background:#0f1e38;border:1px solid #1a3a5c;border-radius:12px;padding:18px'>
        <div style='color:#00d4ff;font-weight:600;margin-bottom:8px'>🔬 Biological Relevance</div>
        <div style='color:#a0c8e0;font-size:0.82rem;line-height:1.8'>
        The (S)-hydroxyl at C-3 mimics the tetrahedral intermediate of the scissile
        peptide bond in the HIV protease active site. The precise spatial arrangement
        of all 5 stereocenters is required for sub-nanomolar binding affinity (Kᵢ ≈ 0.015 nM).
        </div>
      </div>
      <div style='background:#0f1e38;border:1px solid #1a3a5c;border-radius:12px;padding:18px'>
        <div style='color:#00d4ff;font-weight:600;margin-bottom:8px'>⚗️ Synthetic Challenge</div>
        <div style='color:#a0c8e0;font-size:0.82rem;line-height:1.8'>
        With 5 stereocenters, 2⁵ = 32 possible stereoisomers exist. 
        Ritonavir's commercial synthesis relies on chiral pool materials and asymmetric
        reduction to set configurations with >99% ee. The total synthesis was a landmark
        achievement in medicinal chemistry.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PHARMACOLOGY
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "Pharmacology":
    st.markdown('<div class="section-header">Mechanism of Action</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0f1e38;border-left:4px solid #00d4ff;border-radius:0 12px 12px 0;padding:20px 24px;margin-bottom:20px'>
    <p style='color:#c0daea;font-size:0.88rem;line-height:1.9;margin:0'>
    Ritonavir is a <b style='color:#00d4ff'>competitive peptidomimetic inhibitor</b> of the HIV-1 and HIV-2 aspartyl proteases.
    It binds reversibly to the enzyme active site, with the hydroxyl group at the transition-state mimic
    position forming hydrogen bonds with the catalytic Asp25/Asp125 dyad. The binding occludes the substrate
    cleft, preventing cleavage of the Gag and Gag-Pol polyprotein precursors, resulting in production of
    structurally immature, non-infectious virions.
    </p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Pharmacokinetics</div>', unsafe_allow_html=True)
        pk_data = {
            "Parameter": ["Bioavailability", "Tmax", "Protein Binding", "Vd", "t½", "CYP Metabolism",
                           "Primary Metabolite", "Excretion"],
            "Value": ["~75% (with food)", "2–4 hours", "98–99%", "0.41 L/kg", "3–5 hours", "CYP3A4 (major), CYP2D6",
                      "M-2 (isopropylthiazole)", "Feces (~86%), Urine (~11%)"]
        }
        df_pk = pd.DataFrame(pk_data)
        st.dataframe(df_pk, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Drug Interactions</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#a0c8e0;font-size:0.82rem;line-height:1.8'>
        Ritonavir is a potent inhibitor of <b style='color:#ff8080'>CYP3A4</b> and also inhibits
        CYP2D6, P-glycoprotein, and several UGT enzymes. This pharmacokinetic boosting effect
        (<i>ritonavir boosting</i>) is exploited in HIV regimens to elevate AUC of co-PI drugs by 10–20×.
        Contraindicated with drugs dependent on CYP3A4 for clearance with narrow therapeutic indices
        (e.g., simvastatin, midazolam, ergotamine).
        </p>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header">Clinical Dosing</div>', unsafe_allow_html=True)
        dosing = {
            "Indication": ["HIV (monotherapy)", "PK Booster + Lopinavir", "PK Booster + Atazanavir",
                            "PK Booster + Darunavir", "COVID-19 (Paxlovid)"],
            "Dose": ["600 mg BID", "100 mg BID", "100 mg QD", "100 mg QD", "100 mg BID × 5 days"],
            "Notes": ["Historical; not preferred", "Kaletra formulation", "With food", "With food", "With nirmatrelvir"]
        }
        df_dos = pd.DataFrame(dosing)
        st.dataframe(df_dos, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Adverse Effects</div>', unsafe_allow_html=True)
        aes = ["GI disturbances (nausea, diarrhea)", "Lipid abnormalities (hypertriglyceridemia)",
               "Hepatotoxicity", "Oral/perioral paresthesia", "Taste perversion",
               "QTc prolongation (high doses)", "Drug-drug interactions (extensive)"]
        for ae in aes:
            st.markdown(f"<div style='background:#0f1e38;border-left:3px solid #ff6464;padding:7px 14px;margin:5px 0;border-radius:0 8px 8px 0;font-size:0.8rem;color:#c0daea'>⚠ {ae}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: GRAPHS
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "Graphs & Charts":
    st.markdown('<div class="section-header">Elemental Composition</div>', unsafe_allow_html=True)
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        elements = ["Carbon (C)", "Hydrogen (H)", "Nitrogen (N)", "Oxygen (O)", "Sulfur (S)"]
        counts   = [37, 48, 6, 5, 2]
        colors   = ["#00d4ff", "#c0daea", "#4dffc3", "#ff8080", "#ffdd00"]
        fig_elem = go.Figure(go.Bar(
            x=elements, y=counts,
            marker=dict(color=colors, line=dict(color="#0a0e1a", width=2)),
            text=counts, textposition="outside",
            textfont=dict(family="Space Mono", color="#c0daea"),
        ))
        fig_elem.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c0daea", family="Space Mono"),
            yaxis=dict(gridcolor="#1a3a5c", title="Atom Count"),
            xaxis=dict(showgrid=False),
            margin=dict(t=20, b=10, l=0, r=0), height=300,
            title=dict(text="Atom Count by Element", font=dict(color="#00d4ff", size=13)),
        )
        st.plotly_chart(fig_elem, use_container_width=True)

    with col_g2:
        mass_frac = [37*12.01, 48*1.008, 6*14.01, 5*16.00, 2*32.07]
        fig_donut = go.Figure(go.Pie(
            labels=elements, values=mass_frac, hole=0.5,
            marker=dict(colors=colors, line=dict(color="#0a0e1a", width=3)),
            textfont=dict(family="Space Mono", size=10),
        ))
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c0daea", family="Space Mono"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20, b=10, l=0, r=0), height=300,
            title=dict(text="Mass Contribution by Element", font=dict(color="#00d4ff", size=13)),
            annotations=[dict(text="720.94<br>g/mol", x=0.5, y=0.5,
                              font=dict(size=13, color="#00d4ff", family="Space Mono"), showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown('<div class="section-header">CYP3A4 Inhibition: IC₅₀ Comparison</div>', unsafe_allow_html=True)
    drugs = ["Ritonavir", "Ketoconazole", "Itraconazole", "Clarithromycin", "Erythromycin", "Fluoxetine"]
    ic50  = [0.066, 0.008, 0.27, 8.5, 93.1, 14.5]  # μM — approximate literature values
    bar_colors = ["#ff6464" if d == "Ritonavir" else "#00d4ff" for d in drugs]
    fig_cyp = go.Figure(go.Bar(
        x=drugs, y=ic50,
        marker=dict(color=bar_colors, line=dict(color="#0a0e1a", width=2)),
        text=[f"{v} μM" for v in ic50], textposition="outside",
        textfont=dict(family="Space Mono", color="#c0daea", size=10),
    ))
    fig_cyp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c0daea", family="Space Mono"),
        yaxis=dict(gridcolor="#1a3a5c", title="IC₅₀ (μM) — lower = stronger inhibition",
                   type="log"),
        xaxis=dict(showgrid=False),
        margin=dict(t=20, b=10, l=0, r=0), height=320,
    )
    st.plotly_chart(fig_cyp, use_container_width=True)
    st.caption("⚠ Ritonavir is among the most potent CYP3A4 inhibitors known. Log scale used. Source: literature Ki/IC₅₀ values.")

    st.markdown('<div class="section-header">Pharmacokinetic Boost Effect</div>', unsafe_allow_html=True)
    time_h = np.linspace(0, 24, 200)
    conc_alone    = 100 * np.exp(-0.35 * time_h) * (1 - np.exp(-1.2 * time_h))
    conc_boosted  = 800 * np.exp(-0.12 * time_h) * (1 - np.exp(-1.0 * time_h))
    fig_pk = go.Figure()
    fig_pk.add_trace(go.Scatter(x=time_h, y=conc_alone, mode="lines",
                                name="Atazanavir alone", line=dict(color="#7fbcd2", width=2, dash="dash")))
    fig_pk.add_trace(go.Scatter(x=time_h, y=conc_boosted, mode="lines",
                                name="Atazanavir + Ritonavir 100mg", line=dict(color="#ff6464", width=3)))
    fig_pk.add_hrect(y0=150, y1=350, fillcolor="rgba(0,212,255,0.05)",
                     line=dict(color="#00d4ff", width=1, dash="dot"),
                     annotation_text="Therapeutic window", annotation_position="top left",
                     annotation_font=dict(color="#00d4ff", size=10))
    fig_pk.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c0daea", family="Space Mono"),
        xaxis=dict(title="Time (hours)", gridcolor="#1a3a5c"),
        yaxis=dict(title="Plasma Conc. (ng/mL, simulated)", gridcolor="#1a3a5c"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a3a5c", borderwidth=1),
        margin=dict(t=20, b=10, l=0, r=0), height=340,
    )
    st.plotly_chart(fig_pk, use_container_width=True)
    st.caption("📈 Simulated pharmacokinetic curves illustrating ritonavir's boosting effect on atazanavir AUC (~8-10×). Values are illustrative.")

    st.markdown('<div class="section-header">Radar — Drug-likeness Profile</div>', unsafe_allow_html=True)
    categories = ["Lipophilicity\n(LogP≤5)", "MW\n(≤500)", "HBD\n(≤5)", "HBA\n(≤10)", "RotBonds\n(≤10)", "TPSA\n(≤140)"]
    # Normalize: 1=within rule, <1=over rule
    ritvals  = [min(1, 5/abs(-0.27 + 0.01)), min(1, 500/720.94), min(1, 5/4), min(1, 10/11), min(1, 10/14), min(1, 140/147.7)]
    ideals   = [1, 1, 1, 1, 1, 1]
    fig_rad = go.Figure()
    fig_rad.add_trace(go.Scatterpolar(r=ideals + [ideals[0]], theta=categories + [categories[0]],
                                      fill="toself", name="Ideal (Ro5)",
                                      line=dict(color="#4dffc3", width=1, dash="dot"),
                                      fillcolor="rgba(77,255,195,0.06)"))
    fig_rad.add_trace(go.Scatterpolar(r=ritvals + [ritvals[0]], theta=categories + [categories[0]],
                                      fill="toself", name="Ritonavir",
                                      line=dict(color="#00d4ff", width=2.5),
                                      fillcolor="rgba(0,212,255,0.10)"))
    fig_rad.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True, range=[0,1.05], gridcolor="#1a3a5c",
                                   tickfont=dict(color="#7fbcd2", size=9)),
                   angularaxis=dict(gridcolor="#1a3a5c", tickfont=dict(color="#c0daea", size=9))),
        font=dict(color="#c0daea", family="Space Mono"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, b=20, l=40, r=40), height=380,
    )
    st.plotly_chart(fig_rad, use_container_width=True)
    st.caption("Values capped at 1.0 = fully within Lipinski guideline; <1 = violation. Ritonavir is a 'beyond Ro5' molecule.")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;font-family:Space Mono;font-size:0.72rem;color:#3a5a72;padding:16px 0'>
  🧬 Ritonavir Molecular Explorer &nbsp;|&nbsp;
  Prepared by <b style='color:#7fbcd2'>Saksham Malviya</b> &nbsp;|&nbsp;
  Reg No: <b style='color:#7fbcd2'>RA2511026050017</b><br>
  Data sources: PubChem CID 392622 · ChEMBL163 · DrugBank DB00503 · FDA Label
</div>
""", unsafe_allow_html=True)
