"""
generate_ensemble_svg.py
Generates a premium light-mode SVG for the ensemble vote chart.
"""

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

svg = '''<svg viewBox="0 0 720 440" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif">
  <defs>
    <!-- Card shadow -->
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="120%">
      <feDropShadow dx="0" dy="4" stdDeviation="12" flood-color="#00000018"/>
    </filter>

    <!-- Bar gradients -->
    <linearGradient id="barBlue" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%"   stop-color="#6366f1"/>
      <stop offset="100%" stop-color="#4f46e5"/>
    </linearGradient>
    <linearGradient id="barAmber" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%"   stop-color="#f59e0b"/>
      <stop offset="100%" stop-color="#d97706"/>
    </linearGradient>
    <linearGradient id="barEnsemble" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%"   stop-color="#10b981"/>
      <stop offset="100%" stop-color="#059669"/>
    </linearGradient>
    <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%"   stop-color="#ffffff"/>
      <stop offset="100%" stop-color="#f8fafc"/>
    </linearGradient>
  </defs>

  <!-- Card background -->
  <rect width="720" height="440" rx="16" fill="url(#bgGrad)" filter="url(#shadow)"/>

  <!-- Top accent bar -->
  <rect width="720" height="4" rx="2" fill="#6366f1"/>

  <!-- Title block -->
  <text x="360" y="46" text-anchor="middle"
        font-size="18" font-weight="700" fill="#0f172a"
        letter-spacing="-0.3">Model Ensemble — RR vs RCB Pre-Match Prediction</text>
  <text x="360" y="68" text-anchor="middle"
        font-size="12.5" fill="#64748b">
    Each of 4 models votes independently · Final result = ensemble average · Prediction made before toss
  </text>

  <!-- Divider -->
  <line x1="60" y1="82" x2="660" y2="82" stroke="#e2e8f0" stroke-width="1"/>

  <!-- Y-axis grid lines and labels -->
  <!-- 65% -->
  <line x1="100" y1="110" x2="660" y2="110" stroke="#f1f5f9" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="90" y="114" text-anchor="end" font-size="11" fill="#94a3b8">65%</text>
  <!-- 60% -->
  <line x1="100" y1="155" x2="660" y2="155" stroke="#f1f5f9" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="90" y="159" text-anchor="end" font-size="11" fill="#94a3b8">60%</text>
  <!-- 55% -->
  <line x1="100" y1="200" x2="660" y2="200" stroke="#f1f5f9" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="90" y="204" text-anchor="end" font-size="11" fill="#94a3b8">55%</text>
  <!-- 50% — coin flip line -->
  <line x1="100" y1="245" x2="660" y2="245" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="6,3"/>
  <text x="90" y="249" text-anchor="end" font-size="11" fill="#ef4444">50%</text>
  <!-- 45% -->
  <line x1="100" y1="290" x2="660" y2="290" stroke="#f1f5f9" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="90" y="294" text-anchor="end" font-size="11" fill="#94a3b8">45%</text>

  <!-- Y axis line -->
  <line x1="100" y1="100" x2="100" y2="310" stroke="#e2e8f0" stroke-width="1.5"/>

  <!--
    Scale: 40% = y=335, 65% = y=110
    Range = 25% over 225px → 9px per 1%
    bar top y = 335 - (val - 40) * 9
    bar height = (val - 40) * 9
  -->

  <!-- Bar 1: Gradient Boost 60.1% -->
  <!-- top y = 335 - (60.1-40)*9 = 335 - 180.9 = 154.1 ≈ 154, h = 181 -->
  <rect x="148" y="154" width="76" height="181" rx="6" fill="url(#barBlue)"/>
  <!-- value label -->
  <text x="186" y="144" text-anchor="middle"
        font-size="17" font-weight="800" fill="#4f46e5">60.1%</text>
  <text x="186" y="160" text-anchor="middle" font-size="10" fill="#ffffff" opacity="0.0">.</text>
  <!-- model label -->
  <text x="186" y="328" text-anchor="middle" font-size="12" fill="#475569">Gradient</text>
  <text x="186" y="343" text-anchor="middle" font-size="12" fill="#475569">Boost</text>

  <!-- Bar 2: Logistic Regression 59.0% -->
  <!-- top y = 335 - (59-40)*9 = 335 - 171 = 164, h = 171 -->
  <rect x="268" y="164" width="76" height="171" rx="6" fill="url(#barBlue)"/>
  <text x="306" y="154" text-anchor="middle"
        font-size="17" font-weight="800" fill="#4f46e5">59.0%</text>
  <text x="306" y="328" text-anchor="middle" font-size="12" fill="#475569">Logistic</text>
  <text x="306" y="343" text-anchor="middle" font-size="12" fill="#475569">Regression</text>

  <!-- Bar 3: Random Forest 49.7% — amber, coin flip level -->
  <!-- top y = 335 - (49.7-40)*9 = 335 - 87.3 = 247.7 ≈ 248, h = 87 -->
  <rect x="388" y="248" width="76" height="87" rx="6" fill="url(#barAmber)"/>
  <text x="426" y="238" text-anchor="middle"
        font-size="17" font-weight="800" fill="#d97706">49.7%</text>
  <text x="426" y="328" text-anchor="middle" font-size="12" fill="#475569">Random</text>
  <text x="426" y="343" text-anchor="middle" font-size="12" fill="#475569">Forest</text>

  <!-- Bar 4: XGBoost 55.2% -->
  <!-- top y = 335 - (55.2-40)*9 = 335 - 136.8 = 198.2 ≈ 198, h = 137 -->
  <rect x="508" y="198" width="76" height="137" rx="6" fill="url(#barBlue)"/>
  <text x="546" y="188" text-anchor="middle"
        font-size="17" font-weight="800" fill="#4f46e5">55.2%</text>
  <text x="546" y="328" text-anchor="middle" font-size="12" fill="#475569">XGBoost</text>

  <!-- X axis baseline -->
  <line x1="100" y1="335" x2="660" y2="335" stroke="#cbd5e1" stroke-width="1.5"/>

  <!-- 50% coin flip label -->
  <rect x="436" y="233" width="158" height="20" rx="4" fill="#fef2f2"/>
  <text x="515" y="247" text-anchor="middle" font-size="10.5" fill="#ef4444" font-weight="600">← 50% coin flip line</text>

  <!-- RF annotation callout -->
  <rect x="355" y="98" width="190" height="42" rx="6"
        fill="#fffbeb" stroke="#fcd34d" stroke-width="1.2"/>
  <text x="450" y="116" text-anchor="middle" font-size="10.5" fill="#92400e" font-weight="600">⚠ RF at 49.7% — near coin flip</text>
  <text x="450" y="132" text-anchor="middle" font-size="10" fill="#b45309">Signals genuine match uncertainty</text>
  <!-- callout pointer -->
  <line x1="426" y1="140" x2="426" y2="236" stroke="#fcd34d" stroke-width="1" stroke-dasharray="3,2"/>

  <!-- Ensemble result pill at bottom -->
  <rect x="180" y="362" width="360" height="44" rx="22"
        fill="#f0fdf4" stroke="#86efac" stroke-width="1.5"/>
  <text x="360" y="380" text-anchor="middle"
        font-size="11" fill="#166534">ENSEMBLE AVERAGE</text>
  <text x="360" y="399" text-anchor="middle"
        font-size="14" font-weight="800" fill="#15803d">60% RR Win Probability — MODERATE confidence</text>

  <!-- Source note -->
  <text x="360" y="430" text-anchor="middle"
        font-size="10" fill="#94a3b8">
    Source: Live prediction from IPL Prediction Arena · April 10, 2026 · Pre-match
  </text>
</svg>'''

out = OUT_DIR / "blog_diagram1_ensemble.svg"
out.write_text(svg, encoding="utf-8")
print(f"Saved: {out}")

# Also save as high-res PNG using cairosvg if available, else matplotlib fallback
try:
    import cairosvg
    png_out = OUT_DIR / "blog_diagram1_ensemble_hires.png"
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=str(png_out),
        scale=3.0
    )
    print(f"Saved high-res PNG: {png_out}")
except ImportError:
    # Fallback: save SVG only, user can export via browser
    print("cairosvg not available — SVG saved. Open in browser and export as PNG for Medium.")
