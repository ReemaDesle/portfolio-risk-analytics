import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, LineChart, Line, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, ReferenceLine
} from 'recharts';
import {
  ShieldCheck, TrendingUp, TrendingDown, AlertTriangle, Activity,
  Zap, RefreshCw, ArrowUpRight, ArrowDownRight, Minus, Globe,
  DollarSign, Cpu, Eye, Target
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

// ── Colour tokens ──────────────────────────────────────────────────────────
const C = {
  bg:       '#060b17',
  surface:  '#0d1526',
  card:     '#111d35',
  border:   '#1e3058',
  accent:   '#38bdf8',
  success:  '#22c55e',
  warning:  '#f59e0b',
  danger:   '#ef4444',
  muted:    '#64748b',
  text:     '#e2e8f0',
};

// ── Reusable primitives ────────────────────────────────────────────────────
const Card = ({ children, style = {}, glow }) => (
  <div style={{
    background: C.card,
    border: `1px solid ${glow ? glow : C.border}`,
    borderRadius: 16,
    padding: '1.5rem',
    boxShadow: glow ? `0 0 24px ${glow}22` : 'none',
    ...style,
  }}>
    {children}
  </div>
);

const Label = ({ children, style = {} }) => (
  <div style={{ color: C.muted, fontSize: '0.72rem', fontWeight: 700,
    letterSpacing: '0.12em', textTransform: 'uppercase', ...style }}>
    {children}
  </div>
);

const BigNum = ({ children, color = C.accent, size = '2rem' }) => (
  <div style={{ color, fontSize: size, fontWeight: 800, lineHeight: 1.1, marginTop: 6 }}>
    {children}
  </div>
);

const Badge = ({ label, color = C.accent }) => (
  <span style={{
    background: `${color}22`, color, border: `1px solid ${color}44`,
    borderRadius: 999, padding: '2px 10px', fontSize: '0.71rem', fontWeight: 700,
    letterSpacing: '0.06em',
  }}>
    {label}
  </span>
);

const SignalDot = ({ color }) => (
  <span style={{
    display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
    background: color, marginRight: 6, boxShadow: `0 0 6px ${color}`,
  }} />
);

const colorOf = (key) => ({
  danger: C.danger, warning: C.warning, success: C.success, accent: C.accent
}[key] ?? C.accent);

// ─────────────────────────────────────────────────────────────────────────
//  MAIN APP
// ─────────────────────────────────────────────────────────────────────────
export default function App() {
  const [portfolios, setPortfolios]         = useState([]);
  const [selected, setSelected]             = useState('');
  const [activeTab, setActiveTab]           = useState('overview');
  const [inference, setInference]           = useState(null);
  const [analytics, setAnalytics]           = useState(null);
  const [modelStatus, setModelStatus]       = useState(null);
  const [loading, setLoading]               = useState(false);
  const [lastUpdated, setLastUpdated]       = useState(null);

  // Fetch portfolio list once
  useEffect(() => {
    axios.get(`${API_BASE}/portfolios`)
      .then(r => {
        setPortfolios(r.data.portfolios);
        if (r.data.portfolios.length) setSelected(r.data.portfolios[0]);
      }).catch(console.error);

    axios.get(`${API_BASE}/model-status`)
      .then(r => setModelStatus(r.data)).catch(console.error);
  }, []);

  // Fetch on portfolio change
  const fetchAll = useCallback(async () => {
    if (!selected) return;
    setLoading(true);
    try {
      const [infRes, anaRes] = await Promise.all([
        axios.get(`${API_BASE}/suggestions/${selected}`),
        axios.get(`${API_BASE}/analytics/${selected}`),
      ]);
      setInference(infRes.data);
      setAnalytics(anaRes.data);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  }, [selected]);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  // ── Derived values ──────────────────────────────────────────────────────
  const shock     = inference?.shock         ?? {};
  const recovery  = inference?.recovery      ?? {};
  const risk      = inference?.risk_score    ?? {};
  const domSens   = inference?.domain_sensitivity ?? {};
  const portCat   = inference?.portfolio_category ?? {};
  const buysell   = inference?.buysell       ?? {};
  const stocks    = inference?.stock_table   ?? [];
  const chart     = inference?.market_chart  ?? analytics?.market_data ?? [];

  const shockProb   = shock.probability ?? 0;
  const shockSignal = shock.signal ?? 'NORMAL';
  const actionColor = colorOf(buysell.color ?? 'accent');
  const riskColor   = colorOf(risk.risk_color ?? 'success');

  // Domain sensitivity radar data
  const radarData = domSens.ranked_domains?.map(d => ({
    domain: d.domain.slice(0, 3).toUpperCase(),
    value:  Math.abs(d.coefficient) * 1000,
    full:   d.domain,
  })) ?? [];

  // Domain sentiment bars
  const sentBars = domSens.recent_sentiment
    ? Object.entries(domSens.recent_sentiment).map(([d, v]) => ({
        name: d.charAt(0).toUpperCase() + d.slice(1, 4) + '.',
        value: v,
        fill: v >= 0 ? C.success : C.danger,
      }))
    : [];

  // Tabs
  const tabs = [
    { id: 'overview',     label: 'Overview' },
    { id: 'sensitivity',  label: 'Domain Sensitivity' },
    { id: 'portfolio',    label: 'Portfolio Profile' },
    { id: 'advisory',     label: 'AI Advisory' },
    { id: 'stocks',       label: 'Stock Table' },
  ];

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: C.bg, color: C.text,
      fontFamily: "'Inter', sans-serif" }}>

      {/* ── Sidebar ── */}
      <aside style={{
        width: 220, background: C.surface, borderRight: `1px solid ${C.border}`,
        padding: '2rem 1.25rem', display: 'flex', flexDirection: 'column', gap: '2rem',
        position: 'sticky', top: 0, height: '100vh', overflowY: 'auto',
      }}>
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <ShieldCheck size={24} color={C.accent} />
          <span style={{ fontWeight: 800, fontSize: '1rem', letterSpacing: '0.08em',
            color: C.accent }}>NEURAL RISK</span>
        </div>

        {/* Portfolio picker */}
        <div>
          <Label style={{ marginBottom: 8 }}>Active Portfolio</Label>
          <select
            value={selected}
            onChange={e => setSelected(e.target.value)}
            style={{
              width: '100%', background: C.card, border: `1px solid ${C.border}`,
              color: C.text, borderRadius: 8, padding: '0.5rem 0.75rem',
              fontSize: '0.85rem', outline: 'none',
            }}
          >
            {portfolios.map(p => <option key={p} value={p}>{p.toUpperCase()}</option>)}
          </select>
        </div>

        {/* Nav */}
        <nav style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              style={{
                background: activeTab === t.id ? `${C.accent}18` : 'transparent',
                border: `1px solid ${activeTab === t.id ? C.accent : 'transparent'}`,
                color: activeTab === t.id ? C.accent : C.muted,
                borderRadius: 8, padding: '0.55rem 0.85rem',
                textAlign: 'left', fontSize: '0.82rem', fontWeight: 600,
                cursor: 'pointer', transition: 'all 0.15s',
              }}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {/* System status */}
        <div style={{ marginTop: 'auto', fontSize: '0.73rem', color: C.muted }}>
          <div style={{ marginBottom: 4 }}>
            <SignalDot color={modelStatus?.all_ready ? C.success : C.warning} />
            {modelStatus?.all_ready ? 'All 6 models ready' : 'Some models missing'}
          </div>
          {lastUpdated && <div>Updated: {lastUpdated}</div>}
        </div>
      </aside>

      {/* ── Main ── */}
      <main style={{ flex: 1, padding: '2rem', overflowY: 'auto', maxWidth: 1200 }}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between',
          alignItems: 'flex-start', marginBottom: '2rem' }}>
          <div>
            <h1 style={{ fontSize: '1.75rem', fontWeight: 800, textTransform: 'capitalize',
              marginBottom: 4 }}>
              {selected} Portfolio
            </h1>
            <p style={{ color: C.muted, fontSize: '0.85rem' }}>
              Multi-domain sentiment risk analytics · 1,329 trading days · 6 ML models
            </p>
          </div>
          <button
            onClick={fetchAll}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              background: 'transparent', border: `1px solid ${C.border}`,
              color: C.text, borderRadius: 8, padding: '0.5rem 1rem',
              cursor: 'pointer', fontSize: '0.82rem',
            }}
          >
            <RefreshCw size={14} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
            Refresh
          </button>
        </div>

        {loading && (
          <div style={{ textAlign: 'center', padding: '4rem', color: C.muted }}>
            Running ML inference pipeline…
          </div>
        )}

        {!loading && inference && (
          <>
            {/* ══ OVERVIEW TAB ══════════════════════════════════════════ */}
            {activeTab === 'overview' && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: '1rem' }}>

                {/* KPI row */}
                {[
                  {
                    label: 'Shock Probability (M1)',
                    value: `${(shockProb * 100).toFixed(1)}%`,
                    sub: shockSignal,
                    color: colorOf(shock.signal === 'HIGH' ? 'danger'
                           : shock.signal === 'ELEVATED' ? 'warning' : 'success'),
                    icon: <AlertTriangle size={18} />,
                    col: 4,
                  },
                  {
                    label: 'Next-Day Vol Risk (M3)',
                    value: risk.risk_label ?? '—',
                    sub: risk.predicted_vol ? `Vol: ${(risk.predicted_vol * 100).toFixed(3)}%` : '',
                    color: riskColor,
                    icon: <Activity size={18} />,
                    col: 4,
                  },
                  {
                    label: 'Recovery Forecast (M2)',
                    value: recovery.p50_days ? `${recovery.p50_days}d` : '—',
                    sub: recovery.band_label ?? '',
                    color: C.accent,
                    icon: <TrendingUp size={18} />,
                    col: 4,
                  },
                ].map(({ label, value, sub, color, icon, col }) => (
                  <Card key={label} style={{ gridColumn: `span ${col}` }} glow={color}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <Label>{label}</Label>
                      <span style={{ color }}>{icon}</span>
                    </div>
                    <BigNum color={color}>{value}</BigNum>
                    <div style={{ color: C.muted, fontSize: '0.75rem', marginTop: 6 }}>{sub}</div>
                  </Card>
                ))}

                {/* Buy/Sell signal */}
                <Card style={{ gridColumn: 'span 4' }} glow={actionColor}>
                  <Label>Recommended Action</Label>
                  <BigNum color={actionColor} size="1.6rem">{buysell.action ?? '—'}</BigNum>
                  <div style={{ color: C.muted, fontSize: '0.73rem', marginTop: 8 }}>
                    Confidence: {buysell.confidence ?? '—'}
                  </div>
                </Card>

                {/* Portfolio category */}
                <Card style={{ gridColumn: 'span 4' }}>
                  <Label>Portfolio Category (M6)</Label>
                  <BigNum color={C.accent} size="1.1rem" style={{ marginTop: 8 }}>
                    {portCat.label ?? '—'}
                  </BigNum>
                  <div style={{ color: C.muted, fontSize: '0.73rem', marginTop: 8 }}>
                    Shock freq: {portCat.shock_frequency_pct ?? '—'}% of trading days
                  </div>
                </Card>

                {/* News to watch */}
                <Card style={{ gridColumn: 'span 4' }}>
                  <Label>Primary News Signal (M4/M5)</Label>
                  <BigNum color={C.warning} size="1.3rem">
                    {domSens.news_to_watch ?? '—'}
                  </BigNum>
                  <div style={{ color: C.muted, fontSize: '0.73rem', marginTop: 8 }}>
                    {domSens.granger_confirmed
                      ? '✓ Granger-confirmed causal signal'
                      : 'Dominant sensitivity domain'}
                  </div>
                </Card>

                {/* Area chart: SPY / first ticker */}
                <Card style={{ gridColumn: 'span 8' }}>
                  <Label style={{ marginBottom: '1rem' }}>Price Trend (60d)</Label>
                  <ResponsiveContainer width="100%" height={240}>
                    <AreaChart data={chart}>
                      <defs>
                        <linearGradient id="gA" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%"  stopColor={C.accent}  stopOpacity={0.3} />
                          <stop offset="95%" stopColor={C.accent}  stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                      <XAxis dataKey="date" hide />
                      <YAxis stroke={C.muted} fontSize={10} width={50}
                        tickFormatter={v => `$${parseFloat(v).toFixed(0)}`} />
                      <Tooltip
                        contentStyle={{ background: C.surface, border: `1px solid ${C.border}`,
                          borderRadius: 8, fontSize: '0.8rem' }}
                      />
                      {/* Show first 2 tickers that exist in chart data */}
                      {Object.keys(chart[0] ?? {})
                        .filter(k => !['date'].includes(k) && !k.startsWith('sent_'))
                        .slice(0, 2)
                        .map((t, i) => (
                          <Area key={t} type="monotone" dataKey={t}
                            stroke={i === 0 ? C.accent : C.success}
                            fill={i === 0 ? 'url(#gA)' : 'none'}
                            fillOpacity={1} strokeWidth={2} dot={false} />
                        ))
                      }
                    </AreaChart>
                  </ResponsiveContainer>
                </Card>

                {/* Sentiment trend */}
                <Card style={{ gridColumn: 'span 4' }}>
                  <Label style={{ marginBottom: '1rem' }}>7-Day Sentiment (3 Domains)</Label>
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={sentBars} layout="vertical">
                      <XAxis type="number" hide domain={[-0.3, 0.3]} />
                      <YAxis dataKey="name" type="category" stroke={C.muted}
                        fontSize={10} width={36} />
                      <Tooltip
                        contentStyle={{ background: C.surface, border: `1px solid ${C.border}`,
                          borderRadius: 8, fontSize: '0.8rem' }}
                        formatter={v => v.toFixed(4)}
                      />
                      <ReferenceLine x={0} stroke={C.border} />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {sentBars.map((entry, i) => (
                          <rect key={i} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </div>
            )}

            {/* ══ DOMAIN SENSITIVITY TAB ═══════════════════════════════ */}
            {activeTab === 'sensitivity' && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: '1rem' }}>
                <Card style={{ gridColumn: 'span 7' }}>
                  <Label style={{ marginBottom: '1.5rem' }}>M5 Sensitivity Coefficients</Label>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={domSens.ranked_domains?.map(d => ({
                      domain: d.domain,
                      coeff:  parseFloat((d.coefficient * 1000).toFixed(4)),
                    })) ?? []}>
                      <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                      <XAxis dataKey="domain" stroke={C.muted} fontSize={11} />
                      <YAxis stroke={C.muted} fontSize={10}
                        tickFormatter={v => `${v}e-3`} />
                      <Tooltip
                        contentStyle={{ background: C.surface, border: `1px solid ${C.border}`,
                          borderRadius: 8, fontSize: '0.8rem' }}
                        formatter={v => [`${v}×10⁻³`, 'Coefficient']}
                      />
                      <Bar dataKey="coeff" radius={[4, 4, 0, 0]}>
                        {(domSens.ranked_domains ?? []).map((d, i) => (
                          <rect key={i}
                            fill={i === 0 ? C.accent : i === 1 ? C.warning : C.muted} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Card>

                <Card style={{ gridColumn: 'span 5' }}>
                  <Label style={{ marginBottom: '1rem' }}>Domain Interpretation</Label>
                  {domSens.ranked_domains?.map((d, i) => (
                    <div key={d.domain} style={{
                      padding: '0.75rem', marginBottom: '0.6rem',
                      background: C.surface, borderRadius: 8,
                      borderLeft: `3px solid ${i === 0 ? C.accent : i === 1 ? C.warning : C.muted}`,
                    }}>
                      <div style={{ fontWeight: 700, fontSize: '0.85rem',
                        textTransform: 'capitalize' }}>{d.domain}</div>
                      <div style={{ color: C.muted, fontSize: '0.73rem', marginTop: 3 }}>
                        Coeff: {d.coefficient.toExponential(3)} ·
                        7-day avg: {(domSens.recent_sentiment?.[d.domain] ?? 0).toFixed(4)}
                      </div>
                    </div>
                  ))}
                  {domSens.granger_confirmed && (
                    <div style={{ marginTop: '1rem', background: `${C.success}15`,
                      border: `1px solid ${C.success}44`, borderRadius: 8,
                      padding: '0.75rem', fontSize: '0.75rem', color: C.success }}>
                      ✓ {domSens.granger_confirmed}
                    </div>
                  )}
                </Card>

                <Card style={{ gridColumn: 'span 12' }}>
                  <Label style={{ marginBottom: '1rem' }}>Sentiment History (60d)</Label>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={chart}>
                      <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                      <XAxis dataKey="date" hide />
                      <YAxis stroke={C.muted} fontSize={10} />
                      <Tooltip
                        contentStyle={{ background: C.surface, border: `1px solid ${C.border}`,
                          borderRadius: 8, fontSize: '0.8rem' }}
                      />
                      <Line dataKey="sent_geopolitical" stroke={C.danger}  dot={false} strokeWidth={2} name="Geo" />
                      <Line dataKey="sent_financial"    stroke={C.accent}  dot={false} strokeWidth={2} name="Fin" />
                      <Line dataKey="sent_technology"   stroke={C.success} dot={false} strokeWidth={2} name="Tech" />
                    </LineChart>
                  </ResponsiveContainer>
                </Card>
              </div>
            )}

            {/* ══ PORTFOLIO PROFILE TAB ════════════════════════════════ */}
            {activeTab === 'portfolio' && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: '1rem' }}>
                {[
                  { label: 'Category (M6)', value: portCat.label ?? '—', color: C.accent, col: 4 },
                  { label: 'Shock Frequency', value: `${portCat.shock_frequency_pct ?? '—'}%`,
                    color: C.warning, col: 3, sub: 'of trading days' },
                  { label: 'Holdings Correlation', value: portCat.intra_correlation ?? '—',
                    color: C.accent, col: 3, sub: '1 = move together' },
                  { label: 'Safe-Haven Weight', value: `${portCat.safe_haven_pct ?? '—'}%`,
                    color: C.success, col: 2, sub: 'GLD/TLT/VPU' },
                ].map(({ label, value, color, col, sub }) => (
                  <Card key={label} style={{ gridColumn: `span ${col}` }}>
                    <Label>{label}</Label>
                    <BigNum color={color} size="1.8rem">{value}</BigNum>
                    {sub && <div style={{ color: C.muted, fontSize: '0.73rem', marginTop: 6 }}>{sub}</div>}
                  </Card>
                ))}

                {/* Expansion suggestions */}
                <Card style={{ gridColumn: 'span 6' }}>
                  <Label style={{ marginBottom: '1rem' }}>Portfolio Expansion Suggestions</Label>
                  {portCat.expansion?.length ? portCat.expansion.map(s => (
                    <div key={s.ticker} style={{
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                      padding: '0.75rem', marginBottom: '0.5rem',
                      background: C.surface, borderRadius: 8,
                    }}>
                      <div>
                        <span style={{ fontWeight: 700, marginRight: 10,
                          color: C.accent }}>{s.ticker}</span>
                        <span style={{ color: C.muted, fontSize: '0.78rem' }}>{s.rationale}</span>
                      </div>
                      <Badge label="Add +" color={C.success} />
                    </div>
                  )) : (
                    <div style={{ color: C.muted, fontSize: '0.82rem' }}>
                      Portfolio is well-diversified. No additions recommended.
                    </div>
                  )}
                </Card>

                {/* Risk profile summary */}
                <Card style={{ gridColumn: 'span 6' }}>
                  <Label style={{ marginBottom: '1rem' }}>M3 Risk Band (Next Day)</Label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
                    <div style={{
                      width: 20, height: 120, background: `linear-gradient(to bottom, ${C.danger}, ${C.warning}, ${C.success})`,
                      borderRadius: 999, position: 'relative', flexShrink: 0,
                    }}>
                      {risk.predicted_vol != null && (
                        <div style={{
                          position: 'absolute', left: '100%', marginLeft: 8,
                          top: `${Math.min(90, Math.max(5, (risk.predicted_vol / 0.04) * 90))}%`,
                          width: 8, height: 2, background: 'white',
                          whiteSpace: 'nowrap', fontSize: '0.72rem', color: C.text,
                          paddingLeft: 4,
                        }}>
                          ← {(risk.predicted_vol * 100).toFixed(3)}%
                        </div>
                      )}
                    </div>
                    <div>
                      <div style={{ fontWeight: 800, fontSize: '1.2rem', color: riskColor }}>
                        {risk.risk_label ?? '—'}
                      </div>
                      <div style={{ color: C.muted, fontSize: '0.75rem', marginTop: 6 }}>
                        P25: {risk.p25_vol ? (risk.p25_vol * 100).toFixed(3) + '%' : '—'}<br />
                        P75: {risk.p75_vol ? (risk.p75_vol * 100).toFixed(3) + '%' : '—'}
                      </div>
                    </div>
                  </div>
                </Card>
              </div>
            )}

            {/* ══ AI ADVISORY TAB ══════════════════════════════════════ */}
            {activeTab === 'advisory' && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: '1rem' }}>

                {/* Main advisory card */}
                <Card style={{ gridColumn: 'span 12', borderLeft: `4px solid ${actionColor}` }}
                  glow={actionColor}>
                  <div style={{ display: 'flex', justifyContent: 'space-between',
                    alignItems: 'flex-start' }}>
                    <div>
                      <Label>ML-Derived Recommendation</Label>
                      <h2 style={{ fontSize: '2rem', fontWeight: 800, color: actionColor, marginTop: 8 }}>
                        {buysell.action ?? '—'}
                      </h2>
                      <Badge label={`Confidence: ${buysell.confidence ?? '—'}`} color={actionColor} />
                    </div>
                    <Zap size={40} color={actionColor} />
                  </div>
                  <div style={{ marginTop: '1.5rem', lineHeight: 1.7,
                    color: C.muted, fontSize: '0.9rem', maxWidth: 740 }}>
                    {buysell.reasoning ?? '—'}
                  </div>
                </Card>

                {/* Evidence cards */}
                {[
                  {
                    label: 'M1 Evidence',
                    body: shock.note ?? '—',
                    icon: <AlertTriangle size={16} />,
                    color: colorOf(shock.signal === 'HIGH' ? 'danger'
                           : shock.signal === 'ELEVATED' ? 'warning' : 'success'),
                    col: 4,
                  },
                  {
                    label: 'M2 Evidence',
                    body: recovery.band_label ?? '—',
                    icon: <TrendingUp size={16} />,
                    color: C.accent,
                    col: 4,
                  },
                  {
                    label: 'M4/M5 Evidence',
                    body: domSens.granger_confirmed
                      ?? `Dominant domain: ${domSens.dominant_domain ?? '—'}`,
                    icon: <Globe size={16} />,
                    color: C.warning,
                    col: 4,
                  },
                ].map(({ label, body, icon, color, col }) => (
                  <Card key={label} style={{ gridColumn: `span ${col}` }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
                      <span style={{ color }}>{icon}</span>
                      <Label>{label}</Label>
                    </div>
                    <div style={{ color: C.muted, fontSize: '0.82rem', lineHeight: 1.6 }}>
                      {body}
                    </div>
                  </Card>
                ))}

                {/* Disclaimer */}
                <div style={{ gridColumn: 'span 12', color: C.muted,
                  fontSize: '0.72rem', textAlign: 'center', marginTop: '0.5rem' }}>
                  ⚠ These are ML model inferences from historical data (2021–2026), not financial advice.
                  Always do your own research before making investment decisions.
                </div>
              </div>
            )}

            {/* ══ STOCK TABLE TAB ══════════════════════════════════════ */}
            {activeTab === 'stocks' && (
              <Card>
                <Label style={{ marginBottom: '1.25rem' }}>
                  Per-Stock Performance — {selected.toUpperCase()} Portfolio
                </Label>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                  <thead>
                    <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                      {['Ticker', 'Last Price', '7-Day Return', '5-Day Vol', 'Shock Today'].map(h => (
                        <th key={h} style={{ padding: '0.6rem 1rem', textAlign: 'left',
                          color: C.muted, fontWeight: 600, fontSize: '0.72rem',
                          letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {stocks.length ? stocks.map(s => (
                      <tr key={s.ticker} style={{
                        borderBottom: `1px solid ${C.border}`,
                        background: s.shock_today ? `${C.danger}08` : 'transparent',
                      }}>
                        <td style={{ padding: '0.8rem 1rem', fontWeight: 700, color: C.accent }}>
                          {s.ticker}
                        </td>
                        <td style={{ padding: '0.8rem 1rem' }}>${s.last_price}</td>
                        <td style={{ padding: '0.8rem 1rem',
                          color: s.return_7d >= 0 ? C.success : C.danger }}>
                          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                            {s.return_7d >= 0
                              ? <ArrowUpRight size={13} />
                              : <ArrowDownRight size={13} />}
                            {s.return_7d}%
                          </span>
                        </td>
                        <td style={{ padding: '0.8rem 1rem', color: C.muted }}>
                          {(s.vol5 * 100).toFixed(3)}%
                        </td>
                        <td style={{ padding: '0.8rem 1rem' }}>
                          {s.shock_today
                            ? <Badge label="⚡ SHOCK" color={C.danger} />
                            : <Badge label="Normal" color={C.success} />}
                        </td>
                      </tr>
                    )) : (
                      <tr>
                        <td colSpan={5} style={{ padding: '2rem', textAlign: 'center',
                          color: C.muted }}>
                          No stock data available
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </Card>
            )}
          </>
        )}

        {!loading && !inference && (
          <div style={{ textAlign: 'center', padding: '6rem', color: C.muted }}>
            <AlertTriangle size={40} style={{ marginBottom: '1rem' }} />
            <div>Could not load data. Is the backend running?</div>
            <code style={{ fontSize: '0.8rem', display: 'block', marginTop: 8 }}>
              python dashboard/main.py
            </code>
          </div>
        )}
      </main>

      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: ${C.bg}; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        select option { background: ${C.surface}; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: ${C.bg}; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 3px; }
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
      `}</style>
    </div>
  );
}
