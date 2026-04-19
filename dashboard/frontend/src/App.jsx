import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, LineChart, Line, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, ReferenceLine, Cell
} from 'recharts';
import {
  ShieldCheck, TrendingUp, TrendingDown, AlertTriangle, Activity,
  Zap, RefreshCw, ArrowUpRight, ArrowDownRight, Minus, Globe,
  DollarSign, Cpu, Eye, Target, Calendar, ListFilter, LayoutDashboard,
  Settings, Info, ChevronRight, CheckCircle2, FlaskConical, BarChart3, Database,
  BrainCircuit, FileBarChart, Microscope, Binary, Network
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
const Card = ({ children, style = {}, glow, onClick }) => (
  <div 
    onClick={onClick}
    style={{
      background: C.card,
      border: `1px solid ${glow ? glow : C.border}`,
      borderRadius: 16,
      padding: '1.25rem',
      boxShadow: glow ? `0 0 24px ${glow}22` : 'rgba(0,0,0,0.2) 0 4px 12px',
      cursor: onClick ? 'pointer' : 'default',
      transition: 'transform 0.2s ease, box-shadow 0.2s ease',
      ...style,
    }}
    onMouseEnter={e => onClick && (e.currentTarget.style.transform = 'translateY(-2px)')}
    onMouseLeave={e => onClick && (e.currentTarget.style.transform = 'translateY(0)')}
  >
    {children}
  </div>
);

const Label = ({ children, style = {} }) => (
  <div style={{ color: C.muted, fontSize: '0.68rem', fontWeight: 700,
    letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 4, ...style }}>
    {children}
  </div>
);

const BigNum = ({ children, color = C.accent, size = '1.8rem' }) => (
  <div style={{ color, fontSize: size, fontWeight: 800, lineHeight: 1.1 }}>
    {children}
  </div>
);

const Badge = ({ label, color = C.accent, style = {} }) => (
  <span style={{
    background: `${color}22`, color, border: `1px solid ${color}44`,
    borderRadius: 8, padding: '2px 8px', fontSize: '0.68rem', fontWeight: 700,
    ...style
  }}>
    {label}
  </span>
);

const SectionHeading = ({ icon: Icon, title, sub }) => (
  <div style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: 12 }}>
    <div style={{ background: `${C.accent}15`, padding: 10, borderRadius: 12, border: `1px solid ${C.accent}33` }}>
      <Icon size={20} color={C.accent} />
    </div>
    <div>
      <h2 style={{ fontSize: '1.1rem', fontWeight: 800, margin: 0 }}>{title}</h2>
      {sub && <p style={{ color: C.muted, fontSize: '0.75rem', margin: 0 }}>{sub}</p>}
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────────────────
//  MAIN APP
// ─────────────────────────────────────────────────────────────────────────
export default function App() {
  const [view, setView]                     = useState('user'); 
  const [groupedTickers, setGroupedTickers] = useState({});
  const [leadTickers, setLeadTickers]       = useState({});
  const [validDates, setValidDates]         = useState([]);
  
  // Selection state (Initialize from localStorage)
  const [selectedTickers, setSelectedTickers] = useState(() => {
    const saved = localStorage.getItem('selectedTickers');
    return saved ? JSON.parse(saved) : [];
  });
  const [quantities, setQuantities] = useState(() => {
    const saved = localStorage.getItem('quantities');
    return saved ? JSON.parse(saved) : {};
  });
  const [targetDate, setTargetDate]           = useState('');
  
  // Analysis Data
  const [analysis, setAnalysis]             = useState(null);
  const [metrics, setMetrics]               = useState(null);
  const [arima, setArima]                   = useState(null);
  
  // UI State
  const [loading, setLoading]               = useState(false);
  const [lastUpdated, setLastUpdated]       = useState(null);
  const [showDetailedXAI, setShowDetailedXAI] = useState(false);

  // Persistence Effects
  useEffect(() => { localStorage.setItem('selectedTickers', JSON.stringify(selectedTickers)); }, [selectedTickers]);
  useEffect(() => { localStorage.setItem('quantities', JSON.stringify(quantities)); }, [quantities]);

  // Fetch initial data
  useEffect(() => {
    setLoading(true);
    axios.get(`${API_BASE}/api/tickers`)
      .then(r => {
        setGroupedTickers(r.data.grouped);
        setLeadTickers(r.data.lead_tickers);
        setValidDates(r.data.valid_dates);
        if (r.data.valid_dates.length) setTargetDate(r.data.valid_dates[r.data.valid_dates.length - 1]);
      });
    fetchMetrics();
    setLoading(false);
  }, []);

  const fetchMetrics = () => {
    axios.get(`${API_BASE}/api/metrics`).then(r => setMetrics(r.data)).catch(console.error);
  };

  const handleToggleTicker = (ticker) => {
    if (selectedTickers.includes(ticker)) {
      setSelectedTickers(selectedTickers.filter(t => t !== ticker));
      const newQtys = { ...quantities };
      delete newQtys[ticker];
      setQuantities(newQtys);
    } else {
      setSelectedTickers([...selectedTickers, ticker]);
      setQuantities({ ...quantities, [ticker]: 10 });
    }
  };

  const handleRunAnalysis = async () => {
    if (selectedTickers.length === 0) return;
    setLoading(true);
    try {
      const resp = await axios.post(`${API_BASE}/api/analyze`, {
        tickers: selectedTickers,
        quantities: selectedTickers.map(t => quantities[t] || 0),
        date: targetDate
      });
      setAnalysis(resp.data);
      const arimaResp = await axios.get(`${API_BASE}/api/arima/${resp.data.mapped_archetype}`);
      setArima(arimaResp.data);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) { console.error(err); }
    setLoading(false);
  };

  const colorOf = (key) => ({
    danger: C.danger, warning: C.warning, success: C.success, accent: C.accent
  }[key] ?? C.accent);

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: C.bg, color: C.text, fontFamily: "'Inter', sans-serif" }}>

      <aside style={{
        width: 260, background: C.surface, borderRight: `1px solid ${C.border}`,
        padding: '2rem 1.25rem', display: 'flex', flexDirection: 'column', gap: '2rem',
        position: 'sticky', top: 0, height: '100vh', zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <ShieldCheck size={28} color={C.accent} />
          <span style={{ fontWeight: 800, fontSize: '1.2rem', color: C.text }}>PORTFOLIO RISK ANALYTICS</span>
        </div>

        <nav style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <button onClick={() => setView('user')} style={{
            background: view === 'user' ? `${C.accent}15` : 'transparent',
            border: `1px solid ${view === 'user' ? C.accent : 'transparent'}`,
            color: view === 'user' ? C.accent : C.muted, borderRadius: 12, padding: '0.8rem 1rem', display: 'flex', alignItems: 'center', gap: 12, fontWeight: 600, cursor: 'pointer', transition: '0.2s'
          }}>
            <LayoutDashboard size={18} /> User Mode
          </button>
          <button onClick={() => setView('tech')} style={{
            background: view === 'tech' ? `${C.accent}15` : 'transparent',
            border: `1px solid ${view === 'tech' ? C.accent : 'transparent'}`,
            color: view === 'tech' ? C.accent : C.muted, borderRadius: 12, padding: '0.8rem 1rem', display: 'flex', alignItems: 'center', gap: 12, fontWeight: 600, cursor: 'pointer', transition: '0.2s'
          }}>
            <Settings size={18} /> Technical Mode
          </button>
        </nav>

        <div style={{ marginTop: 'auto' }}>
          <Card style={{ padding: '0.75rem', borderRadius: 12, background: `${C.surface}88` }}>
             <Label>System Health</Label>
             <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.75rem', color: C.success }}>
               <CheckCircle2 size={12} /> ML Research Pipeline
             </div>
          </Card>
        </div>
      </aside>

      <main style={{ flex: 1, padding: '2.5rem', overflowY: 'auto', background: `radial-gradient(circle at top right, ${C.accent}05, transparent)` }}>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2.5rem' }}>
          <div>
            <Badge label={view === 'user' ? "Portfolio Engineering" : "Quantitative Research"} color={C.accent} style={{ marginBottom: 8 }} />
            <h1 style={{ fontSize: '2.2rem', fontWeight: 900, margin: 0, letterSpacing: '-0.02em' }}>
              {view === 'user' ? "Risk Analysis & Forecasting" : "Model Governance Matrix"}
            </h1>
          </div>
          {lastUpdated && <div style={{ fontSize: '0.8rem', color: C.muted }}>Last Scored: {lastUpdated}</div>}
        </div>

        {/* ══ USER MODE ══════════════════════════════════════════════════ */}
        {view === 'user' && (
          <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: '2rem' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              <Card>
                <SectionHeading icon={ListFilter} title="Select Holdings" sub="Choose assets per archetype" />
                <div style={{ maxHeight: '450px', overflowY: 'auto', paddingRight: 8 }}>
                  {Object.entries(groupedTickers).map(([arch, tickers]) => (
                    <div key={arch} style={{ marginBottom: '1.5rem' }}>
                      <Label style={{ color: C.accent, marginBottom: 8 }}>{arch}</Label>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
                        {tickers.map(t => (
                          <div key={t} onClick={() => handleToggleTicker(t)} style={{
                            padding: '0.6rem', background: selectedTickers.includes(t) ? `${C.accent}22` : C.surface,
                            borderRadius: 8, border: `1px solid ${selectedTickers.includes(t) ? C.accent : C.border}`,
                            cursor: 'pointer', fontSize: '0.8rem', fontWeight: 700, textAlign: 'center'
                          }}>{t}</div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </Card>

              <Card>
                <SectionHeading icon={Settings} title="Execution Config" sub="Set quantities and target date" />
                {selectedTickers.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {selectedTickers.map(t => (
                      <div key={t} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                         <span style={{ fontWeight: 700, fontSize: '0.85rem' }}>{t}</span>
                         <input type="number" value={quantities[t] || 0} onChange={(e) => setQuantities({...quantities, [t]: parseInt(e.target.value) || 0})} style={{ width: 60, background: C.bg, border: `1px solid ${C.border}`, color: C.text, borderRadius: 4, padding: 4, textAlign: 'center' }} />
                      </div>
                    ))}
                    <div style={{ borderTop: `1px solid ${C.border}`, pt: '1rem', marginTop: '1rem' }}>
                      <Label>Analysis Target Date</Label>
                      <select value={targetDate} onChange={e => setTargetDate(e.target.value)} style={{ width: '100%', padding: '0.6rem', background: C.card, border: `1px solid ${C.border}`, color: C.text, borderRadius: 8 }}>
                         {validDates.slice().reverse().map(d => <option key={d} value={d}>{d}</option>)}
                      </select>
                    </div>
                    <button onClick={handleRunAnalysis} disabled={loading} style={{ width: '100%', marginTop: '1rem', background: C.accent, color: C.bg, border: 'none', borderRadius: 12, padding: '1rem', fontWeight: 800, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                      {loading ? <RefreshCw size={18} className="spin" /> : <Zap size={18} />} Run ML Synthesis
                    </button>
                  </div>
                ) : <p style={{ color: C.muted, fontSize: '0.8rem', textAlign: 'center' }}>Select tickers above to build.</p>}
              </Card>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {!analysis && !loading && <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.muted }}><Target size={48} style={{ opacity: 0.2, marginBottom: '1rem' }} /></div>}
              {loading && <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><RefreshCw size={48} color={C.accent} className="spin" /></div>}
              {analysis && !loading && (
                <>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
                    <Card glow={colorOf(analysis.m1.signal_color)}>
                       <Label>Shock Signal (M1)</Label>
                       <BigNum color={colorOf(analysis.m1.signal_color)}>{analysis.m1.signal}</BigNum>
                       <div style={{ fontSize: '0.75rem', marginTop: 8 }}>
                          Prob: {(analysis.m1.shock_probability * 100).toFixed(1)}%
                       </div>
                    </Card>
                    <Card glow={analysis.action === 'REDUCE / HEDGE' ? C.danger : analysis.action === 'BUY / ADD' ? C.success : C.accent}>
                       <Label>Advisory Engine</Label>
                       <BigNum color={analysis.action === 'REDUCE / HEDGE' ? C.danger : analysis.action === 'BUY / ADD' ? C.success : C.accent} size="1.4rem">
                         {analysis.action}
                       </BigNum>
                       <div style={{ fontSize: '0.75rem', marginTop: 12 }}>
                          Confidence: {analysis.confidence}
                       </div>
                    </Card>
                    <Card>
                       <Label>Mapped Archetype</Label>
                       <BigNum size="1.4rem" style={{ textTransform: 'capitalize' }}>{analysis.mapped_archetype}</BigNum>
                       <div style={{ fontSize: '0.75rem', marginTop: 12 }}>
                          Risk Profile: {metrics?.ml_models?.M6?.assignments?.[analysis.mapped_archetype] || 'Standard'}
                       </div>
                    </Card>
                  </div>

                  <Card style={{ borderLeft: `6px solid ${C.accent}`, background: `${C.card}ee` }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                      <SectionHeading icon={Activity} title="AI Justification" sub="Machine learning synthesis of current conditions" />
                      <button 
                        onClick={() => setShowDetailedXAI(!showDetailedXAI)}
                        style={{ background: C.surface, border: `1px solid ${C.border}`, color: C.accent, borderRadius: 8, padding: '4px 12px', fontSize: '0.7rem', fontWeight: 700, cursor: 'pointer' }}
                      >
                        {showDetailedXAI ? "Show Summary" : "View Detailed XAI"}
                      </button>
                    </div>

                    <p style={{ lineHeight: 1.7, fontSize: '1rem', color: C.text }}>
                      {showDetailedXAI ? analysis.justification.detailed : analysis.justification.short}
                    </p>

                    {showDetailedXAI && (
                      <div style={{ marginTop: '2rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                         <div style={{ border: `1px solid ${C.border}`, borderRadius: 12, padding: '1rem', background: `${C.bg}44` }}>
                            <Label style={{ color: C.accent }}>Recovery Justification (M2)</Label>
                            <p style={{ fontSize: '0.8rem', color: C.muted, margin: '8px 0' }}>{analysis.m2.band_label}</p>
                            <div style={{ fontSize: '0.7rem', color: C.muted, fontStyle: 'italic', marginTop: 8 }}>{analysis.m2.note}</div>
                         </div>
                         <div style={{ border: `1px solid ${C.border}`, borderRadius: 12, padding: '1rem', background: `${C.bg}44` }}>
                            <Label style={{ color: C.warning }}>Domain Sensitivity (M5)</Label>
                            <p style={{ fontSize: '0.8rem', color: C.muted, margin: '8px 0' }}>{analysis.m4_m5.note}</p>
                            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                               {Object.entries(analysis.m4_m5.coefficients).sort((a,b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 2).map(([dom, coef]) => (
                                 <Badge key={dom} label={`${dom}: ${coef.toFixed(4)}`} color={C.warning} />
                               ))}
                            </div>
                         </div>
                      </div>
                    )}
                  </Card>

                  <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '1rem' }}>
                     <Card>
                       <Label>Relative Performance Trend</Label>
                       <ResponsiveContainer width="100%" height={260}>
                         <AreaChart data={analysis.ticker_chart_data}>
                            <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                            <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}` }} />
                            {selectedTickers.slice(0, 4).map((t, i) => (
                              <Area key={t} type="monotone" dataKey={t} stroke={[C.accent, C.success, C.warning, C.danger][i]} fill={`${C.accent}11`} />
                            ))}
                         </AreaChart>
                       </ResponsiveContainer>
                     </Card>
                     <Card>
                        <Label>Holdings Risk Breakdown</Label>
                        <div style={{ overflowY: 'auto', maxHeight: '230px' }}>
                          <table style={{ width: '100%', fontSize: '0.8rem' }}>
                            <tbody>
                              {analysis.stock_table.map(s => (
                                <tr key={s.ticker} style={{ borderBottom: `1px solid ${C.border}44` }}>
                                  <td style={{ padding: '8px', fontWeight: 700 }}>{s.ticker}</td>
                                  <td style={{ padding: '8px' }}>${s.last_price}</td>
                                  <td style={{ padding: '8px' }}><Badge label={s.shock_today ? "Shock" : "Stable"} color={s.shock_today ? C.danger : C.success} /></td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                     </Card>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* ══ TECHNICAL MODE ═════════════════════════════════════════════ */}
        {view === 'tech' && metrics && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: '1.5rem' }}>
               <Card style={{ minHeight: 350 }}>
                  <SectionHeading icon={Binary} title="Model Health Radar" sub="M1–M6 Performance Fingerprint" />
                  <ResponsiveContainer width="100%" height={260}>
                    <RadarChart cx="50%" cy="50%" outerRadius="80%" data={metrics.radar_data}>
                      <PolarGrid stroke={C.border} />
                      <PolarAngleAxis dataKey="subject" tick={{ fill: C.muted, fontSize: 10 }} />
                      <Radar name="Performance" dataKey="A" stroke={C.accent} fill={C.accent} fillOpacity={0.5} />
                    </RadarChart>
                  </ResponsiveContainer>
               </Card>

               <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
                  {Object.entries(metrics.ml_models).map(([key, m]) => (
                    <Card key={key} style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                       <Label>{key}: {m.name}</Label>
                       <BigNum size="1.2rem">{m.val?.toFixed(2)} <span style={{ fontSize: '0.7rem', color: C.muted }}>{m.unit}</span></BigNum>
                    </Card>
                  ))}
               </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '1.5rem' }}>
               <Card>
                  <SectionHeading icon={Microscope} title="The Research Bridge" sub="Linking EDA insights to Machine Learning outcomes" />
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                     {Object.entries(metrics.hypotheses).map(([id, h]) => (
                       <div key={id} style={{ display: 'flex', gap: 16, borderLeft: `3px solid ${C.accent}`, paddingLeft: 16 }}>
                          <div style={{ background: `${C.accent}11`, borderRadius: 8, padding: '8px 12px', minWidth: 100, textAlign: 'center' }}>
                             <Label>{id}</Label>
                             <div style={{ color: C.accent, fontWeight: 900 }}>{h.result}</div>
                          </div>
                          <div>
                             <p style={{ fontSize: '0.85rem', color: C.text, fontWeight: 600 }}>{h.detail}</p>
                             <div style={{ display: 'flex', gap: 12, marginTop: 4 }}>
                                <Badge label="EDA Consistent" color={C.success} />
                                <Badge label="ML Integrated" color={C.accent} />
                             </div>
                          </div>
                       </div>
                     ))}
                  </div>
               </Card>

               <Card>
                  <SectionHeading icon={Network} title="Statistical Significance" sub="P-Value Heatmap (p < 0.05 targets)" />
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                     {Object.entries(metrics.statistical_tests || {}).map(([key, t]) => (
                       <div key={key}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                             <span style={{ fontSize: '0.8rem', fontWeight: 700 }}>{t.description}</span>
                             <span style={{ fontSize: '0.8rem', color: t.significant ? C.success : C.warning }}>p = {t.p_value?.toFixed(4)}</span>
                          </div>
                          <div style={{ height: 6, background: C.bg, borderRadius: 3, overflow: 'hidden' }}>
                             <div style={{ width: `${Math.max(5, (1 - t.p_value) * 100)}%`, height: '100%', background: t.significant ? C.success : C.warning }} />
                          </div>
                          <p style={{ fontSize: '0.7rem', color: C.muted, marginTop: 4 }}>{t.note}</p>
                       </div>
                     ))}
                  </div>
               </Card>
            </div>

            <Card>
               <SectionHeading icon={FlaskConical} title="EDA Research Gallery" sub="Historical visualizations from the 64-month timeline capture" />
               <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
                  {(metrics.plots || []).map(p => (
                    <div key={p.id} style={{ cursor: 'pointer' }} onClick={() => window.open(API_BASE + p.url)}>
                       <div style={{ background: C.surface, borderRadius: 12, overflow: 'hidden', border: `1px solid ${C.border}`, height: 140 }}>
                          <img src={API_BASE + p.url} alt={p.title} style={{ width: '100%', height: '100%', objectFit: 'cover', opacity: 0.8 }} />
                       </div>
                       <Label style={{ textAlign: 'center', marginTop: 8 }}>{p.title}</Label>
                    </div>
                  ))}
               </div>
            </Card>
          </div>
        )}

      </main>

      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: ${C.bg}; color: ${C.text}; }
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: ${C.bg}; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 3px; }
      `}</style>
    </div>
  );
}
