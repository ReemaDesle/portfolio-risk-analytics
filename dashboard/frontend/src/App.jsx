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
  Settings, Info, ChevronRight, CheckCircle2, FlaskConical, BarChart3, Database
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
  const [view, setView]                     = useState('user'); // 'user' | 'tech'
  const [groupedTickers, setGroupedTickers] = useState({});
  const [leadTickers, setLeadTickers]       = useState({});
  const [validDates, setValidDates]         = useState([]);
  
  // Selection state
  const [selectedTickers, setSelectedTickers] = useState([]);
  const [quantities, setQuantities]           = useState({});
  const [targetDate, setTargetDate]           = useState('');
  
  // Analysis Data
  const [analysis, setAnalysis]             = useState(null);
  const [metrics, setMetrics]               = useState(null);
  const [arima, setArima]                   = useState(null);
  
  // UI State
  const [loading, setLoading]               = useState(false);
  const [lastUpdated, setLastUpdated]       = useState(null);
  const [showDetailedXAI, setShowDetailedXAI] = useState(false);

  // Fetch initial data
  useEffect(() => {
    setLoading(true);
    axios.get(`${API_BASE}/api/tickers`)
      .then(r => {
        setGroupedTickers(r.data.grouped);
        setLeadTickers(r.data.lead_tickers);
        setValidDates(r.data.valid_dates);
        if (r.data.valid_dates.length) setTargetDate(r.data.valid_dates[r.data.valid_dates.length - 1]);
      })
      .catch(console.error);

    axios.get(`${API_BASE}/api/metrics`)
      .then(r => setMetrics(r.data))
      .catch(console.error);
      
    setLoading(false);
  }, []);

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
      
      // Also fetch arima for the mapped archetype's lead ticker
      const arimaResp = await axios.get(`${API_BASE}/api/arima/${resp.data.mapped_archetype}`);
      setArima(arimaResp.data);
      
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  // Helper for status colours
  const colorOf = (key) => ({
    danger: C.danger, warning: C.warning, success: C.success, accent: C.accent
  }[key] ?? C.accent);

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: C.bg, color: C.text,
      fontFamily: "'Inter', sans-serif" }}>

      {/* ── Sidebar Nav ── */}
      <aside style={{
        width: 260, background: C.surface, borderRight: `1px solid ${C.border}`,
        padding: '2rem 1.25rem', display: 'flex', flexDirection: 'column', gap: '2rem',
        position: 'sticky', top: 0, height: '100vh', zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <ShieldCheck size={28} color={C.accent} />
          <span style={{ fontWeight: 800, fontSize: '1.2rem', color: C.text }}>ANTIGRAVITY</span>
        </div>

        <nav style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <button 
            onClick={() => setView('user')}
            style={{
              background: view === 'user' ? `${C.accent}15` : 'transparent',
              border: `1px solid ${view === 'user' ? C.accent : 'transparent'}`,
              color: view === 'user' ? C.accent : C.muted,
              borderRadius: 12, padding: '0.8rem 1rem', display: 'flex', alignItems: 'center', gap: 12,
              fontWeight: 600, cursor: 'pointer', transition: '0.2s'
            }}
          >
            <LayoutDashboard size={18} /> User Mode
          </button>
          <button 
            onClick={() => setView('tech')}
            style={{
              background: view === 'tech' ? `${C.accent}15` : 'transparent',
              border: `1px solid ${view === 'tech' ? C.accent : 'transparent'}`,
              color: view === 'tech' ? C.accent : C.muted,
              borderRadius: 12, padding: '0.8rem 1rem', display: 'flex', alignItems: 'center', gap: 12,
              fontWeight: 600, cursor: 'pointer', transition: '0.2s'
            }}
          >
            <Settings size={18} /> Technical Mode
          </button>
        </nav>

        <div style={{ marginTop: 'auto' }}>
          <Card style={{ padding: '0.75rem', borderRadius: 12, background: `${C.surface}88` }}>
             <Label>System Health</Label>
             <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.75rem', color: C.success }}>
               <CheckCircle2 size={12} /> ML Pipeline Ready
             </div>
          </Card>
        </div>
      </aside>

      {/* ── Main Content Area ── */}
      <main style={{ flex: 1, padding: '2.5rem', overflowY: 'auto', background: `radial-gradient(circle at top right, ${C.accent}05, transparent)` }}>
        
        {/* Header Section */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2.5rem' }}>
          <div>
            <Badge label={view === 'user' ? "Portfolio Engineering" : "Quantitative Research"} color={C.accent} style={{ marginBottom: 8 }} />
            <h1 style={{ fontSize: '2.2rem', fontWeight: 900, margin: 0, letterSpacing: '-0.02em' }}>
              {view === 'user' ? "Risk Analysis & Forecasting" : "Model Governance Matrix"}
            </h1>
            <p style={{ color: C.muted, fontSize: '0.95rem', marginTop: 4 }}>
              Integrated ML suite (M1–M6) tracking 27 global tickers and 3 news sentiment domains.
            </p>
          </div>
          {lastUpdated && <div style={{ fontSize: '0.8rem', color: C.muted }}>Last Scored: {lastUpdated}</div>}
        </div>

        {/* ══ USER MODE ══════════════════════════════════════════════════ */}
        {view === 'user' && (
          <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: '2rem' }}>
            
            {/* Left: Input Builder */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              <Card>
                <SectionHeading icon={ListFilter} title="Select Holdings" sub="Choose assets per archetype" />
                
                <div style={{ maxHeight: '450px', overflowY: 'auto', paddingRight: 8 }}>
                  {Object.entries(groupedTickers).map(([arch, tickers]) => (
                    <div key={arch} style={{ marginBottom: '1.5rem' }}>
                      <Label style={{ color: C.accent, marginBottom: 8 }}>{arch}</Label>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
                        {tickers.map(t => (
                          <div 
                            key={t}
                            onClick={() => handleToggleTicker(t)}
                            style={{
                              padding: '0.6rem', background: selectedTickers.includes(t) ? `${C.accent}22` : C.surface,
                              borderRadius: 8, border: `1px solid ${selectedTickers.includes(t) ? C.accent : C.border}`,
                              cursor: 'pointer', fontSize: '0.8rem', fontWeight: 700, textAlign: 'center', transition: '0.1s'
                            }}
                          >
                            {t}
                          </div>
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
                         <input 
                            type="number" 
                            value={quantities[t] || 0}
                            onChange={(e) => setQuantities({...quantities, [t]: parseInt(e.target.value) || 0})}
                            style={{ width: 60, background: C.bg, border: `1px solid ${C.border}`, color: C.text, borderRadius: 4, padding: 4, textAlign: 'center' }}
                         />
                      </div>
                    ))}
                    
                    <div style={{ borderTop: `1px solid ${C.border}`, pt: '1rem', marginTop: '1rem' }}>
                      <Label>Analysis Target Date</Label>
                      <select 
                        value={targetDate} 
                        onChange={e => setTargetDate(e.target.value)}
                        style={{ width: '100%', padding: '0.6rem', background: C.card, border: `1px solid ${C.border}`, color: C.text, borderRadius: 8 }}
                      >
                         {validDates.slice().reverse().map(d => <option key={d} value={d}>{d}</option>)}
                      </select>
                    </div>

                    <button 
                      onClick={handleRunAnalysis}
                      disabled={loading}
                      style={{ 
                        width: '100%', marginTop: '1rem', background: C.accent, color: C.bg, 
                        border: 'none', borderRadius: 12, padding: '1rem', fontWeight: 800,
                        cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8
                      }}
                    >
                      {loading ? <RefreshCw size={18} className="spin" /> : <Zap size={18} />}
                      Run ML Synthesis
                    </button>
                  </div>
                ) : (
                  <p style={{ color: C.muted, fontSize: '0.8rem', textAlign: 'center' }}>Select tickers above to start building.</p>
                )}
              </Card>
            </div>

            {/* Right: Analysis Dashboard */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {!analysis && !loading && (
                <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.muted }}>
                  <div style={{ textAlign: 'center' }}>
                     <Target size={48} style={{ opacity: 0.2, marginBottom: '1rem' }} />
                     <p>Prepare your holdings on the left to view neural risk forecasts.</p>
                  </div>
                </div>
              )}

              {loading && (
                <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                   <div style={{ textAlign: 'center' }}>
                      <RefreshCw size={48} color={C.accent} className="spin" style={{ marginBottom: '1rem' }} />
                      <p>Running multi-model inference pipeline...</p>
                   </div>
                </div>
              )}

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
                          Risk Profile: {metrics?.ml_models.M6.assignments[analysis.mapped_archetype] || 'Standard'}
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
                            <p style={{ fontSize: '0.8rem', color: C.muted, margin: '8px 0' }}>{analysis.recovery.justification}</p>
                            <Badge label={analysis.recovery.band_label} color={C.accent} />
                         </div>
                         <div style={{ border: `1px solid ${C.border}`, borderRadius: 12, padding: '1rem', background: `${C.bg}44` }}>
                            <Label style={{ color: C.warning }}>Domain Sensitivity (M5)</Label>
                            <p style={{ fontSize: '0.8rem', color: C.muted, margin: '8px 0' }}>{analysis.m4_m5.note}</p>
                            <div style={{ display: 'flex', gap: 12 }}>
                               {analysis.m4_m5.ranked_domains.slice(0, 2).map(d => (
                                 <Badge key={d.domain} label={`${d.domain}: ${d.coefficient.toFixed(4)}`} color={C.warning} />
                               ))}
                            </div>
                         </div>
                      </div>
                    )}
                  </Card>

                  <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '1rem' }}>
                     <Card>
                       <Label style={{ marginBottom: '1rem' }}>Relative Performance Trend (Last 60d)</Label>
                       <ResponsiveContainer width="100%" height={260}>
                         <AreaChart data={analysis.ticker_chart_data}>
                            <defs>
                              <linearGradient id="gUser" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={C.accent} stopOpacity={0.2} />
                                <stop offset="95%" stopColor={C.accent} stopOpacity={0} />
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                            <XAxis dataKey="date" hide />
                            <YAxis stroke={C.muted} fontSize={10} domain={['auto', 'auto']} tickFormatter={v => `$${v}`} />
                            <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8 }} />
                            {selectedTickers.slice(0, 4).map((t, i) => (
                              <Area key={t} type="monotone" dataKey={t} stroke={[C.accent, C.success, C.warning, C.danger][i]} fill={`url(#gUser)`} strokeWidth={2} dot={false} />
                            ))}
                         </AreaChart>
                       </ResponsiveContainer>
                     </Card>

                     <Card>
                        <Label style={{ marginBottom: '1rem' }}>Holdings Risk Breakdown</Label>
                        <div style={{ overflowY: 'auto', maxHeight: '260px' }}>
                          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                            <thead>
                              <tr style={{ borderBottom: `1px solid ${C.border}`, textAlign: 'left' }}>
                                <th style={{ padding: '8px', color: C.muted }}>Ticker</th>
                                <th style={{ padding: '8px', color: C.muted }}>Price</th>
                                <th style={{ padding: '8px', color: C.muted }}>Hedge</th>
                              </tr>
                            </thead>
                            <tbody>
                              {analysis.stock_table.map(s => (
                                <tr key={s.ticker} style={{ borderBottom: `1px solid ${C.border}44` }}>
                                  <td style={{ padding: '10px 8px', fontWeight: 700 }}>{s.ticker}</td>
                                  <td style={{ padding: '10px 8px' }}>${s.last_price}</td>
                                  <td style={{ padding: '10px 8px' }}>
                                    {s.shock_today ? <Badge label="⚠" color={C.danger} /> : <Badge label="OK" color={C.success} />}
                                  </td>
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
            
            {/* Model Summary & KPI Strip */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
               <Card>
                 <Label>Shock Pipeline (M1)</Label>
                 <BigNum size="1.4rem">AUC: {metrics.ml_models.M1.auc_roc.toFixed(3)}</BigNum>
                 <p style={{ fontSize: '0.7rem', color: C.muted, mt: 4 }}>Precision-tuned (F1: {metrics.ml_models.M1.f1_calibrated})</p>
               </Card>
               <Card>
                 <Label>Recovery MAE (M2)</Label>
                 <BigNum size="1.4rem">{metrics.ml_models.M2.mae_days} Days</BigNum>
                 <p style={{ fontSize: '0.7rem', color: C.muted, mt: 4 }}>Error margin per shock event</p>
               </Card>
               <Card>
                 <Label>Tech Sensitivity (M5)</Label>
                 <BigNum size="1.4rem">{metrics.hypotheses.H3_portfolio_comparison.tech_sensitivity.toFixed(2)}</BigNum>
                 <p style={{ fontSize: '0.7rem', color: C.muted, mt: 4 }}>Coefficient magnitude (Normalized)</p>
               </Card>
               <Card>
                 <Label>Cluster Profile (M6)</Label>
                 <BigNum size="1.4rem">S={metrics.ml_models.M6.silhouette.toFixed(3)}</BigNum>
                 <p style={{ fontSize: '0.7rem', color: C.muted, mt: 4 }}>Silhouette Separability Score</p>
               </Card>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: '1.5rem' }}>
               {/* Metrics Deep-Dive */}
               <Card>
                  <SectionHeading icon={BarChart3} title="Statistical Significance" sub="Comparison against Null Hypotheses" />
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                     <thead>
                        <tr style={{ borderBottom: `1px solid ${C.border}`, textAlign: 'left' }}>
                           <th style={{ padding: '12px', color: C.muted }}>Validation Test</th>
                           <th style={{ padding: '12px', color: C.muted }}>P-Value</th>
                           <th style={{ padding: '12px', color: C.muted }}>Outcome</th>
                        </tr>
                     </thead>
                     <tbody>
                        {Object.entries(metrics.statistical_tests).map(([key, t]) => (
                          <tr key={key} style={{ borderBottom: `1px solid ${C.border}44` }}>
                            <td style={{ padding: '16px 12px' }}>
                               <div style={{ fontWeight: 700 }}>{t.description}</div>
                               <div style={{ fontSize: '0.7rem', color: C.muted }}>{t.note}</div>
                            </td>
                            <td style={{ padding: '16px 12px', fontFamily: 'monospace', color: t.significant ? C.success : C.warning }}>
                               {t.p_value.toFixed(4)}
                            </td>
                            <td style={{ padding: '16px 12px' }}>
                               {t.significant ? <Badge label="SIG" color={C.success} /> : <Badge label="INSIG" color={C.warning} />}
                            </td>
                          </tr>
                        ))}
                     </tbody>
                  </table>
               </Card>

               {/* ARIMA Forecast */}
               <Card>
                  <SectionHeading icon={TrendingUp} title="Neural Forecasting (ARIMAX)" sub={`Predictive signal for ${arima?.ticker || 'archetype lead'}`} />
                  {arima?.available ? (
                    <div>
                      <div style={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center', background: `${C.bg}66`, borderRadius: 12, border: `1px solid ${C.border}`, mb: '1rem' }}>
                         {arima.plot_available ? (
                           <div style={{ textAlign: 'center' }}>
                              <p style={{ fontSize: '0.8rem', color: C.accent }}>Live forecast chart available in reports directory.</p>
                              <p style={{ color: C.muted, fontSize: '0.7rem' }}>`arimax_{arima.ticker}_lag1_z.png`</p>
                           </div>
                         ) : (
                           <div style={{ textAlign: 'center', color: C.muted }}>Executing forecast calculations...</div>
                         )}
                      </div>
                      <Label>Model Inference</Label>
                      <p style={{ fontSize: '0.82rem', lineHeight: 1.6, color: C.text, margin: '8px 0' }}>{arima.note}</p>
                      <div style={{ background: `${C.accent}11`, p: '0.75rem', borderRadius: 8, mt: '1rem', padding: 8 }}>
                         <div style={{ fontSize: '0.75rem', fontWeight: 700, color: C.accent }}>EXOGENOUS SIGNAL: FinBERT sentiment</div>
                         <div style={{ fontSize: '0.65rem', color: C.muted }}>Lag: 1 Day · Model: ARIMA(1, 1, 1)</div>
                      </div>
                    </div>
                  ) : (
                    <div style={{ textAlign: 'center', padding: '2rem', color: C.muted }}>
                       Select a portfolio archetype in User Mode to prime the lead ticker forecast.
                    </div>
                  )}
               </Card>
            </div>

            <Card>
              <SectionHeading icon={Database} title="Integrations & Data Feeds" sub="Endpoints for Power BI and external dashboards" />
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
                 {[
                   { label: "Master Data Feed", url: "/api/data-feed/master", icon: <Database /> },
                   { label: "Hypothesis Funnel", url: "/api/data-feed/hypothesis-results", icon: <FlaskConical /> },
                   { label: "Model Governance", url: "/api/data-feed/model-metrics", icon: <BarChart3 /> }
                 ].map(feed => (
                   <div key={feed.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: C.surface, p: '1rem', borderRadius: 12, border: `1px solid ${C.border}`, padding: '1rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                         <span style={{ color: C.accent }}>{feed.icon}</span>
                         <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>{feed.label}</span>
                      </div>
                      <a href={`${API_BASE}${feed.url}`} download style={{ textDecoration: 'none', color: C.accent, fontSize: '0.75rem', fontWeight: 800 }}>GET CSV</a>
                   </div>
                 ))}
              </div>
            </Card>

            <Card style={{ textAlign: 'center', border: `1px dashed ${C.border}`, background: 'transparent' }}>
               <Label>External Visualization</Label>
               <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.muted }}>
                  <div style={{ textAlign: 'center' }}>
                     <Globe size={32} style={{ marginBottom: 12, opacity: 0.3 }} />
                     <p>Power BI Public Web Embedding Placeholder</p>
                     <p style={{ fontSize: '0.75rem' }}>Paste your <code>&lt;iframe&gt;</code> code from Power BI Service here to link interactive dashboards.</p>
                  </div>
               </div>
            </Card>
          </div>
        )}

      </main>

      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: ${C.bg}; }
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: ${C.bg}; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 3px; }
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
      `}</style>
    </div>
  );
}
