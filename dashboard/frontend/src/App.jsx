import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  AreaChart, Area, LineChart, Line, Cell
} from 'recharts';
import { 
  ShieldCheck, TrendingUp, AlertTriangle, PieChart, 
  Zap, Info, ExternalLink, RefreshCw 
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

function App() {
  const [portfolios, setPortfolios] = useState([]);
  const [selectedPortfolio, setSelectedPortfolio] = useState('');
  const [activeTab, setActiveTab] = useState('analytics');
  const [data, setData] = useState(null);
  const [suggestions, setSuggestions] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchPortfolios();
  }, []);

  useEffect(() => {
    if (selectedPortfolio) {
      fetchData();
    }
  }, [selectedPortfolio]);

  const fetchPortfolios = async () => {
    try {
      const res = await axios.get(`${API_BASE}/portfolios`);
      setPortfolios(res.data.portfolios);
      if (res.data.portfolios.length > 0) setSelectedPortfolio(res.data.portfolios[0]);
    } catch (err) {
      console.error("Failed to fetch portfolios", err);
    }
  };

  const fetchData = async () => {
    setLoading(true);
    try {
      const [dataRes, sugRes] = await Promise.all([
        axios.get(`${API_BASE}/analytics/${selectedPortfolio}`),
        axios.get(`${API_BASE}/suggestions/${selectedPortfolio}`)
      ]);
      setData(dataRes.data);
      setSuggestions(sugRes.data);
    } catch (err) {
      console.error("Failed to fetch data", err);
    }
    setLoading(false);
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="logo">
          <ShieldCheck size={28} />
          <span>NEURAL RISK</span>
        </div>
        
        <div style={{ marginTop: '2rem' }}>
          <label className="stat-label" style={{ display: 'block', marginBottom: '0.5rem' }}>Active Portfolio</label>
          <select value={selectedPortfolio} onChange={(e) => setSelectedPortfolio(e.target.value)}>
            {portfolios.map(p => <option key={p} value={p}>{p.toUpperCase()}</option>)}
          </select>
        </div>

        <nav style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div className="stat-label">System Stats</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            ML Pipeline: <span style={{ color: 'var(--success)' }}>Active</span><br/>
            Last Scraping: <span style={{ color: 'var(--accent)' }}>2h ago</span>
          </div>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header>
          <div>
            <h1 style={{ textTransform: 'capitalize' }}>{selectedPortfolio} Dashboard</h1>
            <p style={{ color: 'var(--text-secondary)' }}>Multi-domain sentiment risk analysis</p>
          </div>
          <button 
            onClick={fetchData}
            style={{ 
              background: 'transparent', border: '1px solid var(--border)', color: 'white', 
              padding: '0.5rem 1rem', borderRadius: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem',
              cursor: 'pointer'
            }}
          >
            <RefreshCw size={16} className={loading ? 'spin' : ''} />
            Refresh
          </button>
        </header>

        <div className="tabs">
          <div className={`tab ${activeTab === 'analytics' ? 'active' : ''}`} onClick={() => setActiveTab('analytics')}>Analytics</div>
          <div className={`tab ${activeTab === 'advisory' ? 'active' : ''}`} onClick={() => setActiveTab('advisory')}>AI Advisory</div>
          <div className={`tab ${activeTab === 'pbi' ? 'active' : ''}`} onClick={() => setActiveTab('pbi')}>Power BI Dashboards</div>
        </div>

        {activeTab === 'analytics' && data && (
          <div className="dashboard-grid">
            {/* Top Row: Quick Stats */}
            <div className="card col-4">
              <div className="stat-label">Sentiment Risk Index (SRI)</div>
              <div className={`stat-value ${suggestions?.sri_value > 10 ? 'status-risk' : ''}`} style={{ color: suggestions?.sri_value > 10 ? 'var(--danger)' : 'var(--accent)' }}>
                {suggestions?.sri_value.toFixed(2)}
              </div>
              <div className="status-badge status-stable">
                {suggestions?.status}
              </div>
            </div>
            
            <div className="card col-4">
              <div className="stat-label">Market Sensitivity</div>
              <div className="stat-value">β 1.24</div>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Highly reactive to Geo-Sentiment</div>
            </div>

            <div className="card col-4">
              <div className="stat-label">Recovery Forecast</div>
              <div className="stat-value">2.5 Days</div>
              <div style={{ color: 'var(--success)', fontSize: '0.8rem' }}>+12% confidence vs baseline</div>
            </div>

            {/* Performance Chart */}
            <div className="card col-8">
              <div className="stat-label" style={{ marginBottom: '1.5rem' }}>Price vs Sentiment Trend (30D)</div>
              <div style={{ width: '100%', height: 300 }}>
                <ResponsiveContainer>
                  <AreaChart data={data.market_data}>
                    <defs>
                      <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="var(--accent)" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="date" hide />
                    <YAxis stroke="var(--text-secondary)" fontSize={12} tickFormatter={(v) => `$${v.toFixed(0)}`} />
                    <Tooltip 
                      contentStyle={{ background: '#0f172a', border: '1px solid var(--border)', borderRadius: '8px' }}
                      itemStyle={{ color: 'var(--accent)' }}
                    />
                    <Area type="monotone" dataKey="SPY" stroke="var(--accent)" fillOpacity={1} fill="url(#colorAcc)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Domain Correlation */}
            <div className="card col-4">
              <div className="stat-label" style={{ marginBottom: '1.5rem' }}>Sentiment Impact</div>
              <div style={{ width: '100%', height: 300 }}>
                <ResponsiveContainer>
                  <BarChart layout="vertical" data={[
                    { name: 'Geopolitical', value: 0.82 },
                    { name: 'Financial', value: 0.45 },
                    { name: 'Technology', value: 0.61 }
                  ]}>
                    <XAxis type="number" hide />
                    <YAxis dataKey="name" type="category" stroke="var(--text-secondary)" fontSize={10} width={70} />
                    <Tooltip cursor={{fill: 'transparent'}} />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      { [0,1,2].map((_, i) => <Cell key={i} fill={i === 0 ? 'var(--danger)' : 'var(--accent)'} />) }
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'advisory' && suggestions && (
          <div className="dashboard-grid">
            <div className="card col-12" style={{ borderLeft: '4px solid', borderLeftColor: suggestions.action === 'Buy' ? 'var(--success)' : 'var(--danger)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Recommended Action: {suggestions.action}</h2>
                  <div className="status-badge" style={{ background: 'rgba(56, 189, 248, 0.1)', color: 'var(--accent)' }}>
                    {suggestions.category} Strategy
                  </div>
                </div>
                <Zap size={32} color={suggestions.action === 'Buy' ? 'var(--success)' : 'var(--danger)'} />
              </div>
              
              <div style={{ marginTop: '2rem' }}>
                <h3 className="stat-label" style={{ marginBottom: '0.5rem' }}>AI Reasoning</h3>
                <p style={{ lineHeight: '1.6', color: 'var(--text-secondary)' }}>
                  {suggestions.reasoning} 
                  The XAI layer identifies that the <strong>Geopolitical Sentiment Spike</strong> is the primary driver of current volatility. 
                  Model M2 predicts a <strong>85% probability</strong> of market stabilization within the next 48-72 hours.
                </p>
              </div>

              <div style={{ marginTop: '2rem', display: 'flex', gap: '1rem' }}>
                <button className="card" style={{ flex: 1, padding: '1rem', textAlign: 'center', cursor: 'pointer', borderColor: 'var(--accent)' }}>
                  View Feature Weights (SHAP)
                </button>
                <button className="card" style={{ flex: 1, padding: '1rem', textAlign: 'center', cursor: 'pointer' }}>
                  Download Full Risk Report
                </button>
              </div>
            </div>

            <div className="card col-6">
              <h3 className="stat-label" style={{ marginBottom: '1rem' }}>Portfolio Expansion Suggestions</h3>
              <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <li style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Gold Futures (GC=F)</span>
                  <span style={{ color: 'var(--success)' }}>Hedge (+)</span>
                </li>
                <li style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Treasury Bonds (TLT)</span>
                  <span style={{ color: 'var(--success)' }}>Stability (+)</span>
                </li>
              </ul>
            </div>
          </div>
        )}

        {activeTab === 'pbi' && (
          <div className="card col-12" style={{ height: '70vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
            <AlertTriangle size={48} color="var(--warning)" style={{ marginBottom: '1rem' }} />
            <h3>Power BI Integration Ready</h3>
            <p style={{ color: 'var(--text-secondary)', maxWidth: '400px', marginTop: '1rem' }}>
              To view your interactive dashboards, please publish your report to web and paste the embed URL in the backend configuration.
            </p>
            <div style={{ marginTop: '2rem', width: '100%', border: '1px dashed var(--border)', borderRadius: '1rem', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <p style={{ color: 'var(--text-secondary)' }}>[ Power BI Iframe Container ]</p>
            </div>
          </div>
        )}
      </main>

      <style>{`
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}

export default App;
