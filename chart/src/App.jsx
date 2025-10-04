import { useState } from 'react'
import './App.css'
import CirclePacking from './components/CirclePacking'

function App() {
  const [tab, setTab] = useState('tickers') // 'tickers' | 'companies'

  const endpoint = tab === 'tickers' ? '/api/tickers' : '/api/companies'
  const title = tab === 'tickers' ? 'Tickers' : 'Companies'

  return (
    <div style={{ padding: 16 }}>
      {/* Вкладки-переключатели */}
      <div className="tabs" role="tablist" aria-label="Тип данных">
        <button
          role="tab"
          aria-selected={tab === 'tickers'}
          className={`tab-btn ${tab === 'tickers' ? 'is-active' : ''}`}
          onClick={() => setTab('tickers')}
          style={{marginRight: 15}}
        >
          Tickers
        </button>
        <button
          role="tab"
          aria-selected={tab === 'companies'}
          className={`tab-btn ${tab === 'companies' ? 'is-active' : ''}`}
          onClick={() => setTab('companies')}
          style={{marginRight: 15}}
        >
          Companies
        </button>
      </div>

      <CirclePacking
        width={900}
        height={700}
        padding={4}
        endpoint={endpoint}
        subtitle={`${title}: size = Σ hottness.score, color = avg sentiment`}
      />
    </div>
  )
}

export default App
