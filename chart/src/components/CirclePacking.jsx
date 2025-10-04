import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'

// Contract:
// - Fetches endpoint -> { name: 'root', children: [{ name, value, sentiment, count }] }
// - Renders circle packing, circle size ~ value, color ~ sentiment [-1..1] (red->yellow->green)
export default function CirclePacking({ width = 900, height = 700, padding = 4, endpoint = '/api/tickers', subtitle = 'size = Σ hottness.score, color = avg sentiment' }) {
  const svgRef = useRef(null)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let isMounted = true
    setLoading(true)
    setError(null)
    fetch(endpoint)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((json) => {
        if (isMounted) {
          setData(json)
          setError(null)
        }
      })
      .catch((e) => {
        if (isMounted) setError(e.message || 'Failed to load data')
      })
      .finally(() => {
        if (isMounted) setLoading(false)
      })
    return () => {
      isMounted = false
    }
  }, [endpoint])

  const color = useMemo(() => d3.scaleSequential([-1, 1], d3.interpolateRdYlGn), [])

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    if (!data) return

    const root = d3
      .hierarchy(data)
      .sum((d) => (typeof d.value === 'number' ? d.value : 0))
      .sort((a, b) => (b.value || 0) - (a.value || 0))

    const pack = d3.pack().size([width, height]).padding(padding)
    const packed = pack(root)

    // Background
    svg
      .attr('viewBox', [0, 0, width, height].join(' '))
      .attr('width', '100%')
      .attr('height', '100%')
      .style('max-width', `${width}px`)
      .style('height', `${height}px`)
      .style('background', 'transparent')

    const nodes = svg
      .append('g')
      .attr('transform', `translate(0,0)`)
      .selectAll('g')
      .data(packed.leaves())
      .join('g')
      .attr('transform', (d) => `translate(${d.x},${d.y})`)

    nodes
      .append('circle')
      .attr('r', (d) => d.r)
      .attr('fill', (d) => color(d.data?.sentiment ?? 0))
      .attr('fill-opacity', 0.9)
      .attr('stroke', '#333')
      .attr('stroke-opacity', 0.25)

    // Label: name
    const label = nodes
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .style('pointer-events', 'none')

    label
      .append('tspan')
      .text((d) => d.data?.name ?? '')
      .attr('fill', '#111')
      .style('font-weight', 600)
      .style('font-family', 'system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif')
      .style('font-size', (d) => `${Math.max(10, Math.min(24, d.r / 2))}px`)

    // Sub-label: value/count
    label
      .append('tspan')
      .attr('x', 0)
      .attr('dy', '1.5rem')
      .text((d) => {
        const cnt = d.data?.count ?? 0
        return ``
      })
      .attr('fill', '#222')
      .style('font-weight', 400)
      .style('font-size', (d) => `${Math.max(8, Math.min(14, d.r / 4))}px`, 'line-height', '3em')

  }, [data, width, height, padding, color])

  return (
    <div style={{ display: 'grid', gap: 12 }}>
      {loading && <div>Загрузка данных…</div>}
      {error && (
        <div style={{ color: 'crimson' }}>
          Ошибка загрузки: {String(error)} (показываются мок-данные, если API не настроен)
        </div>
      )}
      <svg ref={svgRef} />
    </div>
  )
}
