import React from 'react'

export const Controls = ({ children, className = '' }) => <div className={'controls ' + className}>{children}</div>

export function Slider({ label, min, max, step = 1, value, onChange, fmt = v => v }) {
  return (
    <label className="ctl slider">
      <span className="lbl">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(+e.target.value)} />
      <span className="val">{fmt(value)}</span>
    </label>
  )
}

export function Select({ label, options, value, onChange }) {
  return (
    <label className="ctl select">
      <span className="lbl">{label}</span>
      <select value={value} onChange={e => onChange(e.target.value)}>
        {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
    </label>
  )
}

export function Toggle({ label, value, onChange }) {
  return (
    <label className="ctl toggle">
      <input type="checkbox" checked={value} onChange={e => onChange(e.target.checked)} />
      <span className="track" />
      <span className="lbl">{label}</span>
    </label>
  )
}

export const Btn = ({ children, onClick, primary, className = '' }) => (
  <button className={`btn ${primary ? 'primary' : ''} ${className}`} onClick={onClick}>{children}</button>
)

/** Segmented control: options = [[value,label],...] */
export function Seg({ options, value, onChange }) {
  return (
    <div className="seg-bar">
      {options.map(([v, l]) => (
        <button key={v} className={'seg' + (v === value ? ' on' : '')} onClick={() => onChange(v)}>{l}</button>
      ))}
    </div>
  )
}

export const Readout = ({ children }) => <div className="readout">{children}</div>

/** Tab-style sub views inside a lab. views = [{id,label,render}] */
export function SubViews({ views, initial }) {
  const [id, setId] = React.useState(initial || views[0].id)
  const v = views.find(x => x.id === id) || views[0]
  return (
    <>
      <Seg options={views.map(x => [x.id, x.label])} value={v.id} onChange={setId} />
      <div className="sub-body" key={v.id}>{v.render()}</div>
    </>
  )
}

export const Legend = ({ items }) => (
  <div className="legend">
    {items.map(([c, l]) => <span key={l}><i style={{ background: c }} />{l}</span>)}
  </div>
)
