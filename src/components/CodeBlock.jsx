import React, { useState } from 'react'
import { Highlight } from 'prism-react-renderer'

const theme = {
  plain: { color: '#d4d8ec', backgroundColor: 'transparent' },
  styles: [
    { types: ['comment', 'prolog', 'doctype', 'cdata'], style: { color: '#5d6584', fontStyle: 'italic' } },
    { types: ['keyword', 'operator', 'boolean', 'tag'], style: { color: '#c792ea' } },
    { types: ['builtin', 'class-name', 'constant', 'symbol'], style: { color: '#7fdbca' } },
    { types: ['function'], style: { color: '#82aaff' } },
    { types: ['string', 'char', 'attr-value'], style: { color: '#c3e88d' } },
    { types: ['number'], style: { color: '#f78c6c' } },
    { types: ['punctuation'], style: { color: '#9aa1c0' } },
    { types: ['macro', 'directive', 'important'], style: { color: '#ffcb6b' } }
  ]
}

const LANG = { python: 'Python', py: 'Python', cpp: 'CUDA C++', cuda: 'CUDA C++', text: 'Text', bash: 'Shell', ptx: 'PTX' }

export default function CodeBlock({ src, lang = 'python', title, note }) {
  const [copied, setCopied] = useState(false)
  const pl = lang === 'cuda' ? 'cpp' : lang === 'py' ? 'python' : lang
  const copy = () => {
    navigator.clipboard && navigator.clipboard.writeText(src)
    setCopied(true); setTimeout(() => setCopied(false), 1200)
  }
  return (
    <div className="codeblock">
      <div className="cb-head">
        <span className="cb-lang">{LANG[lang] || lang}</span>
        <strong>{title}</strong>
        <button className="copy" onClick={copy}>{copied ? 'Copied' : 'Copy'}</button>
      </div>
      {note && <p className="cb-note">{note}</p>}
      <Highlight theme={theme} code={src} language={pl}>
        {({ tokens, getLineProps, getTokenProps }) => (
          <pre>
            <code>
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({ line })}>
                  {line.map((token, k) => <span key={k} {...getTokenProps({ token })} />)}
                </div>
              ))}
            </code>
          </pre>
        )}
      </Highlight>
    </div>
  )
}
