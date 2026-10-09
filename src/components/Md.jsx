import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeRaw from 'rehype-raw'
import rehypeKatex from 'rehype-katex'
import CodeBlock from './CodeBlock.jsx'

const remark = [remarkGfm, remarkMath]
const rehype = [rehypeRaw, [rehypeKatex, { throwOnError: false, strict: false }]]

const components = {
  code({ className, children, node, ...rest }) {
    const m = /language-(\w+)/.exec(className || '')
    const text = String(children)
    if (m || text.includes('\n')) return <CodeBlock lang={m ? m[1] : 'text'} src={text.replace(/\n$/, '')} />
    return <code {...rest}>{children}</code>
  },
  pre: ({ children }) => <>{children}</>,
  table: ({ children }) => <div className="tbl"><table>{children}</table></div>
}
const inlineComponents = { ...components, p: ({ children }) => <>{children}</> }

/** Markdown + KaTeX renderer. `inline` unwraps the outer paragraph. */
export default function Md({ children, inline = false }) {
  return (
    <ReactMarkdown remarkPlugins={remark} rehypePlugins={rehype} components={inline ? inlineComponents : components}>
      {children}
    </ReactMarkdown>
  )
}
